import asyncio
import os
import re
from typing import Any, AsyncGenerator, List, Dict, Optional, Set, Tuple
import pinecone
import logging
from dataclasses import dataclass
from openai import OpenAI
import json
import time
from dotenv import load_dotenv

from ingestion.models import ExtractedImage, PaperChunk
from ingestion.processor import OPENAI_API_KEY
from ingestion.section import Section
from ingestion.store import R2ImageStore
from rag.models import Figure, StructuredResponse, TimingStats

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TEI_SERVER = os.getenv("TEI_SERVER") or "http://localhost:8000"
TEI_TOKEN = os.getenv("TEI_TOKEN") or "dummy_token"
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY") or "dummy"
PINECONE_HOST = os.getenv("PINECONE_HOST") or "http://localhost:9090"


def sanitize_metadata(value: Any) -> Any:
    """Convert metadata values to chroma-compatible primitives"""
    if value is None:
        return ""  # Convert None to empty string
    elif isinstance(value, (str, int, float, bool)):
        return value
    elif isinstance(value, (list, dict)):
        return json.dumps(value)  # Convert complex types to JSON strings
    else:
        return str(value)  # Convert anything else to string


def prepare_metadata(data: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively sanitize all metadata values"""
    return {k: sanitize_metadata(v) for k, v in data.items()}


@dataclass
class RetrievedContext:
    """Represents a retrieved chunk with its full context"""

    chunk: PaperChunk
    paper_metadata: Dict
    score: float
    section: Optional[Section] = None

    @property
    def relevant_images(self) -> List[Dict]:
        """Get images from this chunk's section"""
        images = self.paper_metadata.get("images", [])
        if not self.section:
            return []

        return [img for img in images if img["section_id"] == self.section.get_id()]


@dataclass
class GeneratedResponse:
    """Response generated from retrieved contexts"""

    answer: str
    citations: List[Dict]
    confidence: float
    sections_referenced: List[str]
    referenced_images: List[str]


class RAGPipeline:
    def __init__(self, collection_name: str = "papers", batch_size: int = 32):
        # Initialize Chroma client with auth
        self.pinecone_client = pinecone.Pinecone(
            api_key=PINECONE_API_KEY,
        )

        # Get or create collection
        self.collection = self.pinecone_client.Index(
            name=collection_name, host=PINECONE_HOST
        )

        # Setup embedding client
        self.embed_client = OpenAI(api_key=OPENAI_API_KEY)

        self.batch_size = batch_size

        self.image_store = R2ImageStore("arxival-2")

    def _batch_encode(self, texts: List[str]) -> Tuple[List[List[float]], float]:
        """Generate embeddings in batches"""
        start_time = time.time()
        all_embeddings = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            response = self.embed_client.embeddings.create(
                model="text-embedding-3-large", input=batch
            )
            all_embeddings.extend([e.embedding for e in response.data])
        embedding_time = (time.time() - start_time) * 1000
        return all_embeddings, embedding_time

    async def add_paper(
        self,
        chunks: List[PaperChunk],
        sections: List[Section],
        images: List[ExtractedImage],
        paper_metadata: Dict,
    ):
        """Add paper chunks with sanitized metadata"""
        texts = [chunk.text for chunk in chunks]
        ids = [f"{paper_metadata['id']}_{i}" for i in range(len(chunks))]

        section_lookup = {section.get_id(): section for section in sections}

        image_metadata = []
        for img in images:
            path = await self.image_store.store_image(paper_metadata["id"], img)
            image_metadata.append(
                {
                    "figure_number": img.figure_number,
                    "paper_id": paper_metadata["id"],
                    "paper_url": paper_metadata["paper_url"],
                    "xref": img.xref,
                    "width": img.width,
                    "height": img.height,
                    "section_id": img.section_id,
                    "storage_path": path,
                }
            )

        paper_metadata["images"] = image_metadata

        metadata = []
        for chunk in chunks:
            section_id = chunk.metadata.get("section_id")
            section = section_lookup.get(section_id)

            # Prepare section data if available
            section_data = None
            if section:
                section_data = {
                    "id": section.get_id(),
                    "start_page": section.start_page,
                    "name": section.name,
                    "title": section.title,
                    "is_subsection": section.is_subsection,
                    "parent_name": section.parent_name,
                }

            # Build metadata dict with sanitized values
            meta = prepare_metadata(
                {
                    "paper_id": paper_metadata["id"],
                    "paper_url": paper_metadata["paper_url"],
                    "chunk_metadata": chunk.metadata,
                    "paper_metadata": paper_metadata,
                    "section_data": section_data,
                }
            )

            metadata.append(meta)

        # Get embeddings
        embeddings, _ = self._batch_encode(texts)

        # Add to Chroma
        vectors = [
            {
                "id": ids[i],
                "values": embeddings[i],
                "metadata": {
                    "text": texts[i],
                    **metadata[i]
                }
            }
            for i in range(len(ids))
        ]
        self.collection.upsert(vectors)


    async def retrieve(
            self,query:str,top_k : int = 3
    ) -> Tuple[List[RetrievedContext],TimingStats]:
        
        #Retrieve and reconstruct constructs
        start_time = time.time()
        query_embedding,embedding_time = self._batch_encode([query])
        query_embedding = query_embedding[0]

        retrieval_start = time.time()
        results = self.collection.query(
            vector = query_embedding,
            top_k = top_k,
            include_metadata = True
        )

        contexts = []
        for match in results.matches:
            meta = match["metadata"]
            text = meta["text"]
            chunk_meta = json.loads(meta["chunk_metadata"])
            paper_meta = json.loads(meta["paper_metadata"])


            #Reconstruct chunk
            chunk = PaperChunk(text=text,metadata=chunk_meta)

            #Parse and reconstruct section if available
            section = None
            section_data = meta.get("section_data")
            if section_data :
                try:
                    section_data = json.loads(section_data)
                    if section_data:
                        #Ensure all required fields are present
                        if all(
                            field in section_data
                              for field in ["name","title","start_page","is_subsection"]
                        ):
                            section = Section(**section_data)

                    else:
                        logger.warning("Missing fields in section data")

                except(json.JSONDecodeError,TypeError) as e:
                    logger.error(f"Error parsing section data: {e}")


            contexts.append(
                  RetrievedContext(
                      chunk=chunk,
                      paper_metadata=paper_meta,
                      section=section,
                      score=1-match["score"],
                  )
              )
            retrieval_time = (time.time() - retrieval_start) * 1000
            total_time = (time.time() - start_time) * 1000
            timing = TimingStats(
                retrieval_ms=retrieval_time,
                embedding_ms = embedding_time,
                total_ms = total_time,
            )
            return contexts,timing
        


        def build_prompt(self,query:str,contexts:List[RetrievedContext]) -> str:
            """Build structured prompt"""
        prompt = f"""Answer this research question: {query}

         Retrieved content from academic papers:"""
        
         # Group by paper
        paper_contexts = {}
        for ctx in contexts:
            paper_id = ctx.paper_metadata["id"]
            if paper_id not in paper_contexts:
                paper_contexts[paper_id] = {
                    
                }



            






    
