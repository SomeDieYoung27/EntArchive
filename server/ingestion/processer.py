import os
import re
import logging
import pymupdf4llm
import pymupdf
from typing import List, Optional, Tuple
from ingestion.models import ExtractedImage, PaperChunk
from langchain.text_splitter import MarkdownTextSplitter
from ingestion.section import Section, SectionExtractor
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or "dummy_token"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or "dummy_token"


class PDFProcesser:
    def __init__(self,chunk_size:5000,chunk_overlap=300):
        self.llm_client = OpenAI(api_key=OPENAI_API_KEY)
        self.section_extractor = SectionExtractor()
        self.chunk_size = chunk_size
        self.min_dimension = 100
        self.min_size_bytes = 2048
        self.splitter = MarkdownTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

    async def process_pdf(self,pdf_path:str) -> tuple[List[PaperChunk],List[Section],List[ExtractedImage]]:
        """Process PDF and return both chunks and section information"""
        # Get raw text using your existing method
        text = await self._get_pdf_text(pdf_path)

        #Extract the sections first
        sections = await self.section_extractor.extract_sections(text)

        #Creating chunks with the entriched metadata
        chunks = await self._create_chunks(text)

        #Annonate chunks with section information
        self._annotate_chunks_with_sections(chunks, sections)

        images = await self._extract_images(pdf_path,sections)

        return chunks,sections,images
    

    async def _extract_images(self,pdf_path:str,sections:List[Section]) -> List[ExtractedImage]:
            """Extract images while preserving section context"""
            doc = pymupdf.open(pdf_path)
            images = []
            seen_xrefs = set()
            figure_number = 1
            max_images = 50

            for page_num in range(doc.page_count):
                 if(len(images) >= max_images):
                     break
                 
                 page_images = doc.get_images(page_num)

                 for img in page_images:
                      if(len(images) >= max_images):
                          break
                      
                      xref = img[0]
                      if xref in seen_xrefs:
                          continue
                      

                      width,height = img[2],img[3]
                      if  min(width, height) <= self.min_dimension:
                          continue
                      
                      image_dict = self._recover_image(doc, img)
                      if len(image_dict["image"]) <= self.min_size_bytes:
                        continue
                      
                      containing_section = self._find_containing_section(
                          PaperChunk(text="", metadata={"page_num": page_num + 1}),
                          sections
                      )
                      
                      images.append(ExtractedImage( xref=xref,
                        page_num=page_num + 1,
                        width=width,
                        height=height,
                        image_data=image_dict["image"],
                        extension=image_dict["ext"],
                        section_id=containing_section.get_id() if containing_section else None,
                        figure_number=figure_number))
                      
                      seen_xrefs.add(xref)
                      figure_number += 1


            return images
    

    def _recover_image(self,doc:pymupdf.Document,img:Tuple) -> dict:
        """Recover image data from PDF"""
        xref, smask = img[0], img[1]
        if smask > 0:
            pix0 = pymupdf.Pixmap(doc.extract_image(xref)["image"])
            if pix0.alpha:
                pix0 = pymupdf.Pixmap(pix0, 0)

            mask = pymupdf.Pixmap(doc.extract_image(smask)["image"])


            try:
                pix0 = pymupdf.Pixmap(pix0, mask)
            except :
                 pix = pymupdf.Pixmap(doc.extract_image(xref)["image"])


            ext = "png" if pix0.n <= 3 else "pam"


            return {
                "ext": ext,
                "colorspace" : pix.colorspace.n,
                "image": pix.tobytes(ext)
            }
        
        if "/ColorSpace" in doc.xref_object(xref, compressed=True):
             pix = pymupdf.Pixmap(doc, xref)
             pix = pymupdf.Pixmap(pymupdf.csRGB, pix)

             return {
                    "ext": "png",
                    "colorspace": 3,
                    "image": pix.tobytes("png")
                }
        return doc.extract_image(xref)
    

    def _annotate_chunks_with_sections(self,chunks :List[PaperChunk],sections:List[Section]):
        """Annotate chunks with section information"""
        for chunk in chunks:
            containing_section = self._find_containing_section(chunk,sections)
            if containing_section:
                chunk.metadata.update({
                    'section_id': containing_section.get_id(),
                    'is_subsection': containing_section.is_subsection
                })


    async def _get_pdf_text(self,pdf_path:str) -> str:
        """Extract clean text from PDF while preserving structure"""
        try:
            #Using your existing pymupdf4llm integration
            md_text = pymupdf4ll.to_markdown(pdf_path,show_progress=False)
            return self._clean_text(md_text)

        except Exception as e :
             logger.error(f"Error extracting PDF text: {str(e)}")
             raise


    def _find_containing_section(self, chunk: PaperChunk, sections: List[Section]) -> Optional[Section]:
        """Find which section a chunk belongs to based on page numbers"""
        chunk_page = chunk.metadata.get('page_num')
        if not chunk_page:
            return None

        # Sort sections by page number
        sorted_sections = sorted(sections, key=lambda s: s.start_page)

        # Find the last section that starts before or on this page
        containing_section = None
        for section in sorted_sections:
            if section.start_page <= chunk_page:
                containing_section = section
            else:
                break

        return containing_section

    async def _create_chunks(self,text:str) -> List[PaperChunk]:
         """Create overlapping chunks from document text"""
         try:
            split_docs = self.splitter.create_documents([text])
            chunks = []


            for i,doc in enumerate(split_docs):
                page_num = self._estimate_page_num(doc.page_content)

                chunk = PaperChunk(
                    text = doc.page_content,
                    metadata = {
                        'has_equations': bool(re.search(r'\$\$.+?\$\$', doc.page_content)),
                        'page_num': page_num,
                        **doc.metadata
                    }
                )
                chunks.append(chunk)


            return chunks

         except Exception as e:
            logger.error(f"Error creating chunks: {str(e)}")
            raise

    def _estimate_page_num(self,text:str) -> int:
        """
        Estimate page number for a chunk of text using multiple detection methods.
        Falls back through progressively less reliable methods.
        """
         # Method 1: Look for explicit page markers with variations
        page_patterns = [
             r'Page[:\s-]*(\d+)',  # "Page 1", "Page: 1", "Page-1"
            r'\[pg\.?\s*(\d+)\]', # [pg 1], [pg. 1]
            r'\(p\.?\s*(\d+)\)',  # (p 1), (p. 1)
            r'^\s*(\d+)\s*$',     # Standalone numbers at start of lines
            r'_{2,}\s*(\d+)\s*$'  # Page numbers after underscores
        ]

        for pattern in page_patterns:
            matches = re.findall(pattern,text,re.MULTILINE | re.IGNORECASE)
            if matches:
                #Take the most frequent page number found
                page_numbers = [int(m) for m in matches]
                return max(set(page_numbers),key=page_numbers.count)


          #Method 2 : Look for footer/header patterns
            footer_matches = re.findall(r'\n[-−–—]\s*(\d+)\s*[-−–—]', text)
            if footer_matches:
                return int(footer_matches[-1])

          #Method 3 : Extract numbers that appear to be in header/footer positions
            lines = text.split('\n')
            if len(lines) > 4:
                potential_numbers = []
                #Check first and two last lines
                check_lines = [lines[0],lines[1],lines[-2],lines[-1]]
                for line in check_lines:
                    numbers = re.findall(r'\b(\d+)\b', line)
                    potential_numbers.extend([int(n) for n in numbers if 1 <= int(n) <= 9999])

                if potential_numbers:
                    return min(potential_numbers)


            if len(text) < 1000:  #Short text likely from early in document
                return 1
            else:
                return max(1,len(text) // 3000) # Assume ~3000 chars per page


    def _clean_text(self,text:str) -> str:
            """Clean text while preserving markdown elements"""
        # Remove extra whitespace but preserve markdown
            text = re.sub(r'\n\s*\n', '\n\n', text)
            text = re.sub(r' + ', ' ',text)


            #Preserve equations
            text = re.sub(r'\$\$(.*?)\$\$', r'$$\1$$', text, flags=re.DOTALL)

        # Clean up markdown headers    
            text = re.sub(r'^#{1,6}\s*', '# ', text, flags=re.MULTILINE)

            return text.strip()



                 




        