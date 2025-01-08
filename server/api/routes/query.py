import aysncio
import json
from fastapi import APIRouter, HTTPException, Depends, Query
from rag.rag import RAGPipeline
from api.rate_limit import rate_limit
import logging
from api.db import get_chat_context
from sse_starlette.sse import EventSourceResponse

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api")
rag = RAGPipeline()

@router.get("/sse")

async def sse_endpoint():
    async def event_generator():
        for i in range(5):
            yield {"data": f"message {i}"}
            await asyncio.sleep(1)

    return EventSourceResponse(event_generator())

@router.get("/query/stream")

async def stream_query(
    q : str = Query(...,description ="Query string",min_length = 1),
    _:None = Depends(rate_limit),
):
    async def event_generator():
        try:
            async for chunk in rag.generate(query=q):
                if chunk["event"] == "error":
                    yield {"event" : 'err',"data":chunk["data"]}
                    break
                else:
                    yield {"event":chunk["event"],"data":chunk["data"]}

        except Exception as e:
            logger.error(f"Failed to stream query: {str(e)}",exc_info=True)
            yield {"event":"err","data":json.dumps({"message":str(e)})}

        return EventSourceResponse(
            event_generator(),
        )
    
@router.get("/query/followup/stream")

async def stream_followup(
    q:str = Query(...,description = "Follow-up query"),
    session_id: str = Query(...,description = "Previous conversation context"),
    _:None  = Depends(rate_limit)    
    ):
    "Handling the follow up with queries with existing content"

    async def event_generator():
        try:
            context = await get_chat_context(session_id)

            async for chunk in rag.generate_followup(
                query = q,
                context = context,
                top_k = 2,
            ):
                if chunk["event"] == "error":
                    yield {"event": "err", "data": chunk["data"]}
                    break
                else:
                    yield {"event":chunk['event'],"data": chunk["data"]}


        except Exception as e:
            logger.error(f"Streaming error: {str(e)}")
            yield {"event": "err", "data": json.dumps({"message": str(e)})}


    return EventSourceResponse(event_generator())
    
