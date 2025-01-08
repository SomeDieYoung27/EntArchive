from fastapi import FastAPI,HttpException,Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import logging
from datetime import datetime
import os

from api.routes import query,stats
from api.models import HealthResponse
from api.config.settings import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="ENTARCHIVE API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_HOSTS,
    allow_credentials=True,
    allow_methods=["GET"],
    allow_headers=["*"],
)
@app.exception_handler(Exception)

async def global_exception_handler(_:Request,exc : Exception):
    logger.error(f"Global error handler caught: {str(exc)}", exc_info=True)
    error_id = datetime.now().strftime("%Y%m%d%H%M%S")

    content={
            "error": "Internal server error",
            "error_id": error_id,
            "message": str(exc)
            if settings.ENV == "dev"
            else "An unexpected error occurred",
        },
@app.get("/health", response_model=HealthResponse)

async def health_check():
    try:
        last_ingest = None
        last_run_file = os.path.join(settings.DATA_DIR, "last_run.txt")

        if os.path.exists(last_run_file):
            with open(last_run_file, "r") as f:
                last_ingest = f.read().strip()

        return {"status" : "healthy","env":settings.ENV,"last_ingest":last_ingest}
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Health check failed")
    
app.include_router(query.router)
app.include_router(stats.router)
