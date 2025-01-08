from fastapi import Request
import asyncio

async def rate_limit(request:Request):
    await asyncio.sleep(0.1)