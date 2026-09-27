from pydantic import BaseModel
from typing import Optional

class ChatQueryRequest(BaseModel):
    query: str
    report_text: Optional[str] = None  # Ephemeral user lab report content

class IngestResponse(BaseModel):
    status: str
    message: str
    total_chunks: int