import os
import shutil
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, Header, HTTPException, Depends
from fastapi.responses import StreamingResponse
from groq import Groq

from app.config import ADMIN_API_KEY, GROQ_API_KEY, GROQ_MODEL
from app.schemas import ChatQueryRequest, IngestResponse
from app.ingestion import ingest_verified_file
from app.retriever import load_knowledge_base, hybrid_retrieve

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Pre-load master indexes into RAM at startup for zero delay
    load_knowledge_base()
    yield

app = FastAPI(title="Medibot Production Healthcare API", lifespan=lifespan)

def verify_admin(x_admin_token: str = Header(...)):
    if x_admin_token != ADMIN_API_KEY:
        raise HTTPException(status_code=403, detail="Forbidden: Invalid Admin Token")
    return True

@app.get("/api/v1/health")
async def health_check():
    from app.retriever import cached_faiss
    return {
        "status": "healthy",
        "faiss_ready": cached_faiss is not None,
        "runtime_model": GROQ_MODEL
    }

@app.post("/api/v1/admin/upload-verified-book", response_model=IngestResponse)
async def upload_verified_book(
    file: UploadFile = File(...),
    authenticated: bool = Depends(verify_admin)
):
    """Admin endpoint: ingests verified literature permanently."""
    temp_path = f"temp_{file.filename}"
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        chunks = ingest_verified_file(temp_path)
        # Reload master index in RAM so queries immediately use new knowledge
        load_knowledge_base()
        return IngestResponse(
            status="success",
            message=f"Book '{file.filename}' permanently saved to disk and loaded into RAM.",
            total_chunks=chunks
        )
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

@app.post("/api/v1/chat")
async def chat_stream(req: ChatQueryRequest):
    """Serves user questions with zero lag using in-memory indexed facts."""
    search_context = req.query
    user_report_prompt_text = ""

    # Ephemeral user report parsing: only used to augment query, never saved to DB
    if req.report_text:
        user_report_prompt_text = f"\n[User Attached Lab Report Details]:\n{req.report_text}\n"
        search_context = f"{req.query} {req.report_text[:250]}"

    retrieved_facts = hybrid_retrieve(search_context, top_n=3)

    # Diagnostic terminal debug print
    print("\n" + "=" * 50)
    print(f"QUERY: {req.query}")
    print(f"RETRIEVED CONTEXT (first 350 chars):\n{retrieved_facts[:350] if retrieved_facts else 'NO CONTEXT FOUND'}")
    print("=" * 50 + "\n")

    if not retrieved_facts or not retrieved_facts.strip():
        raise HTTPException(
            status_code=400,
            detail="Knowledge base is empty. Please log in as Admin and upload verified medical literature first."
        )

    # Balanced clinical prompt: strict grounding without artificial refusals
    prompt = (
        f"You are Dr. Medibot, an expert clinical AI assistant.\n"
        f"Analyze the patient's inquiry using the retrieved medical excerpts provided below.\n"
        f"Synthesize the literature and correlate clinical terms (e.g. matching headache symptoms to migraine or relevant conditions in the text).\n"
        f"Only if the topic is completely unaddressed in the text, state that the verified literature does not cover this topic.\n\n"
        f"--- VERIFIED MEDICAL EXCERPTS ---\n{retrieved_facts}\n---------------------------------\n"
        f"{user_report_prompt_text}\n"
        f"Patient Question / Symptoms: {req.query}\n\n"
        f"Provide a structured clinical response with these sections:\n"
        f"• Potential Clinical Causes (based on retrieved literature)\n"
        f"• Home Care & Next Steps\n"
        f"• Red Flag Warning Signs (When to seek immediate emergency care)\n"
    )

    if not GROQ_API_KEY or GROQ_API_KEY.startswith("your_"):
        raise HTTPException(
            status_code=500,
            detail="GROQ_API_KEY is not configured. Check your .env file."
        )

    def sse_token_stream():
        try:
            client = Groq(api_key=GROQ_API_KEY)
            stream = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[
                    {"role": "system", "content": "You are a clinical decision-support AI. Provide safe, literature-grounded medical guidance."},
                    {"role": "user", "content": prompt}
                ],
                stream=True
            )
            for chunk in stream:
                content = chunk.choices[0].delta.content
                if content:
                    safe_content = content.replace("\n", "\\n")
                    yield f"data: {safe_content}\n\n"
        except Exception as e:
            yield f"data: Error during generation: {str(e)}\n\n"

    return StreamingResponse(sse_token_stream(), media_type="text/event-stream")