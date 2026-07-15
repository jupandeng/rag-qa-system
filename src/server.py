"""FastAPI 服务 —— 提供 REST API 接口"""
from pathlib import Path
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from .pipeline import RAGPipeline

app = FastAPI(title="RAG QA System", version="0.1.0")
pipeline = RAGPipeline("./data/index")


class AskRequest(BaseModel):
    question: str
    top_k: int = 5


class AskResponse(BaseModel):
    answer: str
    sources: list[dict]


@app.post("/ingest")
async def ingest(files: list[UploadFile] = File(...)):
    """上传文件并构建索引"""
    upload_dir = Path("./data/uploads")
    upload_dir.mkdir(parents=True, exist_ok=True)

    paths = []
    for f in files:
        file_path = upload_dir / f.filename
        content = await f.read()
        file_path.write_bytes(content)
        paths.append(file_path)

    pipeline.ingest(paths)
    return {"status": "ok", "file_count": len(files)}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    """问答接口"""
    result = pipeline.query(req.question, req.top_k)
    return AskResponse(answer=result["answer"], sources=result["sources"])


@app.get("/health")
def health():
    return {"status": "ok"}
