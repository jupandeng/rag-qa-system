"""Embedding 与向量存储模块 —— 离线索引第二步"""
import os
from pathlib import Path
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from .chunker import Chunk, auto_split
from .loader import load_document


def get_embedding_model() -> OpenAIEmbeddings:
    """获取 Embedding 模型（兼容 OpenAI API 格式）"""
    return OpenAIEmbeddings(
        model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
        base_url=os.getenv("OPENAI_BASE_URL"),
        api_key=os.getenv("OPENAI_API_KEY"),
    )


def build_index(
    file_paths: list[Path],
    persist_dir: str,
    collection_name: str = "knowledge_base",
) -> Chroma:
    """构建向量索引：加载文档 → 切分 → 向量化 → 存储"""
    all_chunks: list[Chunk] = []
    for fp in file_paths:
        doc = load_document(fp)
        chunks = auto_split(doc.content, doc.metadata.get("type", "txt"))
        # 每个 chunk 继承文档元数据
        for c in chunks:
            c.metadata["source"] = fp.name
            c.metadata["file_type"] = doc.metadata["type"]
        all_chunks.extend(chunks)

    texts = [c.content for c in all_chunks]
    metadatas = [c.metadata for c in all_chunks]
    ids = [f"chunk_{i}" for i in range(len(texts))]

    embedding = get_embedding_model()
    vectorstore = Chroma.from_texts(
        texts=texts,
        embedding=embedding,
        metadatas=metadatas,
        ids=ids,
        persist_directory=persist_dir,
        collection_name=collection_name,
    )
    return vectorstore


def load_index(persist_dir: str, collection_name: str = "knowledge_base") -> Chroma:
    """加载已有的向量索引"""
    embedding = get_embedding_model()
    return Chroma(
        persist_directory=persist_dir,
        embedding_function=embedding,
        collection_name=collection_name,
    )
