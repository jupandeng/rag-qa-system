"""RAG 主流程：加载→切分→索引→检索→生成"""
import os
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
from .embedder import build_index, load_index
from .retriever import HybridRetriever

load_dotenv()


class RAGPipeline:
    def __init__(self, index_dir: str = "./data/index"):
        self.index_dir = index_dir
        self.llm = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL"),
        )
        self.model = os.getenv("LLM_MODEL", "gpt-4o-mini")
        self.retriever: HybridRetriever | None = None

    def ingest(self, file_paths: list[Path]):
        """离线索引：加载文档 → 切分 → 向量化 → 存储"""
        vectorstore = build_index(file_paths, self.index_dir)
        self.retriever = HybridRetriever(vectorstore)

    def load(self):
        """加载已有索引"""
        if self.retriever is not None:
            return
        vectorstore = load_index(self.index_dir)
        self.retriever = HybridRetriever(vectorstore)

    def query(self, question: str, top_k: int = 5) -> dict:
        """在线问答：检索 + LLM 生成"""
        if self.retriever is None:
            self.load()

        # 1. 检索
        hits = self.retriever.retrieve(question, top_k=top_k)

        # 2. 拼装上下文（带页码溯源）
        context_parts = []
        for i, h in enumerate(hits):
            src = h["metadata"].get("source", "unknown")
            page = h["metadata"].get("page", "")
            page_info = f" 第{page}页" if page else ""
            context_parts.append(f"[来源{i+1}: {src}{page_info}]\n{h['content']}")
        context = "\n\n---\n\n".join(context_parts)

        # 3. 生成回答
        prompt = PROMPT_TEMPLATE.format(context=context, question=question)
        response = self.llm.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        answer = response.choices[0].message.content

        return {"answer": answer, "sources": hits, "context": context}


PROMPT_TEMPLATE = """你是一个知识库问答助手。请根据以下参考资料回答用户问题。

## 规则
- 如果参考资料不足以回答问题，请如实告知，不要编造。
- 回答末尾列出引用的来源文件。
- 用中文回答。

## 参考资料
{context}

## 用户问题
{question}

## 回答"""
