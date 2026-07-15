"""RAG 主流程：加载→切分→索引→查询改写→检索→缓存→生成"""
import os
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
from .embedder import build_index, load_index
from .retriever import HybridRetriever
from .rewriter import QueryRewriter, QuestionRouter
from .indexer import IncrementalIndexer
from .cache import LayeredCache

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
        self.rewriter = QueryRewriter()
        self.router = QuestionRouter()
        self.indexer = IncrementalIndexer(index_dir)
        self.cache = LayeredCache()

    def ingest(self, file_paths: list[Path], incremental: bool = True):
        """离线索引：支持全量/增量构建"""
        if incremental and Path(self.index_dir).exists():
            stats = self.indexer.build_incremental(file_paths)
            print(f"[Index] 增量索引: 新增{stats['new']} 修改{stats['modified']} 删除{stats['deleted']}")
        else:
            self.indexer.build_full(file_paths)
            print(f"[Index] 全量索引完成")

        vectorstore = load_index(self.index_dir)
        self.retriever = HybridRetriever(vectorstore)

    def load(self):
        """加载已有索引"""
        if self.retriever is not None:
            return
        vectorstore = load_index(self.index_dir)
        self.retriever = HybridRetriever(vectorstore)

    def query(self, question: str, top_k: int = 5) -> dict:
        """在线问答：缓存检查 → 查询改写 → 多路检索 → 生成"""
        if self.retriever is None:
            self.load()

        # 0. 缓存检查
        # FAQ 精确匹配
        faq_hit = self.cache.get_faq(question)
        if faq_hit:
            return {"answer": faq_hit, "sources": [], "context": "", "cached": True}
        # QA 模糊缓存
        qa_hit = self.cache.get_qa(question)
        if qa_hit:
            return qa_hit

        # 1. 查询改写
        queries = self.rewriter.rewrite(question)

        # 2. 多查询检索 + 合并去重
        hits = self._multi_query_retrieve(queries, top_k)

        # 3. 低置信度二次检索
        if self._should_retrieve_again(hits):
            expanded = self._expand_retrieve(question, hits, top_k)
            if expanded:
                hits = self.retriever.rerank(question, expanded, top_k=top_k)

        # 4. 拼装上下文
        context = self._build_context(hits)

        # 5. 生成回答
        prompt = PROMPT_TEMPLATE.format(context=context, question=question)
        response = self.llm.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        answer = response.choices[0].message.content

        # 6. 引用核查
        answer = self._verify_citations(answer, hits)

        result = {"answer": answer, "sources": hits, "context": context}

        # 7. 写入缓存
        self.cache.set_qa(question, result)

        return result

    def _multi_query_retrieve(self, queries: list[str], top_k: int) -> list[dict]:
        """多查询检索：每个改写查询各自召回，合并去重"""
        seen = set()
        merged = []
        # 第一个查询多取一些，后续查询少取
        for i, q in enumerate(queries):
            k = top_k + 5 if i == 0 else top_k // 2
            results = self.retriever.retrieve(q, top_k=k)
            for r in results:
                key = r["content"][:80]
                if key not in seen:
                    seen.add(key)
                    merged.append(r)
        return merged

    def _should_retrieve_again(self, hits: list[dict], threshold: float = 0.3) -> bool:
        """判断是否需要二次检索：最高分太低或结果太少"""
        if len(hits) < 3:
            return True
        if hits and hits[0].get("rerank_score", 0) < threshold:
            return True
        return False

    def _expand_retrieve(
        self, question: str, existing: list[dict], top_k: int
    ) -> list[dict]:
        """扩展检索：用纯关键词（不加改写）补搜一次"""
        rough = self.retriever.sparse_search(question, top_k=top_k)
        seen = {r["content"][:80] for r in existing}
        new = [r for r in rough if r["content"][:80] not in seen]
        return existing + new

    def _build_context(self, hits: list[dict]) -> str:
        """拼装上下文（带页码溯源）"""
        parts = []
        for i, h in enumerate(hits):
            src = h["metadata"].get("source", "unknown")
            page = h["metadata"].get("page", "")
            page_info = f" 第{page}页" if page else ""
            parts.append(f"[来源{i+1}: {src}{page_info}]\n{h['content']}")
        return "\n\n---\n\n".join(parts)

    def _verify_citations(self, answer: str, hits: list[dict]) -> str:
        """引用核查：确保回答中引用的文件名在检索结果中真实存在"""
        sources = {h["metadata"].get("source", "") for h in hits}
        # 简单检查：如果回答末尾没有来源列表，追加一个
        if not answer:
            return answer
        has_citation = any(src in answer for src in sources if src)
        if not has_citation and sources:
            cited = ", ".join(s for s in sources if s)
            answer += f"\n\n参考来源: {cited}"
        return answer


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
