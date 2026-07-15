"""检索模块 —— 在线检索第一步：多路召回 + 重排序"""
from typing import Optional
from langchain_chroma import Chroma
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder


class HybridRetriever:
    """混合检索器：稠密向量 + BM25 稀疏检索 + Rerank"""

    def __init__(
        self,
        vectorstore: Chroma,
        reranker_model: str = "BAAI/bge-reranker-v2-m3",
    ):
        self.vectorstore = vectorstore
        # 从 vectorstore 中取出所有文档构建 BM25 索引
        self._build_bm25()
        # 重排序模型（用的时候才加载）
        self._reranker: Optional[CrossEncoder] = None
        self._reranker_model = reranker_model

    def _build_bm25(self):
        """从向量库中提取文档构建 BM25 索引"""
        # Chroma 的 get 方法获取所有文档
        result = self.vectorstore.get(include=["documents", "metadatas"])
        self.documents = result.get("documents", [])
        self.metadatas = result.get("metadatas", [])

        # 对文档分词（中文按字+2-gram 简单处理）
        tokenized = [self._tokenize(d) for d in self.documents]
        self.bm25 = BM25Okapi(tokenized) if tokenized else None

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """简单中文分词：按字切 + 2-gram"""
        text = text.replace("\n", " ").replace("\r", " ")
        chars = list(text)
        unigrams = [c for c in chars if c.strip()]
        bigrams = [
            chars[i] + chars[i + 1]
            for i in range(len(chars) - 1)
            if chars[i].strip() and chars[i + 1].strip()
        ]
        return unigrams + bigrams

    def _get_reranker(self) -> CrossEncoder:
        if self._reranker is None:
            self._reranker = CrossEncoder(self._reranker_model)
        return self._reranker

    def dense_search(self, query: str, top_k: int = 10) -> list[dict]:
        """稠密向量检索"""
        results = self.vectorstore.similarity_search_with_score(query, k=top_k)
        return [
            {"content": doc.page_content, "metadata": doc.metadata, "score": score}
            for doc, score in results
        ]

    def sparse_search(self, query: str, top_k: int = 10) -> list[dict]:
        """BM25 稀疏检索"""
        if self.bm25 is None:
            return []
        tokenized = self._tokenize(query)
        scores = self.bm25.get_scores(tokenized)
        # 取 top_k
        indexed = list(enumerate(scores))
        indexed.sort(key=lambda x: x[1], reverse=True)
        results = []
        for idx, score in indexed[:top_k]:
            if score > 0:
                results.append(
                    {
                        "content": self.documents[idx],
                        "metadata": self.metadatas[idx] if idx < len(self.metadatas) else {},
                        "score": float(score),
                    }
                )
        return results

    def rerank(self, query: str, candidates: list[dict], top_k: int = 5) -> list[dict]:
        """用 CrossEncoder 重排序"""
        if not candidates:
            return []
        model = self._get_reranker()
        pairs = [(query, c["content"]) for c in candidates]
        scores = model.predict(pairs)
        for i, c in enumerate(candidates):
            c["rerank_score"] = float(scores[i])
        candidates.sort(key=lambda x: x["rerank_score"], reverse=True)
        return candidates[:top_k]

    def retrieve(self, query: str, top_k: int = 5) -> list[dict]:
        """混合检索：稠密召回 + BM25 召回 → 合并去重 → Rerank"""
        dense_results = self.dense_search(query, top_k=10)
        sparse_results = self.sparse_search(query, top_k=10)

        # 按 content 去重合并
        seen = set()
        merged = []
        for r in dense_results + sparse_results:
            key = r["content"][:100]  # 前100字当指纹
            if key not in seen:
                seen.add(key)
                merged.append(r)

        # Rerank
        return self.rerank(query, merged, top_k=top_k)
