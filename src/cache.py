"""分层缓存模块 —— FAQ 缓存 + Embedding 缓存 + 问答结果缓存"""
import hashlib
import json
import time
from pathlib import Path
from functools import wraps
from collections import OrderedDict


class LRUCache:
    """简单的 LRU 内存缓存"""

    def __init__(self, max_size: int = 128):
        self.cache: OrderedDict[str, tuple[float, object]] = OrderedDict()
        self.max_size = max_size

    def get(self, key: str) -> object | None:
        if key not in self.cache:
            return None
        # 访问时移到末尾（最近使用）
        self.cache.move_to_end(key)
        return self.cache[key][1]

    def put(self, key: str, value: object):
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = (time.time(), value)
        # 超限时淘汰最旧的
        while len(self.cache) > self.max_size:
            self.cache.popitem(last=False)

    def clear(self):
        self.cache.clear()

    def __len__(self):
        return len(self.cache)


class LayeredCache:
    """三层缓存：Embedding → FAQ → QA 结果

    层级      | 粒度           | 过期策略
    embedding | 每个 chunk     | 永久（跟随索引生命周期）
    faq       | 精确问题匹配    | 永久（问过的问题答案不会变）
    qa_result | 模糊问题匹配    | LRU 淘汰
    """

    def __init__(self, cache_dir: str = "data/cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # 内存缓存
        self.embedding_cache = LRUCache(max_size=1024)
        self.qa_cache = LRUCache(max_size=64)

        # FAQ 持久化缓存（精确匹配，写入磁盘）
        self.faq_path = self.cache_dir / "faq_cache.json"
        self.faq_cache: dict[str, str] = self._load_faq()

    def _load_faq(self) -> dict[str, str]:
        if self.faq_path.exists():
            return json.loads(self.faq_path.read_text("utf-8"))
        return {}

    def _save_faq(self):
        self.faq_path.write_text(
            json.dumps(self.faq_cache, ensure_ascii=False, indent=2),
            "utf-8",
        )

    # --- Embedding 缓存 ---

    def get_embedding(self, text: str) -> list[float] | None:
        key = hashlib.md5(text.encode()).hexdigest()
        return self.embedding_cache.get(key)

    def set_embedding(self, text: str, vector: list[float]):
        key = hashlib.md5(text.encode()).hexdigest()
        self.embedding_cache.put(key, vector)

    # --- FAQ 缓存（精确匹配）---

    def get_faq(self, question: str) -> str | None:
        return self.faq_cache.get(question.strip())

    def set_faq(self, question: str, answer: str):
        self.faq_cache[question.strip()] = answer
        self._save_faq()

    # --- QA 结果缓存（模糊匹配，LRU）---

    def _qa_key(self, question: str) -> str:
        return hashlib.md5(question.strip().encode()).hexdigest()

    def get_qa(self, question: str) -> dict | None:
        return self.qa_cache.get(self._qa_key(question))

    def set_qa(self, question: str, result: dict):
        self.qa_cache.put(self._qa_key(question), result)
