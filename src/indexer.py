"""增量索引模块 —— 基于文档 Hash 的变更检测与热更新"""
import hashlib
import json
import os
from pathlib import Path
from datetime import datetime

from langchain_chroma import Chroma
from langchain_core.documents import Document as LCDocument

from .embedder import get_embedding_model
from .loader import load_document
from .chunker import auto_split


class IncrementalIndexer:
    """增量索引器：检测文档变更，只重建有变化的文档"""

    def __init__(self, index_dir: str, manifest_path: str = "data/index/manifest.json"):
        self.index_dir = index_dir
        self.manifest_path = manifest_path
        self.manifest: dict[str, str] = {}  # {filename: hash}
        self.embedding = get_embedding_model()
        self._load_manifest()

    # ═══════════════════════════════════════════════════
    # 文件 Hash 计算
    # ═══════════════════════════════════════════════════

    @staticmethod
    def compute_hash(file_path: Path) -> str:
        """计算文件 SHA256，用于检测内容变更"""
        sha = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha.update(chunk)
        return sha.hexdigest()

    # ═══════════════════════════════════════════════════
    # Manifest 管理
    # ═══════════════════════════════════════════════════

    def _load_manifest(self):
        """加载上次索引的文件 Hash 快照"""
        path = Path(self.manifest_path)
        if path.exists():
            self.manifest = json.loads(path.read_text("utf-8"))
        else:
            self.manifest = {}

    def _save_manifest(self):
        """保存当前文件 Hash 快照"""
        path = Path(self.manifest_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.manifest, ensure_ascii=False, indent=2), "utf-8")

    # ═══════════════════════════════════════════════════
    # 变更检测
    # ═══════════════════════════════════════════════════

    def detect_changes(self, file_paths: list[Path]) -> dict:
        """检测文件变更，返回三类文件列表

        Returns:
            {"new": [], "modified": [], "deleted": []}
        """
        current = {fp.name: self.compute_hash(fp) for fp in file_paths}
        known = set(self.manifest.keys())
        seen = set(current.keys())

        changes = {
            "new": [fp for fp in file_paths if fp.name not in known],
            "modified": [fp for fp in file_paths
                         if fp.name in known and current[fp.name] != self.manifest[fp.name]],
            "deleted": [name for name in known if name not in seen],
        }
        return changes

    # ═══════════════════════════════════════════════════
    # 增量构建
    # ═══════════════════════════════════════════════════

    def build_full(self, file_paths: list[Path]) -> Chroma:
        """全量构建索引（首次使用）"""
        return self._index_files(file_paths, clear_first=True)

    def build_incremental(self, file_paths: list[Path]) -> dict:
        """增量构建索引

        Returns:
            {"new": n, "modified": n, "deleted": n} 变更统计
        """
        changes = self.detect_changes(file_paths)
        store = Chroma(
            persist_directory=self.index_dir,
            embedding_function=self.embedding,
            collection_name="knowledge_base",
        )

        stats = {"new": 0, "modified": 0, "deleted": 0}

        # 新增 + 修改 → 重新索引
        to_index = changes["new"] + changes["modified"]
        if to_index:
            store = self._index_files(to_index, store=store)
            stats["new"] = len(changes["new"])
            stats["modified"] = len(changes["modified"])

        # 删除 → 从向量库中移除
        for name in changes["deleted"]:
            ids = store.get(where={"source": name}).get("ids", [])
            if ids:
                store.delete(ids=ids)
            stats["deleted"] += 1

        # 更新 manifest
        for fp in file_paths:
            self.manifest[fp.name] = self.compute_hash(fp)
        self._save_manifest()

        return stats

    def _index_files(
        self, file_paths: list[Path],
        store: Chroma | None = None, clear_first: bool = False
    ) -> Chroma:
        """对一批文件执行索引构建"""
        if store is None:
            store = Chroma(
                persist_directory=self.index_dir,
                embedding_function=self.embedding,
                collection_name="knowledge_base",
            )

        for fp in file_paths:
            doc = load_document(fp)
            chunks = auto_split(doc.content, doc.metadata.get("type", "txt"))
            texts = [c.content for c in chunks]
            metadatas = [c.metadata for c in chunks]
            # 使用 文件名+序号 做 ID，方便后续按文件删除
            ids = [f"{fp.name}_{i}" for i in range(len(texts))]

            # 先删旧数据再插入（upsert）
            old_ids = store.get(where={"source": fp.name}).get("ids", [])
            if old_ids:
                store.delete(ids=old_ids)

            store.add_texts(texts=texts, metadatas=metadatas, ids=ids)

        return store
