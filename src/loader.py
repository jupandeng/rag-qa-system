"""文档加载与解析模块 —— 离线索引第一步"""
from pathlib import Path
from typing import Optional
import fitz  # PyMuPDF


class Document:
    """统一文档对象"""

    def __init__(self, content: str, metadata: dict):
        self.content = content
        self.metadata = metadata

    def __repr__(self):
        src = self.metadata.get("source", "unknown")
        return f"Document({src!r}, {len(self.content)} chars)"


def load_pdf(file_path: Path) -> Document:
    """加载 PDF 文件，提取文本和元数据"""
    doc = fitz.open(str(file_path))
    parts = []
    metadata = {
        "source": file_path.name,
        "type": "pdf",
        "pages": doc.page_count,
        "title": doc.metadata.get("title", ""),
    }

    for page in doc:
        text = page.get_text()
        if text.strip():
            parts.append(text)

    doc.close()
    return Document("\n\n".join(parts), metadata)


def load_markdown(file_path: Path) -> Document:
    """加载 Markdown 文件，保留标题层级信息"""
    text = file_path.read_text(encoding="utf-8")
    return Document(
        text,
        {"source": file_path.name, "type": "markdown", "title": file_path.stem},
    )


def load_txt(file_path: Path) -> Document:
    """加载纯文本文件"""
    text = file_path.read_text(encoding="utf-8")
    return Document(
        text, {"source": file_path.name, "type": "txt", "title": file_path.stem}
    )


LOADERS = {
    ".pdf": load_pdf,
    ".md": load_markdown,
    ".txt": load_txt,
}


def load_document(file_path: Path) -> Document:
    """根据文件后缀自动选择加载器"""
    loader = LOADERS.get(file_path.suffix.lower())
    if loader is None:
        raise ValueError(f"不支持的文件类型: {file_path.suffix}")
    return loader(file_path)


def load_from_directory(dir_path: Path, glob_pattern: str = "*.*") -> list[Document]:
    """批量加载目录下的文档"""
    docs = []
    for fp in dir_path.glob(glob_pattern):
        if fp.suffix.lower() in LOADERS and fp.is_file():
            docs.append(load_document(fp))
    return docs
