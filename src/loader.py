"""文档加载与解析模块 —— 离线索引第一步"""
import re
from pathlib import Path
import fitz  # PyMuPDF


class Document:
    """统一文档对象"""

    def __init__(self, content: str, metadata: dict):
        self.content = content
        self.metadata = metadata

    def __repr__(self):
        src = self.metadata.get("source", "unknown")
        return f"Document({src!r}, {len(self.content)} chars)"


# ═══════════════════════════════════════════════════════════════
# 文本清洗
# ═══════════════════════════════════════════════════════════════

def clean_text(text: str) -> str:
    """清洗 PDF 提取出的文本：去页眉页脚、页码、多余空白"""
    # 去掉纯数字的页码行（独立成行，前后可能是空白）
    text = re.sub(r"^\s*\d{1,4}\s*$", "", text, flags=re.MULTILINE)
    # 去掉 "第X页" 格式的页码
    text = re.sub(r"第\s*\d+\s*页", "", text)
    # 合并连续空白行（3个以上换行 → 2个换行）
    text = re.sub(r"\n{3,}", "\n\n", text)
    # 去掉行首行尾多余空格
    text = "\n".join(line.strip() for line in text.splitlines())
    # 去掉首尾空白
    text = text.strip()
    return text


# ═══════════════════════════════════════════════════════════════
# PDF 加载（增强版：按页提取 + OCR + 清洗）
# ═══════════════════════════════════════════════════════════════

def _ocr_image(image_bytes: bytes) -> str:
    """对单张图片执行 OCR，返回识别文字"""
    try:
        from paddleocr import PaddleOCR
    except ImportError:
        return ""

    # PaddleOCR 单例，避免重复初始化
    if not hasattr(_ocr_image, "_ocr"):
        _ocr_image._ocr = PaddleOCR(lang="ch", show_log=False)
    ocr = _ocr_image._ocr

    import numpy as np
    from PIL import Image
    import io

    img = Image.open(io.BytesIO(image_bytes))
    img_array = np.array(img)
    result = ocr.ocr(img_array)
    if not result or not result[0]:
        return ""

    lines = [line[1][0] for line in result[0]]
    return "\n".join(lines)


def load_pdf(file_path: Path, enable_ocr: bool = False, enable_vlm: bool = False) -> Document:
    """加载 PDF，按页提取文本和图片中的文字

    Args:
        file_path: PDF 文件路径
        enable_ocr: 是否对 PDF 中的图片执行 OCR（首次运行会下载模型）
        enable_vlm: 是否用 VLM 对图片生成语义描述（需配置 VLM_API_KEY）
    """
    doc = fitz.open(str(file_path))
    pages_content = []

    # 延迟加载 VLM
    vlm = None
    if enable_vlm:
        from .vision import ImageDescriber
        vlm = ImageDescriber()

    for page_num, page in enumerate(doc, start=1):
        page_parts = []

        # 1. 提取正文文字
        text = page.get_text()
        if text.strip():
            page_parts.append(text.strip())

        # 2. 处理图片：OCR + VLM
        images = page.get_images(full=True)
        for img_info in images:
            xref = img_info[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            img_format = base_image.get("ext", "png")

            # OCR：文字识别
            if enable_ocr:
                img_text = _ocr_image(image_bytes)
                if img_text.strip():
                    page_parts.append(f"[图片文字]\n{img_text}")

            # VLM：语义描述
            if vlm:
                desc = vlm.describe(image_bytes, img_format)
                if desc and not desc.startswith("[VLM不可用"):
                    page_parts.append(f"[图片描述]\n{desc}")

        if page_parts:
            # 标记页码，方便后续溯源
            combined = f"[第{page_num}页]\n" + "\n".join(page_parts)
            pages_content.append(combined)

    doc.close()

    metadata = {
        "source": file_path.name,
        "type": "pdf",
        "pages": doc.page_count if hasattr(doc, 'page_count') else len(pages_content),
        "title": file_path.stem,
    }

    full_text = "\n\n".join(pages_content)
    full_text = clean_text(full_text)
    return Document(full_text, metadata)


# ═══════════════════════════════════════════════════════════════
# Markdown / TXT 加载
# ═══════════════════════════════════════════════════════════════

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


# ═══════════════════════════════════════════════════════════════
# 加载器注册与入口
# ═══════════════════════════════════════════════════════════════

LOADERS = {
    ".pdf": load_pdf,
    ".md": load_markdown,
    ".txt": load_txt,
}


def load_document(file_path: Path, **kwargs) -> Document:
    """根据文件后缀自动选择加载器"""
    loader = LOADERS.get(file_path.suffix.lower())
    if loader is None:
        raise ValueError(f"不支持的文件类型: {file_path.suffix}")
    return loader(file_path, **kwargs)


def load_from_directory(dir_path: Path, glob_pattern: str = "*.*", **kwargs) -> list[Document]:
    """批量加载目录下的文档"""
    docs = []
    for fp in dir_path.glob(glob_pattern):
        if fp.suffix.lower() in LOADERS and fp.is_file():
            docs.append(load_document(fp, **kwargs))
    return docs
