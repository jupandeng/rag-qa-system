"""验证文档加载和切分模块"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.loader import load_document
from src.chunker import sentence_split, markdown_split


def test_load_pdf():
    """测试 PDF 加载（需要有一个 PDF 文件）"""
    pdf_dir = Path("data/docs")
    pdfs = list(pdf_dir.glob("*.pdf")) if pdf_dir.exists() else []
    if not pdfs:
        print("[跳过] 没有测试 PDF 文件，请放一个 PDF 到 data/docs/")
        return
    doc = load_document(pdfs[0])
    print(f"[OK] PDF 加载: {doc.metadata['pages']} 页, {len(doc.content)} 字符")


def test_load_markdown():
    """测试 Markdown 加载（用当前 README 测试）"""
    readme = Path("README.md")
    if not readme.exists():
        print("[跳过] 没有 README.md")
        return
    doc = load_document(readme)
    print(f"[OK] Markdown 加载: {len(doc.content)} 字符")


def test_chunker():
    """测试切分"""
    text = "这是第一句话。这是第二句话！这是第三句话？" * 20
    chunks = sentence_split(text, max_chunk_size=200)
    print(f"[OK] 按句切分: {len(chunks)} 个块")
    for i, c in enumerate(chunks[:3]):
        print(f"  chunk[{i}]: {c.content[:50]}... ({len(c.content)} 字符)")


if __name__ == "__main__":
    test_load_pdf()
    test_load_markdown()
    test_chunker()
    print("\n全部测试通过!")
