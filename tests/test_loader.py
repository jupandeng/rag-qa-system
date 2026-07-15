"""测试文档加载与清洗"""
import sys
import io
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.loader import clean_text, load_document


def test_clean_text_removes_page_numbers():
    text = "这是正文\n\n12\n\n继续正文"
    result = clean_text(text)
    assert "12" not in result.strip().split("\n\n"), f"页码未清除: {result}"


def test_clean_text_removes_chinese_page_numbers():
    text = "第一章 概述\n第 5 页\n正文内容"
    result = clean_text(text)
    assert "第" not in result or "页" not in result


def test_clean_text_collapses_multiple_newlines():
    text = "段落一\n\n\n\n\n段落二"
    result = clean_text(text)
    assert "\n\n\n" not in result


def test_clean_text_preserves_normal_text():
    text = "这是一段正常的文本内容"
    result = clean_text(text)
    assert result == text


def test_load_readme():
    """用 README 测试 Markdown 加载"""
    readme = Path(__file__).resolve().parent.parent / "README.md"
    if readme.exists():
        doc = load_document(readme)
        assert len(doc.content) > 0
        assert doc.metadata["source"].endswith("README.md")


def test_load_pdf():
    """测试 PDF 加载（跳过如果没有文件）"""
    pdf_dir = Path("data/docs")
    pdfs = list(pdf_dir.glob("*.pdf")) if pdf_dir.exists() else []
    if not pdfs:
        print("[跳过] 没有测试 PDF 文件")
        return
    doc = load_document(pdfs[0])
    assert doc.metadata["pages"] > 0
    assert len(doc.content) > 0


if __name__ == "__main__":
    test_clean_text_removes_page_numbers()
    test_clean_text_removes_chinese_page_numbers()
    test_clean_text_collapses_multiple_newlines()
    test_clean_text_preserves_normal_text()
    test_load_readme()
    test_load_pdf()
    print("test_loader: 全部通过")
