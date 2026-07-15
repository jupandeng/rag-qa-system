"""测试文本切分"""
import sys
import io
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.chunker import fixed_size_split, sentence_split


LONG_TEXT = "机器学习是人工智能的一个分支。" * 50


def test_fixed_size_split_creates_chunks():
    chunks = fixed_size_split(LONG_TEXT, chunk_size=200, overlap=50)
    assert len(chunks) > 1, f"期望多个 chunk，实际: {len(chunks)}"


def test_fixed_size_split_chunks_not_too_large():
    chunks = fixed_size_split(LONG_TEXT, chunk_size=200, overlap=50)
    for i, c in enumerate(chunks):
        assert len(c.content) <= 200, f"chunk[{i}] 超长: {len(c.content)}"


def test_fixed_size_split_overlap():
    chunks = fixed_size_split(LONG_TEXT, chunk_size=200, overlap=30)
    if len(chunks) >= 2:
        # 前一个 chunk 尾部应该出现在后一个 chunk 头部
        tail = chunks[0].content[-20:]
        assert tail in chunks[1].content, "overlap 未生效"


def test_sentence_split_preserves_sentences():
    text = "今天天气很好。我们出去玩吧！你觉得呢？"
    chunks = sentence_split(text, min_chunk_size=5)
    total = "".join(c.content for c in chunks)
    assert "今天天气很好" in total
    assert "我们出去玩吧" in total


def test_sentence_split_empty_returns_empty():
    chunks = sentence_split("", min_chunk_size=10)
    assert len(chunks) == 0


if __name__ == "__main__":
    test_fixed_size_split_creates_chunks()
    test_fixed_size_split_chunks_not_too_large()
    test_fixed_size_split_overlap()
    test_sentence_split_preserves_sentences()
    test_sentence_split_empty_returns_empty()
    print("test_chunker: 全部通过")
