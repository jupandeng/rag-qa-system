"""文本切分模块 —— 支持固定长度、语义分块、Markdown 感知切分"""
import re
from dataclasses import dataclass, field


@dataclass
class Chunk:
    content: str
    metadata: dict = field(default_factory=dict)


def fixed_size_split(
    text: str, chunk_size: int = 512, overlap: int = 64
) -> list[Chunk]:
    """固定长度切分（按字符），带重叠"""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk_text = text[start:end]
        chunks.append(Chunk(content=chunk_text))
        start += chunk_size - overlap
    return chunks


def sentence_split(
    text: str, max_chunk_size: int = 1024, min_chunk_size: int = 128
) -> list[Chunk]:
    """按句切分，尽量保持语义完整，超长句再降级为固定长度切割"""
    sentences = re.split(r"(?<=[。！？.!?\n])\s*", text)
    chunks = []
    buffer = ""
    for sent in sentences:
        sent = sent.strip()
        if not sent:
            continue
        # 单句太长就单独切成多个 chunk
        if len(sent) > max_chunk_size:
            if buffer:
                chunks.append(Chunk(content=buffer))
                buffer = ""
            chunks.extend(fixed_size_split(sent, max_chunk_size, 64))
            continue
        # 加上这句不超限
        if len(buffer) + len(sent) <= max_chunk_size:
            buffer += sent
        else:
            if len(buffer) >= min_chunk_size:
                chunks.append(Chunk(content=buffer))
                buffer = sent
            else:
                buffer += sent
    if buffer.strip():
        chunks.append(Chunk(content=buffer))
    return chunks


def markdown_split(text: str, max_chunk_size: int = 1024) -> list[Chunk]:
    """Markdown 感知切分：按标题层级切，段落内部按句切"""
    # 按 ## 标题分割
    sections = re.split(r"\n(?=#{1,3}\s)", text)
    chunks = []
    for section in sections:
        # 找标题
        title_match = re.match(r"(#{1,3}\s.+?)\n", section)
        title = title_match.group(1) if title_match else ""
        body = section[title_match.end() :] if title_match else section
        # body 部分按句切
        sub_chunks = sentence_split(body, max_chunk_size)
        for i, sc in enumerate(sub_chunks):
            # 只有第一个子块带标题
            if i == 0 and title:
                sc.metadata["heading"] = title
            chunks.append(sc)
    return chunks


def auto_split(text: str, file_type: str = "txt") -> list[Chunk]:
    """根据文件类型自动选择切分策略"""
    if file_type == "markdown":
        chunks = markdown_split(text)
    else:
        chunks = sentence_split(text)

    # 尝试从文本中提取页码标记 [第N页]，注入到 chunk 元数据
    _inject_page_numbers(text, chunks)
    return chunks


def _inject_page_numbers(text: str, chunks: list[Chunk]):
    """根据 [第N页] 标记，为每个 chunk 标注来源页码"""
    # 找到所有页码标记的位置
    page_positions = [
        (m.start(), int(m.group(1)))
        for m in re.finditer(r"\[第(\d+)页\]", text)
    ]
    if not page_positions:
        return

    for chunk in chunks:
        chunk_start = text.find(chunk.content[:30])
        if chunk_start == -1:
            continue
        # 找到这个 chunk 之前的最后一个页码标记
        page = 1
        for pos, pn in page_positions:
            if pos <= chunk_start:
                page = pn
            else:
                break
        chunk.metadata["page"] = page
