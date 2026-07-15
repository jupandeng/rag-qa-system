"""VLM 图片语义理解模块 —— 对 PDF 内嵌图片生成文字描述"""
import os
import base64
import io
from pathlib import Path
from PIL import Image


class ImageDescriber:
    """图片语义描述器：调用 VLM 对图片内容生成文字描述"""

    SYSTEM_PROMPT = (
        "你是一个图片内容描述助手。请用中文简洁描述图片中的内容，"
        "包括图表类型、关键数据、流程步骤或核心信息点。"
        "描述控制在 100 字以内，只输出描述内容，不要添加多余的话。"
    )

    def __init__(self):
        from openai import OpenAI
        # VLM 使用独立的 API 配置（可与 LLM 不同）
        self.llm = OpenAI(
            api_key=os.getenv("VLM_API_KEY", os.getenv("OPENAI_API_KEY")),
            base_url=os.getenv("VLM_BASE_URL", os.getenv("OPENAI_BASE_URL")),
        )
        self.model = os.getenv("VLM_MODEL", os.getenv("LLM_MODEL", "deepseek-chat"))

    def describe(self, image_bytes: bytes, image_format: str = "png") -> str:
        """对单张图片生成语义描述"""
        # 压缩大图（VLM 通常有尺寸限制）
        img = Image.open(io.BytesIO(image_bytes))
        max_size = 1568
        if max(img.size) > max_size:
            ratio = max_size / max(img.size)
            new_size = (int(img.width * ratio), int(img.height * ratio))
            img = img.resize(new_size, Image.LANCZOS)

        # 重新编码为 base64
        buffer = io.BytesIO()
        img_format = image_format.upper()
        if img_format == "JPG":
            img_format = "JPEG"
        img.save(buffer, format=img_format)
        encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")

        try:
            response = self.llm.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/{img_format.lower()};base64,{encoded}"
                                },
                            },
                            {"type": "text", "text": "请描述这张图片的内容。"},
                        ],
                    },
                ],
                max_tokens=200,
                temperature=0.2,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            # VLM 不可用时静默降级（不阻塞索引流程）
            return f"[VLM不可用: {e}]"


# ═══════════════════════════════════════════════════════════════
# PDF 图片提取工具
# ═══════════════════════════════════════════════════════════════

def extract_images_from_pdf(file_path: Path) -> list[dict]:
    """从 PDF 中提取所有图片

    Returns:
        [{"page": 1, "bytes": b"...", "format": "png"}, ...]
    """
    import fitz
    doc = fitz.open(str(file_path))
    images = []

    for page_num, page in enumerate(doc, start=1):
        img_list = page.get_images(full=True)
        for img_info in img_list:
            xref = img_info[0]
            base_image = doc.extract_image(xref)
            images.append({
                "page": page_num,
                "bytes": base_image["image"],
                "format": base_image.get("ext", "png"),
            })

    doc.close()
    return images
