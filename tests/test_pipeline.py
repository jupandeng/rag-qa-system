"""端到端验证 RAG 全流程"""
import os, sys, io
from pathlib import Path
from dotenv import load_dotenv

# Windows 控制台 UTF-8 编码修复
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

load_dotenv()
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline import RAGPipeline


def main():
    # 1. 加载文档并建立索引
    docs_dir = Path("data/docs")
    files = list(docs_dir.glob("*.*"))
    print(f"找到 {len(files)} 个文档:")
    for f in files:
        print(f"  - {f.name}")

    pipeline = RAGPipeline("./data/index")
    pipeline.ingest(files)
    print("\n索引构建完成！\n")

    # 2. 测试提问
    questions = [
        "什么是RAG？它解决了什么问题？",
        "Python里字符串拼接为什么推荐用join？",
        "DeepSeek的API怎么接入？",
    ]
    for q in questions:
        print(f"问题: {q}")
        result = pipeline.query(q, top_k=3)
        print(f"回答: {result['answer'][:200]}...")
        print(f"引用来源数: {len(result['sources'])}")
        print()


if __name__ == "__main__":
    api_key = os.getenv("OPENAI_API_KEY", "")
    if "你的" in api_key:
        print("请先在 .env 文件中填入真实的 DeepSeek API Key")
        sys.exit(1)
    main()
