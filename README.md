# RAG 知识库问答系统

基于检索增强生成（RAG）的智能知识库问答系统，支持多格式文档、混合检索、查询重写、增量索引和 VLM 图片理解。

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 架构

```
用户提问
   │
   ▼
┌──────────────┐
│  查询重写      │  ← LLM 多角度扩写
│  QuestionRouter │  ← 问题类型分类
└──────┬───────┘
       │
       ▼
┌──────────────┐
│  分层缓存      │  ← LRU + FAQ 持久化
└──────┬───────┘
       │  miss
       ▼
┌──────────────┐
│  混合检索      │  ← Dense (BGE) + Sparse (BM25)
│  + Rerank     │  ← CrossEncoder 重排序
└──────┬───────┘
       │
       ▼
┌──────────────┐
│  答案生成      │  ← LLM + 来源引用验证
└──────────────┘
```

## 技术栈

| 层次 | 技术 |
|------|------|
| LLM | DeepSeek / OpenAI 兼容 API |
| 向量模型 | BAAI/bge-small-zh-v1.5（本地部署） |
| 向量数据库 | ChromaDB |
| 稀疏检索 | BM25（1-gram + 2-gram 中文分词） |
| 精排模型 | BAAI/bge-reranker-v2-m3（CrossEncoder） |
| 文档解析 | PyMuPDF + PaddleOCR + VLM |
| 测评 | RAGAS（忠实度/答案相关性/上下文相关性） |
| API | FastAPI |
| 缓存 | 三层缓存：Embedding LRU + FAQ 持久化 + QA LRU |

## 功能

- 多格式文档加载 — PDF / Markdown / TXT，支持目录批量导入
- 文档预处理 — 自动清洗页眉页脚、去水印、OCR 文字提取
- 智能切分 — 固定长度 / 句边界 / Markdown 标题三种策略
- 增量索引 — SHA256 哈希检测文件变更，自动增量更新
- VLM 图片理解 — 对文档中图片生成语义描述，提升图文混合检索
- 查询重写 — LLM 将问题扩写为 2-3 种表述，多路并行检索
- 问题路由 — 自动分类为 事实型/步骤型/对比型/代码型
- 混合检索 — 稠密向量 + BM25 稀疏检索 + CrossEncoder 重排序
- 来源引用 — 回答附带文档名称和页码，可验证
- 分层缓存 — 命中缓存时毫秒级响应

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 配置环境变量
cp .env.example .env
# 编辑 .env，填入 OPENAI_API_KEY

# 3. 导入文档
mkdir -p data/docs
# 将你的 PDF / MD / TXT 文件放入 data/docs/
python -c "from src.pipeline import RAGPipeline; p = RAGPipeline(); p.ingest()"

# 4. 启动服务
uvicorn src.server:app --reload
# 访问 http://localhost:8000/docs 查看 API 文档

# 5. 提问
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "你的问题"}'
```

## API

| 接口 | 方法 | 说明 |
|------|------|------|
| `/health` | GET | 健康检查 |
| `/ingest` | POST | 上传文档并构建索引 |
| `/ask` | POST | 提问 |

## 测评结果

| 指标 | 分数 | 说明 |
|------|------|------|
| Faithfulness | 1.0 | 答案忠实度 |
| Answer Relevancy | 1.0 | 答案相关性 |
| Context Relevancy | 1.0 | 上下文相关性 |
| MRR | 1.0 | 平均倒数排名 |
| Hit Rate@3 | 1.0 | 前三命中率 |

## 项目结构

```
rag-qa-system/
├── src/
│   ├── loader.py      # 文档加载（PDF/MD/TXT + OCR + VLM）
│   ├── chunker.py      # 文本切分（固定/句/标题）
│   ├── embedder.py     # 向量化（BGE 本地 / OpenAI）
│   ├── retriever.py    # 混合检索（Dense + BM25 + Rerank）
│   ├── rewriter.py     # 查询重写 + 问题路由
│   ├── evaluator.py    # RAGAS 测评
│   ├── indexer.py      # 增量索引（SHA256）
│   ├── cache.py        # 分层缓存（LRU + FAQ）
│   ├── vision.py       # VLM 图片描述
│   ├── pipeline.py     # 主流程编排
│   └── server.py       # FastAPI 服务
├── data/
│   ├── docs/           # 文档目录
│   └── testset.json    # 测试集
└── tests/              # 测试脚本
```
