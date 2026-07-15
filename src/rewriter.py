"""查询改写与问题路由 —— 提升检索命中率的关键模块"""
import os
from openai import OpenAI


class QueryRewriter:
    """查询改写：将模糊问题转换为更精准的检索 query"""

    REWRITE_PROMPT = """你是一个查询改写助手。将用户的原始问题改写成 2-3 个更适合检索的查询语句。

## 改写策略
- 补充上下文：如果问题里代词不明确（"它""这个"），补全具体指代
- 同义词扩展：用不同术语表达同一个意思（如 "怎么用"→"使用教程""接入方法"）
- 语义拆分：复合问题拆成多个单一问题

## 输出格式
每行一个查询，不要编号，不要其他文字。

## 示例
用户: 它怎么用？
改写:
DeepSeek API 使用教程
DeepSeek 接入方法

用户: {question}
改写:"""

    def __init__(self):
        self.llm = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL"),
        )
        self.model = os.getenv("LLM_MODEL", "deepseek-chat")

    def rewrite(self, question: str) -> list[str]:
        """返回改写后的多个查询（含原问题）"""
        try:
            prompt = self.REWRITE_PROMPT.format(question=question)
            response = self.llm.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.4,
                max_tokens=200,
            )
            rewrites = response.choices[0].message.content.strip().split("\n")
            # 过滤空行
            rewrites = [r.strip() for r in rewrites if r.strip()]
        except Exception:
            rewrites = []

        # 保证原问题一定在列表里
        if question not in rewrites:
            rewrites.insert(0, question)
        return rewrites[:4]  # 最多 4 个查询


class QuestionRouter:
    """问题路由：判断问题类型，选择检索策略"""

    ROUTE_PROMPT = """你是一个问题分类器。判断用户问题的类型，输出一个标签。

## 类型说明
- factual: 事实查询，查"是什么""谁""什么时候"（如 "RAG是什么"）
- procedural: 操作步骤，查"怎么做""如何配置"（如 "怎么接入API"）
- comparison: 对比分析，查"区别""优缺点"（如 "join和+=的区别"）
- code: 代码相关，涉及编程语法或实现（如 "Python里怎么写列表推导式"）

## 输出
只输出一个词：factual / procedural / comparison / code

用户问题: {question}
类型:"""

    def __init__(self):
        self.llm = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL"),
        )
        self.model = os.getenv("LLM_MODEL", "deepseek-chat")

    def route(self, question: str) -> str:
        """返回问题类型标签"""
        try:
            prompt = self.ROUTE_PROMPT.format(question=question)
            response = self.llm.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=20,
            )
            label = response.choices[0].message.content.strip().lower()
            if label in ("factual", "procedural", "comparison", "code"):
                return label
        except Exception:
            pass
        return "factual"  # 默认按事实查询处理
