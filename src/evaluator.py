"""RAG 测评模块 —— 检索与生成联合评估"""
import json
from pathlib import Path
from dataclasses import dataclass, field, asdict
from openai import OpenAI
import os


@dataclass
class EvalSample:
    """单个评估样本"""
    question: str
    ground_truth: str           # 参考答案
    answer: str = ""            # 系统生成答案
    contexts: list[str] = field(default_factory=list)  # 检索到的上下文
    scores: dict = field(default_factory=dict)          # 各项评分


class RAGEvaluator:
    """RAG 评估器：Faithfulness / AnswerRelevancy / ContextRelevancy"""

    FAITHFULNESS_PROMPT = """你是一个评估助手。判断以下"回答"中的每一句话是否都能在"参考资料"中找到依据。

## 参考资料
{context}

## 回答
{answer}

## 评分规则
- 输出一个 0-1 之间的分数
- 1.0: 回答中的所有陈述都能在参考资料中找到明确依据
- 0.5: 大部分可找到依据，少部分可能为推断
- 0.0: 大部分为编造，与参考资料无关

只输出数字，不要其他内容。
分数:"""

    ANSWER_RELEVANCY_PROMPT = """你是一个评估助手。判断以下"回答"是否直接回应了"问题"。

## 问题
{question}

## 回答
{answer}

## 评分规则
- 输出一个 0-1 之间的分数
- 1.0: 回答完全针对问题，无无关内容
- 0.5: 部分相关，但包含不必要的信息
- 0.0: 答非所问

只输出数字，不要其他内容。
分数:"""

    CONTEXT_RELEVANCY_PROMPT = """你是一个评估助手。判断以下"参考资料"是否与"问题"相关。

## 问题
{question}

## 参考资料
{context}

## 评分规则
- 输出一个 0-1 之间的分数
- 1.0: 参考资料高度相关，包含回答问题所需信息
- 0.5: 部分相关，信息不完整
- 0.0: 完全不相关

只输出数字，不要其他内容。
分数:"""

    def __init__(self):
        self.llm = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL"),
        )
        self.model = os.getenv("LLM_MODEL", "deepseek-chat")

    def evaluate(self, sample: EvalSample) -> EvalSample:
        """对单个样本执行三项评估"""
        context_text = "\n\n".join(sample.contexts)

        sample.scores["faithfulness"] = self._judge(
            self.FAITHFULNESS_PROMPT.format(
                context=context_text, answer=sample.answer
            )
        )
        sample.scores["answer_relevancy"] = self._judge(
            self.ANSWER_RELEVANCY_PROMPT.format(
                question=sample.question, answer=sample.answer
            )
        )
        sample.scores["context_relevancy"] = self._judge(
            self.CONTEXT_RELEVANCY_PROMPT.format(
                question=sample.question, context=context_text
            )
        )
        return sample

    def _judge(self, prompt: str) -> float:
        """调用 LLM 打分"""
        try:
            response = self.llm.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=10,
            )
            score = float(response.choices[0].message.content.strip())
            return max(0.0, min(1.0, score))
        except Exception:
            return 0.0

    def evaluate_all(self, samples: list[EvalSample]) -> dict:
        """批量评估，返回汇总指标"""
        for s in samples:
            self.evaluate(s)

        n = len(samples)
        report = {
            "total_samples": n,
            "faithfulness": sum(s.scores.get("faithfulness", 0) for s in samples) / n,
            "answer_relevancy": sum(s.scores.get("answer_relevancy", 0) for s in samples) / n,
            "context_relevancy": sum(s.scores.get("context_relevancy", 0) for s in samples) / n,
            "details": [asdict(s) for s in samples],
        }
        return report


def load_testset(file_path: Path) -> list[dict]:
    """加载测试集（JSON 格式）"""
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


# ═══════════════════════════════════════════════════════════════
# MRR 和 Hit Rate（不需要 LLM，直接算）
# ═══════════════════════════════════════════════════════════════

def compute_mrr(retrieval_results: list[list[bool]]) -> float:
    """计算 Mean Reciprocal Rank

    retrieval_results: 每个问题一个列表，bool 表示该位置的文档是否相关
    例: [[False, True, False], [True, False]] → 第一个在第2位命中，第二个在第1位命中
    MRR = (1/2 + 1/1) / 2 = 0.75
    """
    total = 0.0
    for results in retrieval_results:
        for rank, relevant in enumerate(results, start=1):
            if relevant:
                total += 1.0 / rank
                break
    return total / len(retrieval_results) if retrieval_results else 0.0


def compute_hit_rate(retrieval_results: list[list[bool]], k: int = 5) -> float:
    """计算 Hit Rate@K：前 K 个结果中至少有一个相关的比例"""
    hits = sum(
        1 for results in retrieval_results
        if any(results[:k])
    )
    return hits / len(retrieval_results) if retrieval_results else 0.0
