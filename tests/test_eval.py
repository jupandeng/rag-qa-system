"""RAGAS 评估验证脚本"""
import os, sys, io, json
from pathlib import Path
from dotenv import load_dotenv

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
load_dotenv()
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline import RAGPipeline
from src.evaluator import RAGEvaluator, EvalSample, compute_mrr, compute_hit_rate


def main():
    # 1. 加载测试集
    with open("data/testset.json", "r", encoding="utf-8") as f:
        testset = json.load(f)
    print(f"加载 {len(testset)} 条测试用例\n")

    # 2. 初始化 pipeline（复用已有索引）
    pipeline = RAGPipeline("./data/index")
    pipeline.load()
    evaluator = RAGEvaluator()

    # 3. 逐条评估
    samples = []
    mrr_data = []

    for i, item in enumerate(testset, 1):
        question = item["question"]
        ground_truth = item["ground_truth"]

        print(f"[{i}/{len(testset)}] {question}")
        result = pipeline.query(question, top_k=5)

        # 构造评估样本
        sample = EvalSample(
            question=question,
            ground_truth=ground_truth,
            answer=result["answer"],
            contexts=[s["content"] for s in result["sources"]],
        )
        evaluator.evaluate(sample)
        samples.append(sample)

        # MRR 数据：检查 ground_truth 关键词是否出现在检索结果中
        gt_words = set(ground_truth[:30])  # 取前30字作为关键词指纹
        relevance = [
            any(w in s["content"][:100] for w in gt_words)
            for s in result["sources"]
        ]
        mrr_data.append(relevance)

        print(f"  Faithfulness: {sample.scores['faithfulness']:.2f}")
        print(f"  Answer Relevancy: {sample.scores['answer_relevancy']:.2f}")
        print(f"  Context Relevancy: {sample.scores['context_relevancy']:.2f}")
        print()

    # 4. 汇总报告
    report = evaluator.evaluate_all(samples)
    mrr = compute_mrr(mrr_data)
    hit5 = compute_hit_rate(mrr_data, k=5)

    print("=" * 50)
    print("评估报告")
    print("=" * 50)
    print(f"样本数:           {report['total_samples']}")
    print(f"Faithfulness:     {report['faithfulness']:.3f}  (回答是否忠于参考资料)")
    print(f"Answer Relevancy: {report['answer_relevancy']:.3f}  (回答是否切题)")
    print(f"Context Relevancy:{report['context_relevancy']:.3f}  (检索结果是否相关)")
    print(f"MRR:              {mrr:.3f}  (平均倒数排名)")
    print(f"Hit Rate@5:       {hit5:.3f}  (前5个结果命中率)")

    # 保存报告
    report["mrr"] = mrr
    report["hit_rate_at_5"] = hit5
    report_path = Path("data/eval_report.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n完整报告已保存: {report_path}")


if __name__ == "__main__":
    main()
