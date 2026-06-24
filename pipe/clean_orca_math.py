from datasets import load_dataset

from utils import write_jsonl


# Easy/GSM8k-level word problems, GPT-4-generated WORKED SOLUTIONS (no <think> to strip).
# The solution is the response -> condition "cot". ~200k rows, ~79M BPE-65k tokens.
dataset = load_dataset("microsoft/orca-math-word-problems-200k", split="train")
result = []
for row in dataset:
    q = (row["question"] or "").strip()
    a = (row["answer"] or "").strip()
    if q and a and "http" not in q:
        result.append({
            "condition": "cot",
            "instruction": q,
            "response": a,
        })

write_jsonl("data/orca_math.jsonl", result)
