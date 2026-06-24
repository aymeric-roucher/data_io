from datasets import load_dataset

from utils import write_jsonl


# Hard, verified competition/word math (median llama8b solve-rate ~0.44). The target is the
# SHORT verified final answer (no worked solution) -> condition "direct": the answer-only,
# latent-reasoning regime (model must produce the answer with no scratchpad). ~251k rows,
# ~18M BPE-65k tokens. Uses the UNGATED open-r1 mirror (SynthLabsAI/Big-Math-RL-Verified is gated).
#
# `row["llama8b_solve_rate"]` is a difficulty proxy (lower = harder). To keep Big-Math as the HARD
# complement to easy Orca-Math, optionally drop near-trivial rows, e.g. `if sr is None or sr <= 0.9`.
dataset = load_dataset("open-r1/Big-Math-RL-Verified-Processed", "all", split="train")
result = []
for row in dataset:
    q = (row["prompt"] or "").strip()
    a = (row["solution"] or "").strip()   # the verified final answer, e.g. "10\\%"
    if q and a and "http" not in q:
        result.append({
            "condition": "direct",
            "instruction": q,
            "response": a,
        })

write_jsonl("data/big_math.jsonl", result)
