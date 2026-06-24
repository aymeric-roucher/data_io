"""Compute per-dataset (prefix0 category) token proportions of a sampled dir's epoch_0.

The sampler lays tasks out contiguously in tokens.npy in sorted(task name) order, each task
occupying [offset, offset+sum(inst_len)+sum(resp_len)). inst_start/resp_start in epoch_N already
have the per-task mmap_base_offset added. So we rebuild the same offset boundaries from the SOURCE
(sorted task dirs, summing raw inst_len+resp_len before filtering -- which is exactly what
concat_tokens uses for mmap_length), then bucket each epoch_0 row's inst_start into its task and add
inst_len+resp_len for that row. Category = task.split('__')[0].
"""
import sys, json
from pathlib import Path
import numpy as np

TOK_PATH = Path("/scratch/data_io_run/data_tokenized_bpe_65k")

def build_offsets():
    """Replicate concat_tokens offset layout: sorted dirs, mmap_length = sum(inst_len)+sum(resp_len) (raw, pre-filter)."""
    names, starts, ends = [], [], []
    off = 0
    for d in sorted(TOK_PATH.iterdir()):
        if not d.is_dir():
            continue
        il = np.load(d / "inst_len.npy"); rl = np.load(d / "resp_len.npy")
        length = int(il.sum() + rl.sum())
        names.append(d.name); starts.append(off); ends.append(off + length)
        off += length
    return names, np.array(starts, dtype=np.int64), np.array(ends, dtype=np.int64)

def proportions(sampled_dir, names, starts):
    ep = Path(sampled_dir) / "epoch_0"
    inst_start = np.load(ep / "inst_start.npy")
    inst_len = np.load(ep / "inst_len.npy")
    resp_len = np.load(ep / "resp_len.npy")
    toks = inst_start  # bucket by inst_start (lies in [task_start, task_end))
    # searchsorted: task index = number of task-starts <= inst_start, minus 1
    idx = np.searchsorted(starts, toks, side="right") - 1
    idx = np.clip(idx, 0, len(names) - 1)
    row_tok = inst_len.astype(np.int64) + resp_len.astype(np.int64)
    cat_tok = {}
    task_tok = {}
    for i in range(len(names)):
        mask = idx == i
        if not mask.any():
            continue
        t = int(row_tok[mask].sum())
        task_tok[names[i]] = task_tok.get(names[i], 0) + t
        cat = names[i].split("__")[0]
        cat_tok[cat] = cat_tok.get(cat, 0) + t
    return cat_tok, task_tok, int(row_tok.sum())

def main():
    new_dir = sys.argv[1]
    old_dir = sys.argv[2]
    names, starts, ends = build_offsets()
    new_cat, new_task, new_total = proportions(new_dir, names, starts)
    old_cat, old_task, old_total = proportions(old_dir, names, starts)
    print(f"NEW total epoch_0 tokens: {new_total:,}")
    print(f"OLD total epoch_0 tokens: {old_total:,}")
    print()
    MATH = {"openmathinstruct2","acereason","openthoughts2","sudoku_extreme","dmmath",
            "ampsmathematica","numinamath.jsonl","omnimath.jsonl","gsm8k_train.jsonl",
            "math_train.jsonl","amps_khan.jsonl","webinstruct_verified.jsonl",
            "natural_reasoning.jsonl","principia_collection.jsonl","textbookreasoning",
            "Platypus","SYNTH"}
    cats = sorted(set(new_cat) | set(old_cat), key=lambda c: -new_cat.get(c, 0))
    print(f"{'category':<30} {'NEW %':>8} {'OLD %':>8} {'NEW tok':>16} {'OLD tok':>16} {'incr?':>6}")
    new_math = old_math = 0.0
    for c in cats:
        np_ = 100*new_cat.get(c,0)/new_total
        op_ = 100*old_cat.get(c,0)/old_total
        if c in MATH:
            new_math += np_; old_math += op_
        incr = "UP" if new_cat.get(c,0)/max(1,new_total) > old_cat.get(c,0)/max(1,old_total) else ""
        print(f"{c:<30} {np_:>7.2f}% {op_:>7.2f}% {new_cat.get(c,0):>16,} {old_cat.get(c,0):>16,} {incr:>6}")
    print()
    print(f"MATH/REASONING share  NEW={new_math:.2f}%  OLD={old_math:.2f}%  delta={new_math-old_math:+.2f}pp")
    # x10 HQ presence check
    print("\nx10 HQ sets present in NEW (token counts):")
    for k in ["Platypus","omnimath.jsonl","gsm8k_train.jsonl","math_train.jsonl",
              "webinstruct_verified.jsonl","no_robots.jsonl"]:
        print(f"  {k:<30} {new_cat.get(k,0):>14,}  (old {old_cat.get(k,0):,})")

if __name__ == "__main__":
    main()
