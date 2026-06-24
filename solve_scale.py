"""Solve for the scale factor f on max_per_file so total -> TARGET.

The faithful total = SCALABLE(f) + FIXED, where:
 - SCALABLE = sum over tasks whose cap BINDS (max_per_file < pool rows): tokens ~ proportional to
   max_per_file, so they scale ~linearly with f (until the cap stops binding and the pool floor hits).
 - FIXED = sum over tasks with no cap, or cap that doesn't bind (pool-limited), or repeat-only (x10 HQ).
   These DON'T shrink when we scale max_per_file.
We can't keep the EXACT paper proportions and hit 2B, because the fixed/pool-limited mass alone may
exceed or approach 2B. Best we can do: scale the binding caps to bring total to TARGET, accepting that
the uncapped/pool-limited tail is whatever the pool holds. We binary-search f over the real estimator.
"""
import sys, yaml, json
from pathlib import Path
import numpy as np

TOK_PATH = Path("/scratch/data_io_run/data_tokenized_bpe_65k")
FAITHFUL = Path("/root/data_io/prefix_config.yaml")
CONTEXT_SIZE = 4096 + 1
MIN_RESP = 2
TARGET = 2_000_000_000

def match_prefix(task_name, cfg_list):
    chosen = None
    for item in cfg_list:
        if task_name.startswith(item["prefix"]):
            if chosen is not None:
                break
            chosen = item
    return chosen or {}

# Precompute per-task filtered (n_rows, mean_tok) once.
def build_tasks():
    tasks = []
    for d in sorted(TOK_PATH.iterdir()):
        if not d.is_dir():
            continue
        il = np.load(d / "inst_len.npy"); rl = np.load(d / "resp_len.npy")
        keep = rl >= MIN_RESP
        allowed = CONTEXT_SIZE - np.minimum(il, CONTEXT_SIZE)
        keep &= allowed >= 1
        resp_t = np.minimum(rl, allowed)
        inst_f = il[keep]; resp_f = resp_t[keep]
        n = len(inst_f)
        if n == 0:
            continue
        mean_tok = float(np.sum(inst_f) + np.sum(resp_f)) / n
        tasks.append((d.name, n, mean_tok))
    return tasks

def total_for_f(tasks, cfg_list, f):
    tot = 0.0
    for name, n, mean_tok in tasks:
        item = match_prefix(name, cfg_list)
        max_per = item.get("max_per_file")
        repeat = item.get("repeat", 1)
        if max_per is not None:
            scaled_cap = max(1, int(round(max_per * f)))
            rows_capped = min(scaled_cap, n)
        else:
            rows_capped = n
        tot += mean_tok * rows_capped * repeat
    return tot

def main():
    tasks = build_tasks()
    with open(FAITHFUL) as fh:
        cfg = yaml.safe_load(fh)
    # Report the floor (f -> 0: every capped task = 1 row * repeat; uncapped/repeat = full)
    floor = total_for_f(tasks, cfg, 1e-9)
    full = total_for_f(tasks, cfg, 1.0)
    print(f"floor (f->0, caps=1 row): {floor:,.0f}")
    print(f"full  (f=1, faithful):    {full:,.0f}")
    print(f"TARGET:                   {TARGET:,}")
    if floor > TARGET:
        print("!! Floor already exceeds TARGET: cannot reach 2B by scaling caps alone.")
    # binary search f in [0, 1]
    lo, hi = 0.0, 1.0
    for _ in range(60):
        mid = (lo + hi) / 2
        t = total_for_f(tasks, cfg, mid)
        if t > TARGET:
            hi = mid
        else:
            lo = mid
    f = (lo + hi) / 2
    print(f"\nsolved f = {f:.6f} -> total = {total_for_f(tasks, cfg, f):,.0f}")

if __name__ == "__main__":
    main()
