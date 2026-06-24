"""Estimate per-task sampled tokens for a given prefix_config WITHOUT writing the 127GB pool.

Mirrors sample_tokenized.py logic: truncate_and_filter (truncate mode + min_resp_length),
then per-task rows_to_sample = min(max_per_file, n_rows) * repeat, sampled with replacement
across permutations. Token estimate = mean(inst_len+resp_len over filtered rows) * rows_to_sample
when sampling >= n_rows (covers all + repeats); when sampling < n_rows it is mean*rows_to_sample
in expectation (uniform without replacement within a perm). Both are exact in expectation, and
since the real sampler reshuffles, the expectation is what we compare against.
"""
import sys, json, yaml
from pathlib import Path
import numpy as np

TOK_PATH = Path("/scratch/data_io_run/data_tokenized_bpe_65k")
CONTEXT_SIZE = 4096 + 1
MIN_RESP = 2

def load_prefix_config(path):
    with open(path) as f:
        return yaml.safe_load(f)

def match_prefix(task_name, cfg_list):
    chosen = None
    for item in cfg_list:
        if task_name.startswith(item["prefix"]):
            if chosen is not None:
                break
            chosen = item
    return chosen or {}

def filtered_stats(inst_len, resp_len):
    keep = resp_len >= MIN_RESP
    allowed_resp = CONTEXT_SIZE - np.minimum(inst_len, CONTEXT_SIZE)
    keep &= allowed_resp >= 1
    resp_t = np.minimum(resp_len, allowed_resp)
    inst_f = inst_len[keep]
    resp_f = resp_t[keep]
    n = len(inst_f)
    if n == 0:
        return 0, 0.0, 0.0
    return n, float(np.sum(inst_f)), float(np.sum(resp_f))

def main():
    cfg_path = sys.argv[1]
    cfg_list = load_prefix_config(cfg_path)
    rows = []
    total = 0.0
    for d in sorted(TOK_PATH.iterdir()):
        if not d.is_dir():
            continue
        il = np.load(d / "inst_len.npy")
        rl = np.load(d / "resp_len.npy")
        n_filt, inst_sum, resp_sum = filtered_stats(il, rl)
        if n_filt == 0:
            continue
        mean_tok = (inst_sum + resp_sum) / n_filt
        item = match_prefix(d.name, cfg_list)
        max_per = item.get("max_per_file")
        repeat = item.get("repeat", 1)
        rows_capped = min(max_per, n_filt) if max_per is not None else n_filt
        rows_to_sample = rows_capped * repeat
        cap_binds = (max_per is not None and max_per < n_filt)
        pool_limits = (max_per is not None and max_per >= n_filt)  # cap set but pool smaller
        est_tokens = mean_tok * rows_to_sample
        prefix0 = d.name.split("__")[0]
        rows.append(dict(task=d.name, cat=prefix0, n_filt=n_filt, max_per=max_per,
                         repeat=repeat, rows_to_sample=rows_to_sample, mean_tok=mean_tok,
                         est_tokens=est_tokens, cap_binds=cap_binds, pool_limits=pool_limits))
        total += est_tokens
    rows.sort(key=lambda r: -r["est_tokens"])
    print(f"# Estimating for config: {cfg_path}")
    print(f"# Total estimated sampled tokens (epochs=1): {total:,.0f}")
    print()
    # category aggregation
    cat = {}
    for r in rows:
        c = cat.setdefault(r["cat"], dict(tok=0.0, rows=0))
        c["tok"] += r["est_tokens"]; c["rows"] += r["rows_to_sample"]
    print("## By category (prefix0)")
    print(f"{'category':<28} {'est_tokens':>16} {'pct':>7} {'rows':>14}")
    for c, v in sorted(cat.items(), key=lambda kv: -kv[1]["tok"]):
        print(f"{c:<28} {v['tok']:>16,.0f} {100*v['tok']/total:>6.2f}% {v['rows']:>14,}")
    print()
    print("## Datasets where cap was set but pool is the LIMITING factor (cap didn't bind):")
    for r in rows:
        if r["pool_limits"]:
            print(f"  {r['task']:<45} max_per={r['max_per']:>10,} pool_rows={r['n_filt']:>10,}")
    # dump full json for reuse
    out = dict(config=cfg_path, total=total, rows=rows, cat=cat)
    with open("/tmp/mix_estimate.json", "w") as f:
        json.dump(out, f)
    print("\n# wrote /tmp/mix_estimate.json")

if __name__ == "__main__":
    main()
