"""Produce a down-scaled copy of the FAITHFUL prefix_config that preserves proportions.

Scales every max_per_file by f = TARGET/estimated_faithful_total (keeping >= 1),
leaves `repeat` (the x10 HQ upsampling) and prefix order untouched so relative
proportions are preserved. Writes a YAML compatible with sample_tokenized.py.
"""
import sys, yaml, json, math
from pathlib import Path

FAITHFUL = Path("/root/data_io/prefix_config.yaml")
TARGET = 2_000_000_000

def main():
    f = float(sys.argv[1])          # scale factor applied to every max_per_file
    out_path = sys.argv[2]
    with open(FAITHFUL) as fh:
        cfg = yaml.safe_load(fh)
    scaled = []
    for item in cfg:
        new = dict(item)
        if "max_per_file" in new and new["max_per_file"] is not None:
            new["max_per_file"] = max(1, int(round(new["max_per_file"] * f)))
        scaled.append(new)
    with open(out_path, "w") as fh:
        fh.write(f"# Down-scaled copy of FAITHFUL prefix_config.yaml (paper Table 6 proportions).\n")
        fh.write(f"# Every max_per_file multiplied by f = {f:.6f}; solved so total -> ~{TARGET:,} tokens.\n")
        fh.write(f"# x10 HQ `repeat` values preserved verbatim -> relative proportions unchanged.\n")
        fh.write(f"# First match applies (same prefix order as faithful config).\n\n")
        yaml.safe_dump(scaled, fh, sort_keys=False, default_flow_style=False)
    print(f"scale factor f = {f:.6f}")
    print(f"wrote {out_path}")

if __name__ == "__main__":
    main()
