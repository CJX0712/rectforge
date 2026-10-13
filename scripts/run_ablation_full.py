"""完整配置消融 + 确定性（与全量基准并行，产物合并进最终 artifact）。"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ[_v] = "1"

import json

from rectforge.core.config import load_config
from rectforge.pipeline.benchmark import run_ablation, run_determinism_check


def main():
    cfg = load_config()
    out = {
        "ablation": run_ablation(cfg, verbose=True),
        "determinism": run_determinism_check(verbose=True),
    }
    with open("ablation_full.json", "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=2)
    print("ablation_full.json written")


if __name__ == "__main__":
    main()
