"""
test_transformer_sweep_dense_to_sparse_crazyflie.py
====================================================
Evaluate a saved CrazyFlie transformer CBF checkpoint across a sweep of
obstacle densities, ordered from DENSE to SPARSE.

Training configuration: area_size=3.0 m (l=3), 3-D cubic world.

Motivation
----------
The policy is trained at rho=1.0 (27 obstacles in a 3×3×3 m cube).
Evaluation progressively removes obstacles to test how the policy degrades
as the environment becomes less cluttered.

Density definition (volumetric, 3-D cubic world)
-------------------------------------------------
    rho = n_obs / area_size^3   (area_size=3.0 → volume=27 m^3)

Default sweep — densest first (all at l=3):
    rho=1.00 : n_obs=27   (training density)
    rho=0.75 : n_obs=20   (25% obstacles removed)
    rho=0.50 : n_obs=13   (50% obstacles removed)

Episode length: 256 steps (matches training default at l=3).
Pass --max-step to override.

Usage
-----
    python test_transformer_sweep_dense_to_sparse_crazyflie.py \\
        --path ./logs/CrazyFlie/gcbf_transformer/seed0_XXXXXXXX \\
        --seeds 0 1 2 --n-env 32 --parallel --log

    # Custom sweep (e.g. finer density steps):
    python test_transformer_sweep_dense_to_sparse_crazyflie.py --path ... \\
        --configs "3,27 3,20 3,13 3,7"

    # Override obstacle-size range:
    python test_transformer_sweep_dense_to_sparse_crazyflie.py --path ... \\
        --obs-len-range 0.1 0.4

The script calls test() from test_transformer_eval.py — no logic
is duplicated. A Table-style summary is printed at the end when --log is on.
"""

import argparse
import copy
import datetime
import os
from typing import List, Tuple

from test_transformer_eval import test as eval_test
import numpy as np


# ── Default sweep: dense (rho=1.0) → sparse (rho=0.50) at l=3 ────────────────
# Training was done at area_size=3.0 (l=3), so all configs keep l=3 fixed.
# n_obs = round(rho * 3^3) = round(rho * 27).
# Each entry is (area_size, n_obs, rho_label).
DEFAULT_SWEEP: List[Tuple[float, int, float]] = [
    (3.0, 27, 1.00),   # training density — full obstacle load
    (3.0, 20, 0.75),   # 25% obstacles removed
    (3.0, 13, 0.50),   # 50% obstacles removed
    # (4.0, 64, 1.00),   # training density — full obstacle load
    # (4.0, 48, 0.75),   # 25% obstacles removed
    # (4.0, 32, 0.50),   # 50% obstacles removed
]


def _parse_configs(raw: str) -> List[Tuple[float, int, float]]:
    """Parse '--configs "4,64 3,27 3,20"' → [(4.0,64,rho), (3.0,27,rho), ...]."""
    out = []
    for tok in raw.split():
        l_str, n_str = tok.split(",")
        l_val = float(l_str)
        n_val = int(n_str)
        rho   = n_val / (l_val ** 3)
        out.append((l_val, n_val, rho))
    return out


def _default_max_step(area_size: float) -> int:
    """Episode length scales with area^2; 256 steps at l=3 (training default)."""
    return int(round(256 * (area_size / 3.0) ** 2))


def main():
    parser = argparse.ArgumentParser(
        description="CrazyFlie sweep: dense (rho=1.0) → sparse (rho=0.50)."
    )

    # ── Forwarded verbatim to test_transformer_eval_crazyflie.test() ────────
    parser.add_argument("--path",        type=str,   required=True,
                        help="Run directory containing config.yaml and models/")
    parser.add_argument("--step",        type=int,   default=None,
                        help="Checkpoint step to load (default: latest)")
    parser.add_argument("-n", "--num-agents", type=int, default=1)
    parser.add_argument("--max-step",    type=int,   default=None,
                        help="Override episode length for every config "
                             "(default: scales with area^2, 256 at l=3)")
    parser.add_argument("--n-env",       type=int,   default=32,
                        help="Episodes per seed (default: 32)")
    parser.add_argument("--epi",         type=int,   default=32)
    parser.add_argument("--seed",        type=int,   default=1234)
    parser.add_argument("--seeds",       type=int,   nargs="+", default=None,
                        help="Multi-seed eval, e.g. --seeds 0 1 2.")
    parser.add_argument("--n-rays",      type=int,   default=None,
                        help="LiDAR ray count (overrides config.yaml)")
    parser.add_argument("--obs-len-range", type=float, nargs=2, default=None,
                        metavar=("MIN", "MAX"),
                        help="Obstacle diameter range in metres "
                             "(default: [0.1, 0.6] from CrazyFlie.PARAMS)")
    parser.add_argument("--cpu",         action="store_true", default=False)
    parser.add_argument("--debug",       action="store_true", default=False)
    parser.add_argument("--parallel",    action="store_true", default=False,
                        help="Run all episodes in parallel with jax.vmap")
    parser.add_argument("--no-video",    action="store_true", default=True,
                        help="Default ON for sweeps (disables per-config video).")
    parser.add_argument("--log",         action="store_true", default=False,
                        help="Append per-config results to test_log.csv and "
                             "print a Table-style summary at the end.")
    parser.add_argument("--action-scale", type=float, default=1.0,
                        help="Multiplier on delta_u: u = u_nom + scale*delta_u")
    parser.add_argument("--pretrain-ckpt", type=str, default=None)

    # ── Sweep control ────────────────────────────────────────────────────────
    parser.add_argument("--configs",     type=str,   default=None,
                        help='Custom sweep, space-separated "l,n_obs" pairs '
                             'in the order they should run. '
                             'Example: "4,64 3,27 3,20 3,13"')

    args = parser.parse_args()

    configs = _parse_configs(args.configs) if args.configs else DEFAULT_SWEEP

    stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n===== CrazyFlie: Dense → Sparse sweep  (training: l=3, rho=1.0) =====")
    print(f"  path={args.path}")
    print(f"  seeds={args.seeds or [args.seed]}  epi_per_seed={args.n_env or args.epi}")
    print(f"  configs (densest first):")
    for l_val, n_val, rho in configs:
        ms = args.max_step if args.max_step is not None else _default_max_step(l_val)
        print(f"     rho={rho:.2f}  l={l_val:.0f}  n_obs={n_val:3d}  max_step={ms}")
    print("=" * 60)

    summary_rows = []   # (rho, l, n_obs, safe%, safe_std, finish%, finish_std, succ%, succ_std)

    for i, (l_val, n_obs, rho) in enumerate(configs):
        max_step = args.max_step if args.max_step is not None else _default_max_step(l_val)
        print(f"\n>>> [{i+1}/{len(configs)}]  rho={rho:.2f}  l={l_val:.0f}  "
              f"n_obs={n_obs}  max_step={max_step}")

        # Build per-config args namespace — copy everything then override.
        sub = copy.deepcopy(args)
        sub.area_size      = float(l_val)
        sub.n_obs          = int(n_obs)
        sub.max_step       = int(max_step)
        # Fields expected by test_transformer_eval.test() not in sweep argparse:
        sub.env_id         = "CrazyFlie"
        sub.pretrain_ckpt  = args.pretrain_ckpt
        sub.use_alt_u_ref  = False
        sub.use_dyn_head   = False
        sub.cbf_video      = False
        sub.epi_list       = None
        sub.dpi            = 100
        sub.obs_len_range  = args.obs_len_range   # None → uses _ENV_DEFAULTS [0.1, 0.6]

        eval_test(sub)

        # Read back the latest aggregate CSV row when --log is on.
        if args.log:
            log_path = os.path.join(args.path, "test_log.csv")
            if os.path.exists(log_path):
                with open(log_path) as f:
                    lines = [ln.strip() for ln in f if ln.strip()]
                # Header on line 0; search backward for matching aggregate row.
                # CSV columns (0-indexed):
                #  0:timestamp 1:env_id 2:ckpt_step 3:n_agents
                #  4:n_obs 5:n_rays 6:area_size 7:max_step
                #  8:row_type 9:seed 10:epi_count
                #  11:finish_pct 12:safe_pct 13:success_pct
                #  14:finish_std_pct 15:safe_std_pct 16:success_std_pct
                for ln in reversed(lines):
                    parts = ln.split(",")
                    if len(parts) < 17:
                        continue
                    if (parts[8] == "aggregate"
                            and int(float(parts[4])) == int(n_obs)
                            and abs(float(parts[6]) - float(l_val)) < 1e-6):
                        safe_pct   = float(parts[12])
                        safe_std   = float(parts[15])
                        fin_pct    = float(parts[11])
                        fin_std    = float(parts[14])
                        succ_pct   = float(parts[13])
                        succ_std   = float(parts[16])
                        summary_rows.append((
                            rho, l_val, n_obs,
                            safe_pct,  safe_std,
                            fin_pct,   fin_std,
                            succ_pct,  succ_std,
                        ))
                        break

    # ── Final summary table ────────────────────────────────────────────────────
    if summary_rows:
        print("\n" + "=" * 82)
        print(f" CrazyFlie dense→sparse sweep summary   {stamp}")
        print(f" path={args.path}")
        print("=" * 82)
        print(f"  {'rho':>5}  {'l':>3}  {'n_obs':>5}  "
              f"{'Safety%':>16}  {'Reaching%':>16}  {'Success%':>16}")
        print("-" * 82)
        prev_rho = None
        for (rho, l_val, n_obs,
             safe, safe_s, fin, fin_s, succ, succ_s) in summary_rows:
            if prev_rho is not None and abs(rho - prev_rho) > 1e-4:
                print()   # blank line between density bands
            print(f"  {rho:>5.2f}  {l_val:>3.0f}  {n_obs:>5d}  "
                  f"{safe:>7.2f} ± {safe_s:<5.2f}  "
                  f"{fin:>7.2f} ± {fin_s:<5.2f}  "
                  f"{succ:>7.2f} ± {succ_s:<5.2f}")
            prev_rho = rho
        print("=" * 82)
    elif not args.log:
        print("\n(Tip: pass --log to also print a summary table at the end.)")


if __name__ == "__main__":
    main()
