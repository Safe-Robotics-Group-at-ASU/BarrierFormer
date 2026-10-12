"""SQP teacher diagnostics — convergence and slack instrumentation."""

from __future__ import annotations

import csv
import os
from typing import NamedTuple, Optional, Sequence

import numpy as np

from barrierformer.utils.typing import Array


SQP_SLACK_TOL: float = 1e-8

SQP_CONV_THRESH: float = 0.05

QP_PRI_RES_THRESH: float = 8e-4
QP_DUA_RES_THRESH: float = 1e-3

EXIT_CONVERGED: int = 0
EXIT_MAX_ITERS: int = 1
EXIT_QP_INFEASIBLE: int = 2
EXIT_NONFINITE: int = 3

EXIT_NAMES = {
    EXIT_CONVERGED: "converged",
    EXIT_MAX_ITERS: "max_iters",
    EXIT_QP_INFEASIBLE: "qp_infeasible",
    EXIT_NONFINITE: "nonfinite",
}


class SQPSolveDiagnostics(NamedTuple):
    """Per-solve diagnostics."""

    converged_strict: Array
    converged_thresh: Array
    n_iters: Array
    budget_hit: Array
    exit_code: Array
    has_nonfinite: Array

    xi_max: Array
    xi_sum: Array
    xi_nonzero_count: Array
    xi_max_over_eta_s: Array
    xi_below_eta_s: Array
    violates_untightened: Array
    untightened_margin_max: Array

    xi_qp_max: Array
    xi_qp_sum: Array
    xi_qp_nonzero_count: Array
    xi_qp_max_over_eta_s: Array
    xi_qp_exceeds_eta_s: Array

    qp_pri_res: Array
    qp_dua_res: Array
    qp_degraded: Array


def _frac(mask: np.ndarray) -> float:
    """Fraction of True, 0.0 for an empty array."""
    return float(np.mean(mask)) if mask.size else 0.0


def _pct(values: np.ndarray, q: float) -> float:
    """Percentile, NaN for an empty array."""
    return float(np.percentile(values, q)) if values.size else float("nan")


def aggregate(
        diag: SQPSolveDiagnostics,
        *,
        env: str,
        horizon: int,
        seed: int,
        train_step: int,
        eta_s: float,
        lambda_xi: float,
        alpha_decay: float,
        n_sqp_iter: int,
        ckpt: str = "",
        teacher: str = "true",
        excluded_mask: Optional[np.ndarray] = None,
        excluded_nonconverged: Optional[np.ndarray] = None,
        excluded_violating: Optional[np.ndarray] = None,
) -> dict:
    """Reduce a batch of per-solve diagnostics to one CSV row."""
    d = {k: np.asarray(v) for k, v in diag._asdict().items()}
    n = int(d["n_iters"].shape[0])

    xi_max = d["xi_max"].astype(np.float64)
    xi_nonzero = xi_max > SQP_SLACK_TOL
    xi_nz_vals = xi_max[xi_nonzero]

    xi_qp_max = d["xi_qp_max"].astype(np.float64)
    xi_qp_nonzero = xi_qp_max > SQP_SLACK_TOL
    xi_qp_nz_vals = xi_qp_max[xi_qp_nonzero]

    budget_under_thresh = d["budget_hit"] & d["converged_thresh"] & ~d["converged_strict"]
    budget_under_thresh_safe = budget_under_thresh & ~d["violates_untightened"]

    exit_counts = {
        name: int(np.sum(d["exit_code"] == code))
        for code, name in sorted(EXIT_NAMES.items())
    }

    row = {
        "env": env,
        "ckpt": ckpt,
        "teacher": teacher,
        "H": horizon,
        "seed": seed,
        "train_step": train_step,
        "n_solves": n,
        "converged_frac": _frac(d["converged_strict"]),
        "converged_strict_frac": _frac(d["converged_strict"]),
        "converged_thresh_frac": _frac(d["converged_thresh"]),
        "mean_iters": float(np.mean(d["n_iters"])) if n else float("nan"),
        "max_iters_used": int(np.max(d["n_iters"])) if n else 0,
        "budget_hit_frac": _frac(d["budget_hit"]),
        "budget_hit_under_thresh_frac": _frac(budget_under_thresh),
        "budget_hit_under_thresh_safe_frac": _frac(budget_under_thresh_safe),
        "xi_nonzero_frac": _frac(xi_nonzero),
        "xi_mean_given_nonzero": float(np.mean(xi_nz_vals)) if xi_nz_vals.size else 0.0,
        "xi_p50": _pct(xi_max, 50),
        "xi_p95": _pct(xi_max, 95),
        "xi_p50_given_nonzero": _pct(xi_nz_vals, 50),
        "xi_p95_given_nonzero": _pct(xi_nz_vals, 95),
        "xi_max": float(np.max(xi_max)) if n else 0.0,
        "xi_mean_sum": float(np.mean(d["xi_sum"])) if n else 0.0,
        "xi_nonzero_count_mean": float(np.mean(d["xi_nonzero_count"])) if n else 0.0,
        "xi_exceeds_eta_s_frac": _frac(xi_max > eta_s),
        "xi_below_eta_s_frac": _frac(d["xi_below_eta_s"]),
        "violates_untightened_frac": _frac(d["violates_untightened"]),
        "untightened_margin_p95": _pct(d["untightened_margin_max"].astype(np.float64), 95),
        "xi_qp_nonzero_frac": _frac(xi_qp_nonzero),
        "xi_qp_mean_given_nonzero": float(np.mean(xi_qp_nz_vals)) if xi_qp_nz_vals.size else 0.0,
        "xi_qp_p50": _pct(xi_qp_max, 50),
        "xi_qp_p95": _pct(xi_qp_max, 95),
        "xi_qp_p95_given_nonzero": _pct(xi_qp_nz_vals, 95),
        "xi_qp_max": float(np.max(xi_qp_max)) if n else 0.0,
        "xi_qp_exceeds_eta_s_frac": _frac(d["xi_qp_exceeds_eta_s"]),
        "nonfinite_frac": _frac(d["has_nonfinite"]),
        "qp_degraded_frac": _frac(d["qp_degraded"]),
        "qp_pri_res_p95": _pct(d["qp_pri_res"].astype(np.float64), 95),
        "qp_dua_res_p95": _pct(d["qp_dua_res"].astype(np.float64), 95),
        "excluded_from_lact_frac": _frac(excluded_mask) if excluded_mask is not None else 0.0,
        "excluded_nonconverged_frac": (
            _frac(excluded_nonconverged) if excluded_nonconverged is not None else 0.0
        ),
        "excluded_violating_frac": (
            _frac(excluded_violating) if excluded_violating is not None else 0.0
        ),
        "eta_s": eta_s,
        "lambda_xi": lambda_xi,
        "alpha_decay": alpha_decay,
        "n_sqp_iter": n_sqp_iter,
        "exit_status_counts": ";".join(f"{k}={v}" for k, v in exit_counts.items()),
    }
    return row


CSV_FIELDS: Sequence[str] = (
    "env", "ckpt", "teacher", "H", "seed", "train_step", "n_solves",
    "converged_frac", "converged_strict_frac", "converged_thresh_frac",
    "mean_iters", "max_iters_used",
    "budget_hit_frac", "budget_hit_under_thresh_frac", "budget_hit_under_thresh_safe_frac",
    "xi_nonzero_frac", "xi_mean_given_nonzero",
    "xi_p50", "xi_p95", "xi_p50_given_nonzero", "xi_p95_given_nonzero", "xi_max",
    "xi_mean_sum", "xi_nonzero_count_mean",
    "xi_exceeds_eta_s_frac", "xi_below_eta_s_frac",
    "violates_untightened_frac", "untightened_margin_p95",
    "xi_qp_nonzero_frac", "xi_qp_mean_given_nonzero",
    "xi_qp_p50", "xi_qp_p95", "xi_qp_p95_given_nonzero", "xi_qp_max",
    "xi_qp_exceeds_eta_s_frac",
    "nonfinite_frac", "qp_degraded_frac", "qp_pri_res_p95", "qp_dua_res_p95",
    "excluded_from_lact_frac", "excluded_nonconverged_frac", "excluded_violating_frac",
    "eta_s", "lambda_xi", "alpha_decay", "n_sqp_iter",
    "exit_status_counts",
)


def write_csv(rows: Sequence[dict], path: str) -> None:
    """Write (or append to) the stats CSV."""
    if not rows:
        return
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    exists = os.path.isfile(path) and os.path.getsize(path) > 0

    fields = list(CSV_FIELDS)
    if exists:
        with open(path, newline="") as f:
            existing = next(csv.reader(f), None)
        if existing and existing != fields:
            missing = [c for c in fields if c not in existing]
            fields = existing
            if missing:
                import warnings
                warnings.warn(
                    f"{path} was written with an older diagnostics schema "
                    f"({len(existing)} columns). Appending under that header to "
                    f"keep columns aligned; {missing!r} will NOT be recorded in "
                    f"this file. Use a different --out for the current schema.",
                    stacklevel=2,
                )

    with open(path, "a" if exists else "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        for r in rows:
            writer.writerow(r)


def format_summary(row: dict) -> str:
    """Human-readable per-environment summary block."""
    eta_s = row["eta_s"]
    over_eta = (
        f"{row['xi_exceeds_eta_s_frac'] * 100:5.1f}%" if eta_s > 0 else "  n/a  (eta_s=0)"
    )
    lines = [
        f"SQP teacher diagnostics (env={row['env']}, H={row['H']}, "
        f"seed={row['seed']}, step={row['train_step']}, n={row['n_solves']})"
        + (f"\n  checkpoint: {row['ckpt']}" if row.get("ckpt") else "")
        + f"\n  transition model: {row.get('teacher', 'true')}"
        + ("   (dynamics head; u_nom and obs reconstruction remain analytic)"
           if row.get("teacher") == "learned" else "   (analytic step_simulator)"),
        f"  convergence rate:              {row['converged_strict_frac'] * 100:5.1f}%   "
        f"(mean {row['mean_iters']:.1f} iters, budget={row['n_sqp_iter']})",
        f"    at {SQP_CONV_THRESH} tolerance:        {row['converged_thresh_frac'] * 100:5.1f}%",
        f"  nonzero slack:                 {row['xi_nonzero_frac'] * 100:5.1f}% of solves",
        f"  slack magnitude (p50 / p95):   {row['xi_p50_given_nonzero']:.3e} / "
        f"{row['xi_p95_given_nonzero']:.3e}   (over nonzero solves)",
        f"  slack exceeding eta_s:         {over_eta} of solves",
        f"  slack nonzero but < eta_s:     {row['xi_below_eta_s_frac'] * 100:5.1f}% of solves",
        f"  violates untightened DTCBF:    {row['violates_untightened_frac'] * 100:5.1f}% of solves",
        f"  excluded from L_act:           {row['excluded_from_lact_frac'] * 100:5.1f}% of samples"
        f"   (under the ACTIVE rule set)",
        f"    would exclude, non-converged:{row['excluded_nonconverged_frac'] * 100:5.1f}%"
        f"   (rule off unless --filter-nonconverged)",
        f"    would exclude, violating:    {row['excluded_violating_frac'] * 100:5.1f}%"
        f"   (rule off unless --filter-violating)",
        f"  QP slack (Eq.15) nonzero:      {row['xi_qp_nonzero_frac'] * 100:5.1f}% of solves, "
        f"p95(nonzero)={row['xi_qp_p95_given_nonzero']:.3e}",
        f"  budget hit:                    {row['budget_hit_frac'] * 100:5.1f}%  "
        f"(of which {row['budget_hit_under_thresh_frac'] * 100:.1f}% land in "
        f"(0, {SQP_CONV_THRESH}] and are still used as labels)",
        f"  non-finite solves:             {row['nonfinite_frac'] * 100:5.1f}%",
        f"  degraded final QP:             {row['qp_degraded_frac'] * 100:5.1f}%",
        f"  exit status:                   {row['exit_status_counts']}",
    ]
    return "\n".join(lines)
