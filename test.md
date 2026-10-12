# Testing BarrierFormer

All commands are run from inside `BarrierFormer/`. `test.py` is the only test script.

**Defaults (no need to type them):** checkpoint step 1000, evaluation seeds 0 1 2 with 32 episodes each (96 episodes), and CPU. CPU results are deterministic. The environment settings are read from each checkpoint's `config.yaml`.

```bash
cd BarrierFormer
PY=python      # the conda env with jax/flax/optax, here: /home/anandsc/miniconda3/envs/GCBFMHA/bin/python
```

## 1. Normal test (training setting, 256-step episodes)

```bash
$PY test.py --env DI --ckpt checkpoints/DI/dynhead
$PY test.py --env DI --ckpt checkpoints/DI/simroll
$PY test.py --env DC --ckpt checkpoints/DC/dynhead
$PY test.py --env DC --ckpt checkpoints/DC/simroll
$PY test.py --env CF --ckpt checkpoints/CF/dynhead
$PY test.py --env CF --ckpt checkpoints/CF/simroll
```

Test settings:
- **DI and DC:** 4 m × 4 m, 8 obstacles.
- **CF:** 3 m cube, 6 obstacles of size 0.2–0.7.

## 2. Generalization (larger workspace, denser obstacles, 1024-step episodes)

```bash
$PY test.py --env DI --ckpt checkpoints/DI/dynhead --mode generalization
$PY test.py --env DI --ckpt checkpoints/DI/simroll --mode generalization
$PY test.py --env DC --ckpt checkpoints/DC/dynhead --mode generalization
$PY test.py --env DC --ckpt checkpoints/DC/simroll --mode generalization
$PY test.py --env CF --ckpt checkpoints/CF/dynhead --mode generalization
$PY test.py --env CF --ckpt checkpoints/CF/simroll --mode generalization
```

Density is ρ = M / l^n_p, where M is the number of obstacles, l the side length and n_p ∈ {2, 3} the dimension of the position space. Each episode is generated at ρ = 1.00, and the lower densities remove obstacles while keeping all other obstacle poses, so the comparison across densities is paired.

| Env | Side length l | M at ρ = 1.00 / 0.75 / 0.50 | Training setting |
|---|---|---|---|
| DI, DC | 6 m | 36 / 27 / 18 | l = 4, M = 8 (ρ = 0.50) |
| DI, DC | 4 m | 16 / 12 / 8 | |
| CF | 3 m | 27 / 20 / 13 | l = 3, M = 6 (ρ ≈ 0.22) |
| CF | 4 m | 64 / 48 / 32 | |

CF uses obstacle size 0.1–0.6.

## 3. Robustness to dynamics perturbations (DI)

```bash
$PY test.py --env DI --ckpt checkpoints/DI/dynhead --mode robustness
$PY test.py --env DI --ckpt checkpoints/DI/simroll --mode robustness
```

The policy is not retrained. Only the true simulator step is perturbed:
- **Velocity mismatch:** ṗ = α_vel · v, with α_vel ∈ {0.8, 0.9, 1.0, 1.1, 1.2}.
- **Process noise:** w_t ~ N(0, diag(σ_p², σ_p², σ_v², σ_v²)) added at every 0.03 s step, with σ_p = 4σ and σ_v = 0.5σ, and σ ∈ {0, 0.001, 0.0025, 0.005, 0.0075, 0.01}.

`--noise-scale absolute` uses σ_p = σ_v = σ instead.

## Options

| Option | Meaning |
|---|---|
| `--step N` | another checkpoint step (default 1000) |
| `--seeds ...` | other evaluation seeds (default 0 1 2) |
| `--epi N` | episodes per seed (default 32) |
| `--max-step N` | episode length (default 256 normal / robustness, 1024 generalization) |
| `--obs-len-range MIN MAX` | obstacle size range |
| `--triples "l,M1,M2,M3 ..."` | custom generalization cells |
| `--show-failures` | list failed episodes as `s<seed>e<episode>:C` (collision) or `:T` (timeout) |
| `--csv FILE` | also save the results table as CSV |
| `--gpu` | run on GPU (results may differ by about an episode because of GPU non-determinism) |

**Metrics:**
- **Safe:** the agent never collides; a collision is the agent's 0.05 m disk/sphere overlapping an obstacle.
- **Reach:** the agent comes within 0.10 m of the goal at some step.
- **Success:** both.

Each is reported as **mean ± sample std (ddof=1) over the evaluation seeds**, with the pooled Wilson 95% CI in brackets. Runtime on CPU: DI about 1–3 min per run, DC a few minutes, CF longer (its 3D LiDAR casts 514 rays per step).
