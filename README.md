# BarrierFormer

BarrierFormer is a Transformer-based Control Barrier Function (CBF) framework for safe single-agent control. It is built on top of the environments and training infrastructure from [GCBF+](https://mit-realm.github.io/gcbfplus-website/).

## Dependencies

We recommend using [CONDA](https://www.anaconda.com/) to manage the environment:

```bash
conda create -n barrierformer python=3.10
conda activate barrierformer
```

Install JAX following the [official instructions](https://github.com/google/jax#installation), then install the remaining dependencies:

```bash
pip install -r requirements.txt
```

## Installation

```bash
pip install -e .
```

## Environments

Two 2D environments: `DoubleIntegrator`, `DubinsCar`  
One 3D environments:`CrazyFlie`


**Sim**: same command with simulator rollout substitution (see above).

## Evaluation

Reproduce all evaluation results using the pretrained checkpoints in `Trained-Weights/`. All commands run 3 seeds × 32 parallel episodes and skip video rendering.

| Flag | Meaning |
|---|---|
| `--seeds 0 1 2` | Evaluate over 3 random seeds, report aggregate |
| `--n-env 32` | 32 parallel episodes per seed |
| `--parallel` | Batch all episodes with `jax.vmap` |
| `--max-step 256` | Episode length |
| `--no-video` | Skip `.mp4` / h-trajectory rendering |
| `--log` | Append results to `<path>/test_log.csv` |

### Double Integrator (DI)

**Table 1 — Sim**
```bash
python test_transformer_eval.py --path ./Trained-Weights/DI/DI-Sim/ \
    --seeds 0 1 2 --n-env 32 --parallel --max-step 256 --no-video \
    --env-id DoubleIntegrator --log --area 4 --n-obs 8
```

**Table 1 — Dyn**
```bash
python test_transformer_eval.py --path ./Trained-Weights/DI/DI-Dyn/ \
    --seeds 0 1 2 --n-env 32 --parallel --max-step 256 --no-video \
    --env-id DoubleIntegrator --area 4 --n-obs 8
```

**Table 2 — Sim**
```bash
python test_transformer_progressive_remove_3step_all.py \
    --env-id DoubleIntegrator --path ./Trained-Weights/DI/DI-Sim/ \
    --seeds 0 1 2 --n-env 32 --parallel
```

**Table 2 — Dyn**
```bash
python test_transformer_progressive_remove_3step_all.py \
    --env-id DoubleIntegrator --path ./Trained-Weights/DI/DI-Dyn/ \
    --seeds 0 1 2 --n-env 32 --parallel
```

---

### Dubins Car (DC)

**Table 1 — Sim**
```bash
python test_transformer_eval.py --path ./Trained-Weights/DC/Dubins-Sim/ \
    --seeds 0 1 2 --n-env 32 --parallel --max-step 256 --no-video \
    --env-id DubinsCar --area 4 --n-obs 8
```

**Table 1 — Dyn**
```bash
python test_transformer_eval.py --path ./Trained-Weights/DC/Dubins-DYN/ \
    --seeds 0 1 2 --n-env 32 --parallel --max-step 256 --no-video \
    --env-id DubinsCar --log --area 4 --n-obs 8
```

**Table 2 — Sim**
```bash
python test_transformer_progressive_remove_3step_all.py \
    --env-id DubinsCar --path ./Trained-Weights/DC/Dubins-Sim/ \
    --seeds 0 1 2 --n-env 32 --parallel
```

**Table 2 — Dyn**
```bash
python test_transformer_progressive_remove_3step_all.py \
    --env-id DubinsCar --path ./Trained-Weights/DC/Dubins-DYN/ \
    --seeds 0 1 2 --n-env 32 --parallel
```

---

### CrazyFlie (CF)

`--area-size` and `--n-obs` are not needed: they are read from the checkpoint's `config.yaml`, which for the released CrazyFlie weights is `area_size=3.0`, `n_obs=6`. The per-environment entry in `test_transformer_eval.py` is only a fallback for checkpoints that do not record them.

**Table 1 — Sim**
```bash
python test_transformer_eval.py --path ./Trained-Weights/CF/CF-Sim/ \
    --seeds 0 1 2 --n-env 32 --parallel --max-step 256 --no-video \
    --env-id CrazyFlie  --obs-len-range 0.2 0.7
```

**Table 1 — Dyn**
```bash
python test_transformer_eval.py --path ./Trained-Weights/CF/CF-Dyn/ \
    --seeds 0 1 2 --n-env 32 --parallel --max-step 256 --no-video \
    --env-id CrazyFlie  --obs-len-range 0.2 0.7
```

**Table 2 — Sim**
```bash
python test_transformer_sweep_dense_to_sparse_crazyflie.py \
    --path ./Trained-Weights/CF/CF-Sim/ --seeds 0 1 2
```

**Table 2 — Dyn**
```bash
python test_transformer_sweep_dense_to_sparse_crazyflie.py \
    --path ./Trained-Weights/CF/CF-Dyn/ --seeds 0 1 2
```

---


## Training

BarrierFormer uses a **dynamics head** — a learned MLP that predicts the next-state residual `Δx` from the transformer's latent and the applied action. During training, the DTCBF horizon rollout uses the dynamics head's predicted states. For the **Sim** variant, this rollout is replaced by the known ground-truth simulator (`step_simulator` in `barrierformer/algo/barrierformer.py`), which is JAX-differentiable via Euler integration.

Transformer pretraining (SDF-based, required before training all environments):

```bash
python pretrain_transformer.py
```

### Double Integrator

```bash
python train_transformer.py \
    --area-size 4.0 --steps 5000 --n-env-train 16 \
    --buffer-size 1024 --n-sqp-iter 35 --relax-penalty 5000 \
    --alpha 0.3 --beta 20.0 --gamma 0.1 \
    --lr-actor 3e-5 --lr-cbf 1e-5 \
    --loss-dt-cbf-coef 0.5 --loss-dyn-coef 1.0 \
    --lr-schedule \
    --pretrain-ckpt Pretrained-ckpts/DI/params_best.pkl \
    --name DI-Dyn
```

**Sim** (replace dynamics head rollout with simulator): same command, but in `barrierformer/algo/barrierformer.py` replace the dynamics head state propagation with `self._env.step_simulator`.

### Dubins Car

```bash
python train_transformer_dubins.py \
    --area-size 4.0 --steps 5000 --n-env-train 16 \
    --buffer-size 1024 --n-sqp-iter 50 --relax-penalty 5000 \
    --alpha 0.3 --beta 20.0 --gamma 0.1 \
    --lr-actor 3e-5 --lr-cbf 1e-5 --max-grad-norm 1.0 \
    --loss-dt-cbf-coef 0.5 --loss-dyn-coef 2.0 \
    --lr-schedule \
    --pretrain-ckpt Pretrained-ckpts/DC/params_best.pkl \
    --name DC-Dyn
```

**Sim**: same command with simulator rollout substitution (see above).

### CrazyFlie

CrazyFlie observations are normalized, so the pretrained dynamics head predicts the **normalized** state increment `Δx / scale` rather than raw `Δx`. CrazyFlie training therefore uses `barrierformer/algo/gcbf_plus_cf.py` instead of the default `gcbf_plus.py`. `gcbf_plus_cf.py` loads a per-dimension scale vector from a `.npz` file (passed via `dyn_scale_path`) and de-normalizes the dynamics head output before advancing state: `x_{k+1} = x_k + delta_x_hat * scale`. For DI and DubinsCar the scale is all-ones (no-op), so the standard `gcbf_plus.py` is used for those environments.

```bash
python train_transformer_crazyflie.py \
    --area-size 3.0 --steps 5000 --n-env-train 16 --n-obs 6 \
    --buffer-size 512 --n-sqp-iter 25 --relax-penalty 1000 \
    --alpha 0.3 --beta 10.0 --gamma 0.0 \
    --lr-actor 3e-5 --lr-cbf 3e-5 --max-grad-norm 2.0 \
    --loss-dt-cbf-coef 0.2 --loss-dyn-coef 1.0 \
    --pretrain-ckpt Pretrained-ckpts/CF/params_best.pkl \
    --name CF-Dyn
```

## Citation

This work builds on GCBF+ (Public Repository), and adding appropriate citations:

```bibtex
@ARTICLE{zhang2025gcbf+,
      author={Zhang, Songyuan and So, Oswin and Garg, Kunal and Fan, Chuchu},
      journal={IEEE Transactions on Robotics},
      title={{GCBF}+: A Neural Graph Control Barrier Function Framework for Distributed Safe Multiagent Control},
      year={2025},
      volume={41},
      pages={1533-1552},
      doi={10.1109/TRO.2025.3530348}
}
```
