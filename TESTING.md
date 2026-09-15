# Testing / Evaluation Commands

Reproduce the evaluation results for every environment using the trained
checkpoints in [`Trained-Weights/`](Trained-Weights/). All commands use
`test_transformer_eval.py`, run 3 seeds × 32 parallel episodes, and skip video
rendering.

Each command prints per-seed and aggregate **safe / finish / success** rates
(aggregate ± is std across seeds). Add `--log` to also append the results to a
`test_log.csv` inside the checkpoint's run directory.

| Common flag        | Meaning                                            |
|--------------------|----------------------------------------------------|
| `--seeds 0 1 2`    | Evaluate over 3 random seeds, report aggregate     |
| `--n-env 32`       | 32 parallel episodes per seed                      |
| `--parallel`       | Batch all episodes with `jax.vmap`                 |
| `--max-step 256`   | Episode length                                     |
| `--no-video`       | Skip `.mp4` / h-trajectory rendering               |
| `--log`            | Append results to `<path>/test_log.csv`            |

---

## Double Integrator (DI)

**Sim**
```bash
python test_transformer_eval.py --path ./Trained-Weights/DI/DI-Sim/ \
    --seeds 0 1 2 --n-env 32 --parallel --max-step 256 --no-video \
    --env-id DoubleIntegrator --log --area 4 --n-obs 8
```

**Dyn**
```bash
python test_transformer_eval.py --path ./Trained-Weights/DI/DI-Dyn/ \
    --seeds 0 1 2 --n-env 32 --parallel --max-step 256 --no-video \
    --env-id DoubleIntegrator --area 4 --n-obs 8
```
Table 2:
SIM:
python test_transformer_progressive_remove_3step_all.py     --env-id DoubleIntegrator --path ./Trained-Weights/DI/DI-Sim/     --seeds 0 1 2 --n-env 32 --parallel 

DYN:
python test_transformer_progressive_remove_3step_all.py     --env-id DoubleIntegrator --path ./Trained-Weights/DI/DI-Dyn/     --seeds 0 1 2 --n-env 32 --parallel 


## Dubins Car (DC)

**Sim**
```bash
python test_transformer_eval.py --path ./Trained-Weights/DC/Dubins-Sim/ \
    --seeds 0 1 2 --n-env 32 --parallel --max-step 256 --no-video \
    --env-id DubinsCar --area 4 --n-obs 8
```

**Dyn**
```bash
python test_transformer_eval.py --path ./Trained-Weights/DC/Dubins-DYN/ \
    --seeds 0 1 2 --n-env 32 --parallel --max-step 256 --no-video \
    --env-id DubinsCar --log --area 4 --n-obs 8
```
Table 2:
SIM:
python test_transformer_progressive_remove_3step_all.py     --env-id DubinsCar --path ./Trained-Weights/DC/Dubins-Sim/     --seeds 0 1 2 --n-env 32 --parallel 

DYN: 
python test_transformer_progressive_remove_3step_all.py     --env-id DubinsCar --path ./Trained-Weights/DC/Dubins-DYN/     --seeds 0 1 2 --n-env 32 --parallel 


## CrazyFlie (CF)

CrazyFlie uses environment-specific defaults (`area=3`, `n_obs=27`, `n_rays=32`,
`obs_len_range=[0.1, 0.6]`), so `--area` / `--n-obs` are not needed.

**Sim**
```bash
python test_transformer_eval.py --path ./Trained-Weights/CF/CF-Sim/ \
    --seeds 0 1 2 --n-env 32 --parallel --max-step 256 --no-video \
    --env-id CrazyFlie
```

**Dyn**
```bash
python test_transformer_eval.py --path ./Trained-Weights/CF/CF-Dyn/ \
    --seeds 0 1 2 --n-env 32 --parallel --max-step 256 --no-video \
    --env-id CrazyFlie
```
Table 2:
SIM:
python test_transformer_sweep_dense_to_sparse_crazyflie.py --path ./Trained-Weights/CF/CF-Sim/  --seeds 0 1 2


DYN: python test_transformer_sweep_dense_to_sparse_crazyflie.py --path ./Trained-Weights/CF/CF-Dyn/  --seeds 0 1 2

