# BarrierFormer

**Project website:** https://safe-robotics-group-at-asu.github.io/BarrierFormer-website/

A causal-transformer policy for safe navigation. A learned control barrier function (CBF) is enforced over an H-step predicted rollout, and the policy is trained in Phase 2 with a CBF/SQP teacher. Three environments: DoubleIntegrator (DI), DubinsCar (DC) and CrazyFlie (CF).

```
BarrierFormer/
├── barrierformer/     one shared package
│   ├── nn/            causal transformer, action head, dynamics head, CBF network
│   ├── algo/          Phase 2 training algorithm (CBF losses, SQP teacher, receding-horizon policy)
│   ├── env/           double_integrator.py, dubins_car.py, crazyflie.py (+ LiDAR / obstacle utils)
│   ├── trainer/       training loop, replay buffers
│   └── utils/
├── train_di.py        Phase 2 training: DoubleIntegrator
├── train_dc.py        Phase 2 training: DubinsCar
├── train_cf.py        Phase 2 training: CrazyFlie
├── test.py            evaluation: normal / generalization / robustness (all envs)
├── checkpoints/{DI,DC,CF}/{dynhead,simroll}/   final models: config.yaml + models/1000/{actor,cbf,cbf_tgt}.pkl
├── requirements.txt   Python dependencies
├── README.md          this file (layout + training)
└── test.md            all test commands + expected results
```

## Final models

All models use K = 12 (history), H = 6 (prediction horizon) and checkpoint 1000. There are two per environment:
- `dynhead`: the CBF rollout is predicted by the learned dynamics head (**the method**).
- `simroll`: the CBF rollout comes from the simulator (ablation).

Within each environment the two were trained with identical settings except `--simulator-rollout`.

## Requirements

Python 3.10. Install the dependencies with `pip install -r requirements.txt` (install JAX with CUDA support separately for GPU training). Run all scripts from inside `BarrierFormer/`.

## Testing

See `test.md`. In short:

```bash
python test.py --env DI --ckpt checkpoints/DI/dynhead                        # normal test (seeds 0 1 2)
python test.py --env DI --ckpt checkpoints/DI/dynhead --mode generalization
```

## Phase 2 training (reproduce the final models)

GPU recommended. Add `--simulator-rollout=True` to train the simulator-rollout ablation.

```bash
COMMON="--n-rays 32 --max-step 256 --history-len 12 --hidden-dim 128 --num-heads 2 --num-layers 1 --mlp-ratio 4.0 \
 --action-head-hidden-dim 128 --action-head-num-layers 4 --dynamics-head-hidden-dim 128 --dynamics-head-num-layers 3 \
 --horizon 6 --n-sqp-iter 100 --relax-penalty 1e4 --alpha 0.3 --gamma 0.02 --beta 20.0 --cbf-tgt-tau 0.5 --labeling-horizon 32 \
 --loss-action-coef 0.5 --loss-dt-cbf-coef 0.5 --loss-dyn-coef 1.0 --loss-safe-coef 1.0 --loss-unsafe-coef 1.5 \
 --lr-actor 3e-5 --lr-cbf 3e-6 --lr-schedule --max-grad-norm 1.0 --batch-size 256 --buffer-size 1024 --inner-epoch 8 \
 --n-env-train 16 --n-env-test 32 --eval-interval 1 --save-interval 10 --seed 0"

# DoubleIntegrator
python train_di.py --area-size 4.0 --n-obs 8 --lidar-ray-order --pretrain-ckpt pretrained/DI/params_final.pkl $COMMON --steps 1000 --name di_k12_h6_dynhead

# DubinsCar
python train_dc.py --area-size 4.0 --num-agents 1 --n-obs 8 --use-orig-u-ref --pid-k-v 2.7 --lidar-ray-order --state-obs vxy --pretrain-ckpt pretrained/DC/params_final.pkl $COMMON --steps 1000 --name dc_k12_h6_dynhead

# CrazyFlie
python train_cf.py --area-size 3.0 --num-agents 1 --n-obs 6 --lidar-encoding sector --pretrain-ckpt pretrained/CF/params_final.pkl $COMMON --steps 1000 --name cf_k12_h6_dynhead
```

**Outputs:** `logs/<Env>/gcbf_transformer/seed0_<timestamp>/`, holding `config.yaml` and `models/<step>/`. Point `test.py --ckpt` at that folder.

## Acknowledgements

This codebase is built on [GCBF+](https://github.com/MIT-REALM/gcbfplus) (Zhang, So, Garg and Fan, "GCBF+: A Neural Graph Control Barrier Function Framework for Distributed Safe Multi-Agent Control", IEEE T-RO 2025). We thank the GCBF+ authors for making their code publicly available.
