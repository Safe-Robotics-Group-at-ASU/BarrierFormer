"""
nn package — causal transformer policy for CBF training.

Public API
----------
CausalTransformer        : GPT-2-style causal transformer backbone (transformer.py).
TransformerConfig        : Frozen dataclass config for the backbone.
ActionHead               : Residual action-correction MLP (action_head.py).
DynamicsHead             : State-increment prediction MLP (dynamics_head.py).
CausalTransformerPolicy  : Top-level module wiring all sub-modules (model.py).
TransformerPolicyConfig  : Aggregated frozen config for the full policy.
ModelOutput              : Single-step forward-pass output container.
RolloutOutput            : H-step autoregressive rollout output container.
CBFMLP                   : MLP Control Barrier Function; output in (-1, 1) (cbf.py).
"""

from .transformer import (
    CausalTransformer,
    TransformerConfig,
    CausalTransformerBlock,
    MultiHeadSelfAttention,
)

from .action_head import ActionHead

from .dynamics_head import DynamicsHead

from .model import (
    CausalTransformerPolicy,
    TransformerPolicyConfig,
    ModelOutput,
    RolloutOutput,
)

from .cbf import CBFMLP

__all__ = [
    "CausalTransformer",
    "TransformerConfig",
    "CausalTransformerBlock",
    "MultiHeadSelfAttention",
    "ActionHead",
    "DynamicsHead",
    "CausalTransformerPolicy",
    "TransformerPolicyConfig",
    "ModelOutput",
    "RolloutOutput",
    "CBFMLP",
]
