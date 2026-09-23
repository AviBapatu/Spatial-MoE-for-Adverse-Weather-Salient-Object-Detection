"""Configuration dataclasses for the Spatial-MoE SOD pipeline.

Provides a hierarchical ``ExperimentConfig`` built from sub-configs
(``DataConfig``, ``ModelConfig``, ``LossConfig``, ``OptimizationConfig``,
``TrainingConfig``, ``EvaluationConfig``, ``DiagnosticsConfig``) with
serialization, canonical hashing, validation, and preset loading.
"""
import dataclasses
import hashlib
import json
import os
import warnings
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Type

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _filter_dataclass_kwargs(dc_cls: Type, raw: Dict[str, Any]) -> Dict[str, Any]:
    """Return only the keys from *raw* that are valid fields of *dc_cls*.

    Unknown keys are silently dropped so that forward-compatible configs
    (with fields that don't exist in an older dataclass) load without error.
    """
    valid = {f.name for f in dataclasses.fields(dc_cls)}
    return {k: v for k, v in raw.items() if k in valid}


# ---------------------------------------------------------------------------
# Sub-configs
# ---------------------------------------------------------------------------

@dataclass
class DataConfig:
    """Dataset paths and sample counts."""

    dataset_root: str = "/kaggle/input/wxsod-dataset"
    split_manifest_path: str = ""
    split_manifest_hash: str = ""
    train_sample_count: int = 0
    validation_sample_count: int = 0
    test_sample_counts: Dict[str, int] = field(default_factory=dict)
    max_samples: Optional[int] = None


@dataclass
class ModelConfig:
    """Spatial-MoE architecture hyperparameters."""

    backbone: str = "pvt_v2_b4"
    working_dim: int = 256
    decoder_dim: int = 256
    num_experts: int = 8
    top_k: int = 2
    router_variant: str = "token_only"
    expert_variant: str = "standard"
    attention_config: str = "default"
    window_size: int = 7
    deep_supervision: bool = False
    moe_type: str = "sparse"  # "none", "dense", "sparse"
    moe_16_mode: str = "sparse"  # "sparse" | "dense"
    router_noise_enabled: bool = True
    router_noise_scale: float = 1.0
    router_noise_min_std: float = 0.05
    # How the per-token gate weights are formed from the top-k router logits.
    #   "renormalized" — softmax over the k selected logits (historical default).
    #       Shift-invariant within the selected pair, so the task loss has zero
    #       gradient for every expert the token did not pick and selection can
    #       only drift under noise.
    #   "dense" — the experts' probabilities from the full softmax over all E
    #       experts, gathered at the top-k indices (Switch/Shazeer style).  The
    #       loss then has a gradient for non-selected experts, so the router can
    #       actually learn which experts to prefer.
    gate_mode: str = "renormalized"


@dataclass
class LossConfig:
    """Loss component weights."""

    bce_weight: float = 1.0
    iou_weight: float = 1.0
    ssim_weight: float = 0.0
    boundary_weight: float = 0.0
    load_balance_weight: float = 0.01
    importance_weight: float = 0.01
    z_loss_weight: float = 0.0
    # Weight on the mean normalised routing entropy (minimising it makes routing
    # more confident).  0.0 disables the term, i.e. the historical behaviour.
    entropy_confidence_weight: float = 0.0
    aux_boundary_weight: float = 0.0
    deep_supervision_weight: float = 0.4
    # Per-stage load-balance weights: [w_stride4, w_stride8, w_stride16].
    # None = fall back to uniform average weighted by load_balance_weight
    # (identical to behaviour before this field was added).
    # When set, CombinedLoss uses a weighted sum of per-stage l_lb values
    # instead of a pre-averaged scalar, allowing collapsed stages to be
    # penalised independently.  load_balance_weight is ignored when this
    # field is not None.
    load_balance_weights: Optional[List[float]] = None
    moe_16_dense: bool = False


@dataclass
class OptimizationConfig:
    """Optimizer, scheduler, and batch-size settings."""

    optimizer: str = "AdamW"
    backbone_lr: float = 1e-4
    new_module_lr: float = 1e-4
    weight_decay: float = 1e-4
    warmup_ratio: float = 0.01
    scheduler: str = "WarmupCosine"
    gradient_clipping: float = 1.0
    amp: bool = True
    batch_per_gpu: int = 1
    grad_accum_steps: int = 16
    effective_global_batch: int = 32


@dataclass
class TrainingConfig:
    """Training loop settings."""

    epochs: int = 50
    freeze_backbone_epochs: int = 1
    seed: int = 42
    num_workers: int = 2
    checkpoint_every_n_steps: int = 200


@dataclass
class EvaluationConfig:
    """Evaluation and metric selection settings."""

    validation_metric_selection: str = "MAE"
    tta_mode: str = "none"
    metric_config: str = "default"


@dataclass
class DiagnosticsConfig:
    """Routing diagnostics and expert ablation settings."""

    routing_diagnostic_epochs: int = 1
    expert_ablation_enabled: bool = False


# ---------------------------------------------------------------------------
# Preset registry
# ---------------------------------------------------------------------------

_PRESETS_DIR = os.path.join(os.path.dirname(__file__), "presets")
_PRESET_CACHE: Dict[str, Dict[str, Any]] = {}


def _load_preset_registry() -> Dict[str, Dict[str, Any]]:
    """Discover and cache all ``*.json`` files under ``src/presets/``."""
    if _PRESET_CACHE:
        return _PRESET_CACHE
    if not os.path.isdir(_PRESETS_DIR):
        return _PRESET_CACHE
    for fname in sorted(os.listdir(_PRESETS_DIR)):
        if fname.endswith(".json"):
            name = fname[: -len(".json")]
            path = os.path.join(_PRESETS_DIR, fname)
            with open(path, "r") as fh:
                _PRESET_CACHE[name] = json.load(fh)
    return _PRESET_CACHE


def list_presets() -> List[str]:
    """Return the names of all available presets."""
    return sorted(_load_preset_registry().keys())


# ---------------------------------------------------------------------------
# ExperimentConfig
# ---------------------------------------------------------------------------

@dataclass
class ExperimentConfig:
    """Top-level configuration tree for a Spatial-MoE SOD experiment.

    Attributes:
        data: Dataset paths and metadata.
        model: Architecture hyperparameters.
        loss: Loss component weights.
        opt: Optimizer, scheduler, and batch-size settings.
        train: Training loop settings.
        eval: Evaluation and metric selection settings.
        diag: Routing diagnostics settings.
        experiment_id: Auto-generated short experiment identifier (volatile).
        run_id: Auto-generated run identifier with timestamp (volatile).
        batch_equivalence: ``"MATCHED"`` or ``"NON_MATCHED"`` (volatile).
    """

    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    opt: OptimizationConfig = field(default_factory=OptimizationConfig)
    train: TrainingConfig = field(default_factory=TrainingConfig)
    eval: EvaluationConfig = field(default_factory=EvaluationConfig)
    diag: DiagnosticsConfig = field(default_factory=DiagnosticsConfig)

    # Internal registry fields (not included in hash)
    experiment_id: str = ""
    run_id: str = ""
    batch_equivalence: str = "MATCHED"
    # Optional free-text label appended to the generated experiment_id, so two
    # runs of the same architecture (e.g. a gate_mode ablation) get distinct
    # checkpoint/diagnostic names instead of overwriting each other's artifacts.
    variant: str = ""

    # -- Serialization -----------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Return a plain-dict deep copy suitable for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExperimentConfig":
        """Construct an ``ExperimentConfig`` from a plain dict.

        Unknown keys in *data* (and in each sub-dict) are silently ignored
        so that old config files that predate newly-added fields still load
        without error — the new fields simply take their defaults.
        """
        return cls(
            data=DataConfig(**_filter_dataclass_kwargs(DataConfig, data.get("data", {}))),
            model=ModelConfig(**_filter_dataclass_kwargs(ModelConfig, data.get("model", {}))),
            loss=LossConfig(**_filter_dataclass_kwargs(LossConfig, data.get("loss", {}))),
            opt=OptimizationConfig(**_filter_dataclass_kwargs(OptimizationConfig, data.get("opt", {}))),
            train=TrainingConfig(**_filter_dataclass_kwargs(TrainingConfig, data.get("train", {}))),
            eval=EvaluationConfig(**_filter_dataclass_kwargs(EvaluationConfig, data.get("eval", {}))),
            diag=DiagnosticsConfig(**_filter_dataclass_kwargs(DiagnosticsConfig, data.get("diag", {}))),
            experiment_id=data.get("experiment_id", ""),
            run_id=data.get("run_id", ""),
            batch_equivalence=data.get("batch_equivalence", "MATCHED"),
            variant=data.get("variant", ""),
        )

    def save(self, filepath: str) -> None:
        """Serialize this config to a JSON file, creating parent dirs."""
        dir_name = os.path.dirname(filepath)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        with open(filepath, "w") as fh:
            json.dump(self.to_dict(), fh, indent=4)

    @classmethod
    def load(cls, filepath: str) -> "ExperimentConfig":
        """Load a config from a JSON file previously written by ``save()``."""
        with open(filepath, "r") as fh:
            data = json.load(fh)
        return cls.from_dict(data)

    # -- Canonical hash ----------------------------------------------------

    def get_canonical_hash(self) -> str:
        """SHA-256 of the config, excluding volatile / runtime-only fields.

        Fields excluded from the hash (they do not change the mathematical
        output of the model):

        - ``experiment_id``, ``run_id``, ``batch_equivalence``
        - ``data.dataset_root``, ``data.split_manifest_path``
        - ``train.num_workers``, ``train.checkpoint_every_n_steps``
        """
        d = self.to_dict()
        volatile_keys = ["experiment_id", "run_id", "batch_equivalence", "variant"]
        for k in volatile_keys:
            d.pop(k, None)

        if "data" in d:
            d["data"].pop("dataset_root", None)
            d["data"].pop("split_manifest_path", None)

        if "train" in d:
            d["train"].pop("num_workers", None)
            d["train"].pop("checkpoint_every_n_steps", None)

        def _sort_dict(item: Any) -> Any:
            if isinstance(item, dict):
                return {k: _sort_dict(v) for k, v in sorted(item.items())}
            elif isinstance(item, list):
                return [_sort_dict(i) for i in item]
            else:
                return item

        sorted_d = _sort_dict(d)
        json_str = json.dumps(sorted_d, separators=(",", ":"))
        return hashlib.sha256(json_str.encode("utf-8")).hexdigest()

    # -- Validation --------------------------------------------------------

    def validate(self) -> None:
        """Check the config for internal consistency.

        Raises:
            ValueError: If any invariant is violated.

        Warnings:
            If ``batch_per_gpu * grad_accum_steps`` does not divide
            ``effective_global_batch`` evenly (batch mismatch that can only
            be fully validated at runtime when ``world_size`` is known).
        """
        errors: List[str] = []

        # --- Model invariants ---
        if self.model.top_k > self.model.num_experts:
            errors.append(
                f"model.top_k ({self.model.top_k}) must be <= "
                f"model.num_experts ({self.model.num_experts})"
            )
        if self.model.moe_type not in {"none", "dense", "sparse"}:
            errors.append(
                f"model.moe_type must be one of {{'none','dense','sparse'}}, "
                f"got '{self.model.moe_type}'"
            )
        if self.model.moe_16_mode not in {"sparse", "dense"}:
            errors.append(
                f"model.moe_16_mode must be one of {{'sparse','dense'}}, "
                f"got '{self.model.moe_16_mode}'"
            )
        if self.model.gate_mode not in {"renormalized", "dense"}:
            errors.append(
                f"model.gate_mode must be one of {{'renormalized','dense'}}, "
                f"got '{self.model.gate_mode}'"
            )
        if (self.model.moe_16_mode == "dense") != self.loss.moe_16_dense:
            errors.append(
                "model.moe_16_mode and loss.moe_16_dense must agree: "
                "set both to dense or both to sparse."
            )
        if self.model.num_experts < 1:
            errors.append(
                f"model.num_experts must be >= 1, got {self.model.num_experts}"
            )
        if self.model.top_k < 1:
            errors.append(
                f"model.top_k must be >= 1, got {self.model.top_k}"
            )
        if self.model.working_dim < 1 or self.model.decoder_dim < 1:
            errors.append(
                f"model.working_dim and model.decoder_dim must be >= 1, "
                f"got {self.model.working_dim} and {self.model.decoder_dim}"
            )

        # --- Optimization invariants ---
        if self.opt.batch_per_gpu < 1:
            errors.append(
                f"opt.batch_per_gpu must be >= 1, got {self.opt.batch_per_gpu}"
            )
        if self.opt.grad_accum_steps < 1:
            errors.append(
                f"opt.grad_accum_steps must be >= 1, got {self.opt.grad_accum_steps}"
            )
        if self.opt.effective_global_batch < 1:
            errors.append(
                f"opt.effective_global_batch must be >= 1, "
                f"got {self.opt.effective_global_batch}"
            )
        if not 0.0 <= self.opt.warmup_ratio <= 1.0:
            errors.append(
                f"opt.warmup_ratio must be in [0, 1], got {self.opt.warmup_ratio}"
            )
        if self.opt.gradient_clipping < 0:
            errors.append(
                f"opt.gradient_clipping must be >= 0, "
                f"got {self.opt.gradient_clipping}"
            )

        # --- Training invariants ---
        if self.train.epochs < 1:
            errors.append(
                f"train.epochs must be >= 1, got {self.train.epochs}"
            )
        if self.train.seed < 0:
            errors.append(
                f"train.seed must be >= 0, got {self.train.seed}"
            )

        # --- Loss invariants ---
        for name in (
            "bce_weight", "iou_weight", "ssim_weight", "boundary_weight",
            "load_balance_weight", "importance_weight", "z_loss_weight",
            "aux_boundary_weight", "deep_supervision_weight",
        ):
            val = getattr(self.loss, name)
            if val < 0:
                errors.append(f"loss.{name} must be >= 0, got {val}")

        if self.loss.load_balance_weights is not None:
            lbw = self.loss.load_balance_weights
            if len(lbw) != 3:
                errors.append(
                    f"loss.load_balance_weights must have exactly 3 elements "
                    f"[w_stride4, w_stride8, w_stride16], got {len(lbw)}"
                )
            for idx, w in enumerate(lbw):
                if w < 0:
                    errors.append(
                        f"loss.load_balance_weights[{idx}] must be >= 0, got {w}"
                    )

        # --- Evaluation invariants ---
        if self.eval.validation_metric_selection not in {"MAE", "F-measure", "S-measure", "max Dice"}:
            errors.append(
                f"eval.validation_metric_selection unknown: "
                f"'{self.eval.validation_metric_selection}'"
            )

        if errors:
            raise ValueError(
                "ExperimentConfig validation failed:\n  - " + "\n  - ".join(errors)
            )

        # --- Soft warnings (not fatal) ---
        per_gpu_product = self.opt.batch_per_gpu * self.opt.grad_accum_steps
        if self.opt.effective_global_batch % per_gpu_product != 0:
            warnings.warn(
                f"effective_global_batch ({self.opt.effective_global_batch}) is not "
                f"divisible by batch_per_gpu * grad_accum_steps ({per_gpu_product}). "
                f"This may indicate a non-matched batch configuration.",
                UserWarning,
                stacklevel=2,
            )

    # -- Presets -----------------------------------------------------------

    @classmethod
    def from_preset(cls, name: str, overrides: Optional[Dict[str, Any]] = None) -> "ExperimentConfig":
        """Load a config from a named preset, optionally applying overrides.

        Presets are JSON files stored in ``src/presets/``.  Available presets
        can be discovered via :func:`list_presets`.

        Args:
            name: Preset name (maps to ``src/presets/<name>.json``).
            overrides: Optional dict of dot-separated key overrides merged
                on top of the preset (e.g. ``{"model.num_experts": 4}``).

        Returns:
            A fully constructed ``ExperimentConfig``.

        Raises:
            FileNotFoundError: If the preset file does not exist.
            KeyError: If the override path does not map to a valid field.
        """
        registry = _load_preset_registry()
        if name not in registry:
            available = ", ".join(registry.keys()) if registry else "(none)"
            raise FileNotFoundError(
                f"Preset '{name}' not found. Available: {available}"
            )
        cfg = cls.from_dict(registry[name])
        if overrides:
            _apply_overrides(cfg, overrides)
        return cfg


# ---------------------------------------------------------------------------
# Override helper
# ---------------------------------------------------------------------------

def _apply_overrides(cfg: ExperimentConfig, overrides: Dict[str, Any]) -> None:
    """Apply dot-separated key overrides (e.g. ``"model.num_experts" -> 4``).

    The override is applied in-place.  ``KeyError`` is raised if any path
    does not resolve to a real attribute.
    """
    for dotpath, value in overrides.items():
        parts = dotpath.split(".")
        obj: Any = cfg
        for part in parts[:-1]:
            obj = getattr(obj, part)
        if not hasattr(obj, parts[-1]):
            raise KeyError(f"Override path '{dotpath}' does not resolve to a field")
        setattr(obj, parts[-1], value)
