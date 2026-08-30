import hashlib
import json
import os
from dataclasses import dataclass, asdict, field
from typing import Optional, List, Dict, Any

@dataclass
class DataConfig:
    dataset_root: str = "/kaggle/input/wxsod-dataset"
    split_manifest_path: str = ""
    split_manifest_hash: str = ""
    train_sample_count: int = 0
    validation_sample_count: int = 0
    test_sample_counts: Dict[str, int] = field(default_factory=dict)
    max_samples: Optional[int] = None

@dataclass
class ModelConfig:
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
    moe_type: str = "sparse" # Options: "none" (no-moe), "dense" (dense-moe), "sparse" (sparse-moe)

@dataclass
class LossConfig:
    bce_weight: float = 1.0
    iou_weight: float = 1.0
    ssim_weight: float = 0.0
    boundary_weight: float = 0.0
    load_balance_weight: float = 0.01
    importance_weight: float = 0.01
    z_loss_weight: float = 0.0
    aux_boundary_weight: float = 0.0
    deep_supervision_weight: float = 0.4

@dataclass
class OptimizationConfig:
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
    epochs: int = 50
    freeze_backbone_epochs: int = 1
    seed: int = 42
    num_workers: int = 2
    checkpoint_every_n_steps: int = 200

@dataclass
class EvaluationConfig:
    validation_metric_selection: str = "MAE"
    tta_mode: str = "none"
    metric_config: str = "default"

@dataclass
class DiagnosticsConfig:
    routing_diagnostic_epochs: int = 1
    expert_ablation_enabled: bool = False

@dataclass
class ExperimentConfig:
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

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        return cls(
            data=DataConfig(**data.get('data', {})),
            model=ModelConfig(**data.get('model', {})),
            loss=LossConfig(**data.get('loss', {})),
            opt=OptimizationConfig(**data.get('opt', {})),
            train=TrainingConfig(**data.get('train', {})),
            eval=EvaluationConfig(**data.get('eval', {})),
            diag=DiagnosticsConfig(**data.get('diag', {})),
            experiment_id=data.get('experiment_id', ''),
            run_id=data.get('run_id', ''),
            batch_equivalence=data.get('batch_equivalence', 'MATCHED')
        )

    def get_canonical_hash(self) -> str:
        d = self.to_dict()
        # Remove volatile fields from hash calculation
        volatile_keys = ['experiment_id', 'run_id', 'batch_equivalence']
        for k in volatile_keys:
            d.pop(k, None)
            
        # Volatile runtime paths/configs that don't change mathematical output
        if 'data' in d:
            d['data'].pop('dataset_root', None)
            d['data'].pop('split_manifest_path', None)
            
        if 'train' in d:
            d['train'].pop('num_workers', None)
            d['train'].pop('checkpoint_every_n_steps', None)

        def sort_dict(item):
            if isinstance(item, dict):
                return {k: sort_dict(v) for k, v in sorted(item.items())}
            elif isinstance(item, list):
                return [sort_dict(i) for i in item]
            else:
                return item

        sorted_d = sort_dict(d)
        json_str = json.dumps(sorted_d, separators=(',', ':'))
        return hashlib.sha256(json_str.encode('utf-8')).hexdigest()

    def save(self, filepath: str):
        dir_name = os.path.dirname(filepath)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=4)
            
    @classmethod
    def load(cls, filepath: str):
        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)
