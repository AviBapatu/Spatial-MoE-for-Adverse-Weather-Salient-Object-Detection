import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LRScheduler
import math

def get_parameter_groups(model, backbone_lr=2e-5, new_module_lr=1e-4, weight_decay=1e-4):
    """
    Categorize model parameters into 4 logical groups, with decay/no-decay subgroups.
    """
    groups = {
        'backbone_decay': {'params': [], 'lr': backbone_lr, 'weight_decay': weight_decay},
        'backbone_no_decay': {'params': [], 'lr': backbone_lr, 'weight_decay': 0.0},
        
        'moe_decay': {'params': [], 'lr': new_module_lr, 'weight_decay': weight_decay},
        'moe_no_decay': {'params': [], 'lr': new_module_lr, 'weight_decay': 0.0},
        
        'decoder_decay': {'params': [], 'lr': new_module_lr, 'weight_decay': weight_decay},
        'decoder_no_decay': {'params': [], 'lr': new_module_lr, 'weight_decay': 0.0},
        
        'heads_decay': {'params': [], 'lr': new_module_lr, 'weight_decay': weight_decay},
        'heads_no_decay': {'params': [], 'lr': new_module_lr, 'weight_decay': 0.0},
    }
    
    seen_params = set()
    
    # Pre-calculate which modules are normalization modules
    no_decay_modules = set()
    for m_name, module in model.named_modules():
        if isinstance(module, (nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm, nn.InstanceNorm2d)):
            no_decay_modules.add(m_name)
        elif module.__class__.__name__ in ['LayerNorm', 'LayerNorm2d', 'BatchNorm2d']:
            no_decay_modules.add(m_name)
            
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
            
        seen_params.add(param)
        
        # Determine group
        if name.startswith('backbone'):
            group_prefix = 'backbone'
        elif 'moe' in name or 'expert' in name or 'router' in name:
            group_prefix = 'moe'
        elif 'head' in name:
            group_prefix = 'heads'
        else:
            group_prefix = 'decoder'
            
        # Determine no-decay
        is_no_decay = False
        if name.endswith('.bias'):
            is_no_decay = True
        else:
            # Check if parent module is a normalization module
            parent_name = name.rsplit('.', 1)[0] if '.' in name else ''
            if parent_name in no_decay_modules:
                is_no_decay = True
            elif 'relative_position_bias_table' in name:
                is_no_decay = True # usually no decay for position embeddings/biases
                
        if is_no_decay:
            groups[f'{group_prefix}_no_decay']['params'].append(param)
        else:
            groups[f'{group_prefix}_decay']['params'].append(param)
            
    all_trainable_params = set(p for p in model.parameters() if p.requires_grad)
    missing = all_trainable_params - seen_params
    if missing:
        raise ValueError(f"Found {len(missing)} trainable parameters that were not assigned to any group!")
        
    return [g for g in groups.values() if len(g['params']) > 0]

def freeze_backbone(model):
    for name, param in model.named_parameters():
        if name.startswith('backbone'):
            param.requires_grad = False

def unfreeze_backbone(model):
    for name, param in model.named_parameters():
        if name.startswith('backbone'):
            param.requires_grad = True

class WarmupCosineScheduler(LRScheduler):
    def __init__(self, optimizer, warmup_steps, total_steps, min_lr_ratio=0.01, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr_ratio = min_lr_ratio
        super().__init__(optimizer, last_epoch)
        
    def get_lr(self):
        step = self.last_epoch
        if step < self.warmup_steps:
            alpha = float(step) / float(max(1, self.warmup_steps))
            return [base_lr * alpha for base_lr in self.base_lrs]
        else:
            progress = float(step - self.warmup_steps) / float(max(1, self.total_steps - self.warmup_steps))
            cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
            return [
                base_lr * self.min_lr_ratio + base_lr * (1.0 - self.min_lr_ratio) * cosine_decay
                for base_lr in self.base_lrs
            ]

class OptimizationEngine:
    def __init__(self, model, optimizer, scheduler, max_grad_norm=1.0, amp_enabled=True, amp_dtype=torch.float16):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.max_grad_norm = max_grad_norm
        self.amp_enabled = amp_enabled
        self.amp_dtype = amp_dtype
        
        self.scaler = torch.amp.GradScaler('cuda', enabled=amp_enabled)
        self.global_step = 0
        
    def step(self):
        """
        Performs:
        1. Unscale
        2. Grad norm measurement and clipping
        3. Scaler step
        4. Scaler update
        5. Check for overflow
        6. Scheduler step and global step increment if successful
        """
        # Unscale gradients
        self.scaler.unscale_(self.optimizer)
        
        # Clip and measure norm
        pre_clip_grad_norm = nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        
        # Step optimizer
        scale_before = self.scaler.get_scale()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        scale_after = self.scaler.get_scale()
        
        # Detect overflow
        overflow = scale_after < scale_before
        
        if not overflow:
            self.global_step += 1
            if self.scheduler is not None:
                self.scheduler.step()
            
        return overflow, pre_clip_grad_norm

    def state_dict(self):
        return {
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict() if self.scheduler is not None else None,
            'scaler': self.scaler.state_dict(),
            'global_step': self.global_step
        }
        
    def load_state_dict(self, state):
        self.optimizer.load_state_dict(state['optimizer'])
        if self.scheduler is not None and state.get('scheduler') is not None:
            self.scheduler.load_state_dict(state['scheduler'])
        self.scaler.load_state_dict(state['scaler'])
        self.global_step = state['global_step']
