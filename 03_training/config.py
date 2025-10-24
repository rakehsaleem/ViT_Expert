"""
Training Configuration for ViT Expert Distillation

This module contains configuration settings and hyperparameters for training
the teacher and student models using knowledge distillation.
"""

import os
from typing import Dict, Any


class TrainingConfig:
    """
    Configuration class for training parameters and hyperparameters.
    """
    
    def __init__(self):
        """Initialize training configuration with default values."""
        
        # Model Configuration
        self.model_config = {
            # Teacher Model
            'teacher_vit_model': 'vit_base_patch16_224',
            'teacher_visual_dim': 768,
            'teacher_gaze_dim': 512,
            'teacher_fusion_dim': 512,
            'teacher_checkpoint': None,  # Path to pre-trained teacher model
            
            # Student Model
            'student_backbone': 'mobilenet_v2',
            'student_width_mult': 1.0,
            'student_pretrained': True,
            'img_size': 224,
        }
        
        # Training Configuration
        self.training_config = {
            'num_epochs': 100,
            'batch_size': 32,
            'learning_rate': 1e-4,
            'weight_decay': 1e-4,
            'grad_clip': 1.0,
            'scheduler': 'cosine',  # 'cosine', 'step', 'plateau'
            'min_lr': 1e-6,
            'step_size': 30,
            'gamma': 0.1,
        }
        
        # Loss Configuration
        self.loss_config = {
            'distillation_weight': 1.0,
            'attention_weight': 0.5,
            'hard_label_weight': 0.3,
            'temperature': 3.0,
            'hard_loss_type': 'nss',  # 'nss', 'ce', 'mse', 'bce'
        }
        
        # Data Configuration
        self.data_config = {
            'data_dir': '01_data',
            'gaze_seq_len': 10,
            'num_workers': 4,
            'pin_memory': True,
        }
        
        # Logging and Checkpointing
        self.logging_config = {
            'use_wandb': False,
            'checkpoint_dir': '02_models/student_model/checkpoints',
            'save_freq': 10,
            'log_freq': 10,
        }
        
        # Device Configuration
        self.device_config = {
            'device': 'auto',  # 'auto', 'cuda', 'cpu'
            'mixed_precision': True,
        }
    
    def update_config(self, updates: Dict[str, Any]):
        """
        Update configuration with new values.
        
        Args:
            updates: Dictionary of configuration updates
        """
        for key, value in updates.items():
            if hasattr(self, key):
                if isinstance(value, dict) and isinstance(getattr(self, key), dict):
                    getattr(self, key).update(value)
                else:
                    setattr(self, key, value)
            else:
                print(f"Warning: Unknown configuration key '{key}'")
    
    def get_device(self):
        """Get the appropriate device for training."""
        import torch
        
        if self.device_config['device'] == 'auto':
            return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            return torch.device(self.device_config['device'])
    
    def create_checkpoint_dir(self):
        """Create checkpoint directory if it doesn't exist."""
        os.makedirs(self.logging_config['checkpoint_dir'], exist_ok=True)
    
    def print_config(self):
        """Print the current configuration."""
        print("=" * 50)
        print("TRAINING CONFIGURATION")
        print("=" * 50)
        
        for section_name, section_config in [
            ("Model", self.model_config),
            ("Training", self.training_config),
            ("Loss", self.loss_config),
            ("Data", self.data_config),
            ("Logging", self.logging_config),
            ("Device", self.device_config),
        ]:
            print(f"\n{section_name.upper()} CONFIG:")
            for key, value in section_config.items():
                print(f"  {key}: {value}")
        
        print("=" * 50)


# Predefined configurations for different scenarios
def get_small_config() -> TrainingConfig:
    """Get configuration for small-scale training/testing."""
    config = TrainingConfig()
    
    config.update_config({
        'model_config': {
            'student_width_mult': 0.5,  # Smaller student model
            'student_pretrained': False,  # Disable pretrained for smaller model
        },
        'training_config': {
            'num_epochs': 10,
            'batch_size': 16,
            'learning_rate': 2e-4,
        },
        'data_config': {
            'num_workers': 0,  # Avoid multiprocessing issues
        },
        'logging_config': {
            'save_freq': 5,
        }
    })
    
    return config


def get_fast_config() -> TrainingConfig:
    """Get configuration for fast training with reduced complexity."""
    config = TrainingConfig()
    
    config.update_config({
        'model_config': {
            'teacher_vit_model': 'vit_small_patch16_224',  # Smaller teacher
            'student_width_mult': 0.75,
        },
        'training_config': {
            'num_epochs': 50,
            'batch_size': 64,
            'learning_rate': 2e-4,
        },
        'loss_config': {
            'temperature': 2.0,  # Lower temperature for faster convergence
        }
    })
    
    return config


def get_production_config() -> TrainingConfig:
    """Get configuration for production training with full resources."""
    config = TrainingConfig()
    
    config.update_config({
        'training_config': {
            'num_epochs': 200,
            'batch_size': 64,
            'learning_rate': 5e-5,
            'weight_decay': 1e-5,
        },
        'loss_config': {
            'distillation_weight': 1.2,
            'attention_weight': 0.8,
            'hard_label_weight': 0.5,
            'temperature': 4.0,
        },
        'logging_config': {
            'use_wandb': True,
            'save_freq': 5,
        },
        'device_config': {
            'mixed_precision': True,
        }
    })
    
    return config


# Example usage
if __name__ == "__main__":
    # Test default configuration
    config = TrainingConfig()
    config.print_config()
    
    # Test small configuration
    print("\n" + "="*50)
    print("SMALL CONFIGURATION")
    print("="*50)
    small_config = get_small_config()
    small_config.print_config()
    
    # Test configuration updates
    print("\n" + "="*50)
    print("UPDATED CONFIGURATION")
    print("="*50)
    config.update_config({
        'training_config': {
            'num_epochs': 50,
            'batch_size': 16,
        },
        'model_config': {
            'student_width_mult': 0.5,
        }
    })
    config.print_config()
