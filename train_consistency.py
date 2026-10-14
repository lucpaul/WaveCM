# train_consistency.py
import torch
from pathlib import Path
from src.consistency_model.model_new import ConsistencyWavelet as Consistency
from src.training import training_consistency_from_diffusion
from config.configuration import Config as config
from src.utils.checkpoint_converter import convert_diffusion_to_consistency
from src.metrics import logger as Logger
from src.metrics import metrics as Metrics


def recover_ema_from_checkpoint(model, checkpoint_path):
    """Manually copy trained weights to EMA if EMA wasn't saved"""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Extract state_dict (handle both direct and nested formats)
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    
    # Load main model weights (strict=False to handle any key mismatches)
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    
    if missing_keys:
        print(f"Missing keys: {len(missing_keys)} keys")
    if unexpected_keys:
        print(f"Unexpected keys: {len(unexpected_keys)} keys")
    
    # Copy current trained weights to EMA models
    model.model_ema.load_state_dict(model.model.state_dict())
    model.dwsr_ema.load_state_dict(model.dwsr.state_dict())
    
    print("⚠ Initialized EMA from trained weights (not true EMA state)")
    print("✓ Model weights loaded from checkpoint")
    print("✓ EMA models synchronized with trained weights")
    
    return model

def check_ema_in_checkpoint(checkpoint_path):
    """Check if checkpoint contains EMA states"""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    has_ema = 'model_ema_state_dict' in checkpoint
    has_dwsr_ema = 'dwsr_ema_state_dict' in checkpoint
    return has_ema and has_dwsr_ema

def _training_paths_from_config(cfg):
    """Resolve training paths exclusively from config values."""
    resume_from_checkpoint = (getattr(cfg, "resume_ckpt_path", "") or "").strip() or None
    diffusion_checkpoint = (getattr(cfg, "diffusion_checkpoint_path", "") or "").strip() or None
    conversion_output_dir = (getattr(cfg, "consistency_conversion_output_dir", "") or "").strip()

    if conversion_output_dir:
        conversion_output_dir = Path(conversion_output_dir)
    else:
        conversion_output_dir = Path(cfg.checkpoint_path) / "conversion"

    return resume_from_checkpoint, diffusion_checkpoint, str(conversion_output_dir)


def main():
    """
    Main training function with support for resumption.

    All filesystem paths are read from config:
    - config.diffusion_checkpoint_path
    - config.resume_ckpt_path
    - config.consistency_conversion_output_dir (optional)
    - config.checkpoint_path (fallback conversion dir)
    """
    resume_from_checkpoint, diffusion_checkpoint, output_dir = _training_paths_from_config(config)
    
    # === SCENARIO 1: Starting fresh from diffusion model ===
    if resume_from_checkpoint is None:
        if not diffusion_checkpoint:
            raise ValueError(
                "config.diffusion_checkpoint_path is empty. "
                "Set it in config/configuration.py to start from a diffusion checkpoint."
            )

        print("\n=== Starting Fresh from Diffusion Model ===\n")
        
        # Convert checkpoint and create config
        conversion_results = convert_diffusion_to_consistency(
            diffusion_checkpoint_path=diffusion_checkpoint,
            output_dir=output_dir,
            run_validation=True
        )
        
        # Create consistency model
        model = Consistency(
            config=config,
            diffusion_checkpoint_path=conversion_results['clean_checkpoint'],
            bins_min=10,
            bins_max=150,
            loss_func='HybridWavelet',
            use_ema=True
        )
        
        model.configure_metrics(use_lpips=False, use_clip=True)
        
        # Train from scratch
        trainer, final_model_path = training_consistency_from_diffusion(
            config=config,
            model=model,
            diffusion_checkpoint_path=conversion_results['clean_checkpoint'],
            resume_ckpt_path=None
        )
    
    # === SCENARIO 2: Resuming from consistency checkpoint ===
    else:
        print(f"\n=== Resuming from Consistency Checkpoint ===")
        print(f"Checkpoint: {resume_from_checkpoint}\n")
        
        # Check if checkpoint has EMA states
        has_ema = check_ema_in_checkpoint(resume_from_checkpoint)
        
        if has_ema:
            # === SCENARIO 2A: EMA states exist - normal resumption ===
            print("✓ EMA states found in checkpoint - using normal resumption\n")
            
            # Create model (will load everything including EMA from checkpoint)
            model = Consistency(
                config=config,
                diffusion_checkpoint_path=None,  # Don't load diffusion weights
                bins_min=10,
                bins_max=150,
                loss_func='HybridWavelet',
                use_ema=True
            )
            
            model.configure_metrics(use_lpips=False, use_clip=True)
            
            # Resume training - Lightning handles loading everything
            trainer, final_model_path = training_consistency_from_diffusion(
                config=config,
                model=model,
                diffusion_checkpoint_path=None,
                resume_ckpt_path=resume_from_checkpoint
            )
        
        else:
            # === SCENARIO 2B: EMA states missing - manual recovery ===
            print("⚠ EMA states NOT found in checkpoint - using manual recovery\n")
            
            # Create model with random initialization
            model = Consistency(
                config=config,
                diffusion_checkpoint_path=None,
                bins_min=10,
                bins_max=150,
                loss_func='HybridWavelet',
                use_ema=True
            )
            
            model.configure_metrics(use_lpips=False, use_clip=True)
            
            # **THIS IS WHERE MANUAL RECOVERY HAPPENS**
            # Load trained weights and copy to EMA before training
            model = recover_ema_from_checkpoint(model, resume_from_checkpoint)
            
            # Resume training - Lightning will load optimizer/scheduler
            # but model weights are already loaded above
            trainer, final_model_path = training_consistency_from_diffusion(
                config=config,
                model=model,
                diffusion_checkpoint_path=None,
                resume_ckpt_path=resume_from_checkpoint
            )
    
    print(f"\n=== Training Complete ===")
    print(f"Final model saved at: {final_model_path}")
    return trainer, model


if __name__ == "__main__":
    trainer, model = main()