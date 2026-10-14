from dataclasses import asdict
import json
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

from src.data import get_dataloaders_superres
from config.configuration import Config
from src.utils.utils import MyProgressBar


def training_consistency_from_diffusion(
    config: Config(),
    model: pl.LightningModule,
    diffusion_checkpoint_path: str,
    verbose=True,
    resume_ckpt_path=None
):
    """ Training function for consistency model initialized from diffusion model.

    Args:
        config: Configuration containing hyperparameters and file paths
        model: The consistency model to be trained
        diffusion_checkpoint_path: Path to the pre-trained diffusion model checkpoint
        verbose: Prints the training configuration
        resume_ckpt_path: Resumes training from a saved consistency model checkpoint if provided.
    """

    pl.seed_everything(42, workers=True)

    if verbose:
        print(f'Initializing consistency model from diffusion checkpoint: {diffusion_checkpoint_path}')
        print(f'Saving checkpoints at: {config.checkpoint_path}')
        # print(json.dumps(asdict(config), sort_keys=False, indent=4))

    # Callbacks
    callbacks = []

    # Resume path
    # resume_ckpt_path = config.resume_ckpt_path
    
    # Model checkpoints
    checkpoints = ModelCheckpoint(
        dirpath=config.checkpoint_path, 
        save_top_k=3,  # Keep more checkpoints for consistency training
        monitor="val_loss",
        filename=config.name+"-{epoch:02d}-{val_loss:.4f}",
        save_last=True,
        every_n_epochs = 5
    )

    checkpoints.CHECKPOINT_NAME_LAST = config.name+"_last"

    callbacks.append(checkpoints)
    
    # Learning rate monitor
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    callbacks.append(lr_monitor)

    # Custom progress bar
    if hasattr(config, 'use_progress_bar') and config.use_progress_bar:
        progressbar = MyProgressBar()
        callbacks.append(progressbar)

    # TensorBoard logger
    tb_logger = TensorBoardLogger(
        config.tensorboard_path,
        name=f"{config.name}_consistency",
        default_hp_metric=False,
        # version=f"{config.date_time}_from_diffusion"
    )

    model.config.tensorboard_path = config.tensorboard_path

    # Initialize trainer with consistency-specific settings
    trainer = pl.Trainer(
        max_epochs=config.n_epochs,
        callbacks=callbacks,
        deterministic=False,
        check_val_every_n_epoch=config.check_val_every_n_epoch,
        gradient_clip_val=config.grad_clip_norm,
        logger=tb_logger,
        log_every_n_steps=25,  # More frequent logging for consistency training
        precision='16-mixed' if hasattr(config, 'use_mixed_precision') and config.use_mixed_precision else 32,
        accumulate_grad_batches=getattr(config, 'accumulate_grad_batches', 1),
        # Uncomment for debugging:
        # limit_train_batches=20,
        # limit_val_batches=20
    )

    # Get dataloaders
    dataloaders = get_dataloaders_superres(
        config,
        n_workers=config.n_workers,
        use_mnist=getattr(config, 'use_mnist', False)
    )

    print(f"Training dataset size: {len(dataloaders['train'].dataset)}")
    print(f"Validation dataset size: {len(dataloaders['val'].dataset)}")

    # Train the consistency model
    trainer.fit(
        model,
        train_dataloaders=dataloaders['train'],
        val_dataloaders=dataloaders['val'],
        ckpt_path=resume_ckpt_path
    ) 

    # Save final model
    final_model_path = f"{config.checkpoint_path}/{config.name}_final.ckpt"
    trainer.save_checkpoint(final_model_path)
    print(f"Final model saved to: {final_model_path}")

    return trainer, final_model_path


def create_consistency_config_from_diffusion(diffusion_config, consistency_specific_params=None):
    """Helper function to create consistency config from diffusion config"""
    
    # Copy base configuration
    consistency_config = Config(**diffusion_config.__dict__)
    
    # Consistency-specific adjustments
    if consistency_specific_params:
        for key, value in consistency_specific_params.items():
            setattr(consistency_config, key, value)
    
    # Default consistency-specific parameters
    consistency_defaults = {
        'lr': 1e-5,  # Lower learning rate for consistency training
        'n_epochs': 50,  # Fewer epochs needed
        'batch_size': max(4, consistency_config.batch_size // 2),  # Smaller batch size
        'name': f"{consistency_config.name}_consistency",
        'check_val_every_n_epoch': 1,  # Validate every epoch
    }
    
    for key, value in consistency_defaults.items():
        if not hasattr(consistency_config, key) or getattr(consistency_config, key) is None:
            setattr(consistency_config, key, value)
    
    return consistency_config


# Example usage function
def main_training_pipeline(diffusion_checkpoint_path: str, diffusion_config: Config):
    """Complete pipeline for training consistency model from diffusion checkpoint"""
    
    # Create consistency-specific config
    consistency_params = {
        'lr': 5e-6,  # Lower learning rate
        'n_epochs': 30,
        'batch_size': 16,
        'log_images_every_n_steps': 100,
    }
    
    consistency_config = create_consistency_config_from_diffusion(
        diffusion_config, 
        consistency_params
    )
    
    # Initialize consistency model with diffusion checkpoint
    from src.model import Consistency
    
    consistency_model = Consistency(
        config=consistency_config,
        diffusion_checkpoint_path=diffusion_checkpoint_path,
        bins_min=2,
        bins_max=150,
        loss_func='LPIPS',  # or 'MSE' depending on your diffusion model
        use_ema=True,
        log_images_every_n_steps=consistency_params['log_images_every_n_steps']
    )
    
    # Train the consistency model
    trainer, final_model_path = training_consistency_from_diffusion(
        config=consistency_config,
        model=consistency_model,
        diffusion_checkpoint_path=diffusion_checkpoint_path,
        verbose=True
    )
    
    return trainer, final_model_path, consistency_model


if __name__ == "__main__":
    # Example usage
    diffusion_checkpoint = "/path/to/your/diffusion/model.ckpt"
    
    # You'll need to load or recreate your diffusion config
    # This is just a placeholder - use your actual diffusion config
    from config.configuration import Config
    diffusion_config = Config()
    
    trainer, model_path, model = main_training_pipeline(
        diffusion_checkpoint, 
        diffusion_config
    )
    
    print(f"Consistency model training completed!")
    print(f"Final model saved at: {model_path}")