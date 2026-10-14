import torch
import json
from pathlib import Path
from typing import Dict, Any


class DiffusionToConsistencyConverter:
    """Utility to convert diffusion model checkpoints for consistency training"""

    def __init__(self, diffusion_checkpoint_path: str):
        self.diffusion_checkpoint_path = diffusion_checkpoint_path
        self.checkpoint = self.load_checkpoint()

    def load_checkpoint(self):
        """Load the diffusion checkpoint"""
        try:
            checkpoint = torch.load(self.diffusion_checkpoint_path, map_location='cpu')
            print(f"Successfully loaded checkpoint from {self.diffusion_checkpoint_path}")
            return checkpoint
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            raise

    def analyze_checkpoint(self) -> Dict[str, Any]:
        """Analyze the structure of the diffusion checkpoint"""
        analysis = {
            'checkpoint_keys': list(self.checkpoint.keys()),
            'has_state_dict': 'state_dict' in self.checkpoint,
            'has_model_state': 'model' in self.checkpoint,
            'pytorch_lightning_format': 'state_dict' in self.checkpoint and 'epoch' in self.checkpoint,
        }

        if 'state_dict' in self.checkpoint:
            state_dict = self.checkpoint['state_dict']
            analysis['model_keys'] = [k for k in state_dict.keys() if k.startswith('model.')]
            analysis['other_keys'] = [k for k in state_dict.keys() if not k.startswith('model.')]
            analysis['total_parameters'] = len(state_dict)

        if 'hyper_parameters' in self.checkpoint:
            analysis['hyperparameters'] = self.checkpoint['hyper_parameters']

        return analysis

    def extract_model_weights(self) -> Dict[str, torch.Tensor]:
        """Extract just the model weights from the checkpoint"""

        if 'state_dict' in self.checkpoint:
            # PyTorch Lightning format
            state_dict = self.checkpoint['state_dict']
            model_weights = {}

            for key, value in state_dict.items():
                if key.startswith('model.'):
                    # Remove 'model.' prefix
                    new_key = key.replace('model.', '')
                    model_weights[new_key] = value
                elif not any(skip in key for skip in ['optimizer', 'lr_scheduler', '_train_loss', '_val_loss', '_ema']):
                    # Include other model-related keys
                    model_weights[key] = value

        else:
            # Direct state dict format
            model_weights = self.checkpoint

        return model_weights

    def save_clean_checkpoint(self, output_path: str):
        """Save a clean checkpoint with just model weights"""

        model_weights = self.extract_model_weights()

        clean_checkpoint = {
            'state_dict': model_weights,
            'source': 'diffusion_model_conversion',
            'original_checkpoint': self.diffusion_checkpoint_path,
        }

        # Preserve important metadata if available
        if 'hyper_parameters' in self.checkpoint:
            clean_checkpoint['hyper_parameters'] = self.checkpoint['hyper_parameters']

        if 'epoch' in self.checkpoint:
            clean_checkpoint['source_epoch'] = self.checkpoint['epoch']

        torch.save(clean_checkpoint, output_path)
        print(f"Clean checkpoint saved to {output_path}")

        return output_path

    def create_consistency_config(self, output_path: str) -> Dict[str, Any]:
        """Create a configuration file for consistency training"""

        config = {}

        # Extract hyperparameters if available
        if 'hyper_parameters' in self.checkpoint:
            hparams = self.checkpoint['hyper_parameters']

            # Map diffusion hyperparameters to consistency parameters
            config.update({
                'in_channels': hparams.get('in_channels', 3),
                'out_channels': hparams.get('out_channels', 3),
                'channels': hparams.get('channels', [128, 256, 512, 1024]),
                'sample_dimension': hparams.get('sample_dimension', [192, 192]),
                'data_std': hparams.get('data_std', 1.0),
                'time_min': hparams.get('time_min', 0.002),
                'time_max': hparams.get('time_max', 80.0),
                'clip_output': hparams.get('clip_output', True),
                'down_block_types': hparams.get('down_block_types', ["DownBlock2D", "DownBlock2D", "DownBlock2D", "AttnDownBlock2D"]),
                'up_block_types': hparams.get('up_block_types', ["AttnUpBlock2D", "UpBlock2D", "UpBlock2D", "UpBlock2D"]),
            })

        # Add consistency-specific defaults
        consistency_defaults = {
            'lr': 5e-6,
            'n_epochs': 30,
            'batch_size': 8,
            'bins_min': 2,
            'bins_max': 150,
            'bins_rho': 7.0,
            'initial_ema_decay': 0.9,
            'loss_func': 'LPIPS',
            'use_ema': True,
            'log_images_every_n_steps': 100,
            'check_val_every_n_epoch': 1,
            'grad_clip_norm': 1.0,
        }

        config.update(consistency_defaults)

        with open(output_path, 'w') as f:
            json.dump(config, f, indent=2)

        print(f"Consistency configuration saved to {output_path}")
        return config

    def validate_compatibility(self, consistency_config) -> bool:
        """Validate that the diffusion checkpoint is compatible with consistency training"""

        issues = []

        # Check if model weights exist
        model_weights = self.extract_model_weights()
        if not model_weights:
            issues.append("No model weights found in checkpoint")

        # Check for required architecture parameters
        required_params = ['in_channels', 'out_channels', 'channels']
        if 'hyper_parameters' in self.checkpoint:
            hparams = self.checkpoint['hyper_parameters']
            for param in required_params:
                if param not in hparams:
                    issues.append(f"Missing required parameter: {param}")

        # Check for conditioning support (super-resolution specific)
        if 'hyper_parameters' in self.checkpoint:
            hparams = self.checkpoint['hyper_parameters']
            expected_input_channels = hparams.get('in_channels', 3) * 2

            # Look for evidence of conditioning in the model
            first_conv_key = None
            for key in model_weights.keys():
                if 'conv_in.weight' in key or 'down_blocks.0.resnets.0.conv1.weight' in key:
                    first_conv_key = key
                    break

            if first_conv_key:
                first_conv_shape = model_weights[first_conv_key].shape
                if first_conv_shape[1] != expected_input_channels:
                    issues.append(f"Input channel mismatch: expected {expected_input_channels}, got {first_conv_shape[1]}")

        if issues:
            print("Compatibility issues found:")
            for issue in issues:
                print(f"  - {issue}")
            return False

        print("Checkpoint is compatible with consistency training")
        return True


def convert_diffusion_to_consistency(
    diffusion_checkpoint_path: str,
    output_dir: str,
    run_validation: bool = True,
) -> Dict[str, str]:
    """Complete conversion pipeline"""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    converter = DiffusionToConsistencyConverter(diffusion_checkpoint_path)

    # Analyze checkpoint
    print("Analyzing diffusion checkpoint...")
    analysis = converter.analyze_checkpoint()
    print(json.dumps(analysis, indent=2, default=str))

    # Save clean checkpoint
    clean_checkpoint_path = output_dir / "diffusion_weights_for_consistency.ckpt"
    converter.save_clean_checkpoint(str(clean_checkpoint_path))

    # Create consistency config
    config_path = output_dir / "consistency_config.json"
    consistency_config = converter.create_consistency_config(str(config_path))

    # Validate compatibility
    if run_validation:
        is_compatible = converter.validate_compatibility(consistency_config)
        if not is_compatible:
            print("WARNING: Compatibility issues detected. Manual adjustments may be needed.")

    return {
        'clean_checkpoint': str(clean_checkpoint_path),
        'config': str(config_path),
        'analysis': analysis,
    }


if __name__ == "__main__":
    diffusion_ckpt = "/path/to/your/diffusion_model.ckpt"
    output_directory = "./consistency_conversion_output"

    results = convert_diffusion_to_consistency(
        diffusion_checkpoint_path=diffusion_ckpt,
        output_dir=output_directory,
        run_validation=True,
    )

    print("\nConversion completed!")
    print(f"Clean checkpoint: {results['clean_checkpoint']}")
    print(f"Configuration: {results['config']}")
    print(f"Ready for consistency training!")

    from src.consistency_model.model_new import ConsistencyWavelet as Consistency
    from config.configuration import Config

    import json
    with open(results['config'], 'r') as f:
        config_dict = json.load(f)

    config = Config(**config_dict)
    consistency_model = Consistency(
        config=config,
        diffusion_checkpoint_path=results['clean_checkpoint'],
    )

    print("Consistency model initialized successfully!")
