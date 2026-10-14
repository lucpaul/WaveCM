import torch
from tqdm import tqdm
import numpy as np 
from pathlib import Path
from diffusers import UNet2DModel

from src.sde_model.inference import Inference
from config.configuration import Config
from src.consistency_model.model import Consistency
from src.utils.transforms import apply_inverse_transforms


class ConsistencyInference(Inference):

    def __init__(self, config: Config) -> None:
        super().__init__(config)

        self.config = config


    def load_model(self,
                   checkpoint_fname: str = 'best'):
        """Loads the model from a checkpoint.

            checkpoint_fname: Path to the .ckpt file. Default 'best' loads .../best_model.ckpt
        """

        if checkpoint_fname == 'best':
            self.checkpoint_path = f'{self.config.checkpoint_path}/best_model.ckpt'
        else:
            self.checkpoint_path = f'{self.config.checkpoint_path}/{checkpoint_fname}'
        assert Path(self.checkpoint_path).exists(), f"Path {self.checkpoint_path} does not exist."

        self.checkpoint = torch.load(self.checkpoint_path)

        model_hyperparameters = ['channels', 'down_block_types', 'up_block_types', 'diffusion_model',
                                 'sigma', 'sigma_max', 'sigma_min', 'epsilon']

        config_checkpoint = {}
        for key in self.checkpoint['hyper_parameters'].keys(): 

            if key in model_hyperparameters:
                setattr(self.config, key, self.checkpoint['hyper_parameters'][key])

            config_checkpoint[key] = self.checkpoint['hyper_parameters'][key]

        network = UNet2DModel(
            in_channels=1,
            out_channels=1,
            block_out_channels=(128, 128, 256, 256),
            down_block_types=(
                "DownBlock2D",
                "DownBlock2D",
                "DownBlock2D",
                "DownBlock2D"
            ),
            up_block_types=(
                "UpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
            ),
        )
        self.model = Consistency.load_from_checkpoint(model=network,
                                         checkpoint_path=self.checkpoint_path,
                                         config=self.config)

        self.model.config_checkpoint = config_checkpoint

        self.model.to(self.device)
        self.model.eval()

    def run(self,
            convert_to_xarray: bool=True,
            inverse_transform: bool=True,
            num_samples: int = 16,
            steps: int = 1,
            use_ema: bool = False
            ) -> dict:
        """Executes the inference sampling unconditionally from the learned distribution.

        Args:
            convert_to_xarray: Convertes torch tensor results to xarray dataset.
            inverse_transform: Transform results back to phyical space.
            num_samples: Number of generated samples per batch.
            steps: Number of integration steps.
            use_ema: Enables exponential moving average model 

        Returns:
            Dictionary containing generated samples
        """
 
        all_samples = []

        for b in tqdm(range(self.config.num_batches)):

            samples = self.model.sample(
                                        num_samples = num_samples,
                                        steps = steps,
                                        x_image_size = self.config.sample_dimension[0],
                                        y_image_size = self.config.sample_dimension[1],
                                        use_ema = use_ema)
            all_samples.append(samples)

        all_samples = torch.cat(all_samples).cpu().numpy()

        if all_samples.shape[-2] == 64:
            all_samples = all_samples[:,:,2:62] # remove padding

        if convert_to_xarray:
            all_samples = self.convert_to_xarray(all_samples)

        else:
            all_samples = all_samples

        if inverse_transform:
            all_samples = apply_inverse_transforms(all_samples,
                                                   self.training_target,
                                                   self.config)
    
        return {'generated': all_samples}


    def run_stroke_guidance(self,
            esm_dataloader,
            sample_times,
            convert_to_xarray: bool=True,
            inverse_transform: bool=True,
            num_samples = 1,
            num_batches = 1,
            steps = 1,
            use_ema=True) -> dict:
        """Executes the inference sampling by noising an upsampled ESM field.

        Args:
            esm_dataloader: Dataloader instance from which ESM fields a loaded.
            sample_times: Noising time t*.
            convert_to_xarray: Convertes torch tensor results to xarray dataset.
            inverse_transform: Yransform results back to phyical space.
            num_samples: Number of generated samples per batch.
            num_batches: Number of generated batches.
            steps: Number of integration steps.
            use_ema: Enables exponential moving average model 

        Returns:
            Dictionary of generated results, noised esm fields and raw esm fields.

          
            
         """
 
        all_esm = []
        all_samples = []
        all_conditions = []

        for x in tqdm(esm_dataloader):
            x = torch.from_numpy(x.values)
            for b in range(num_batches):
                init_x = x.to(self.device)
                init_x = init_x.unsqueeze(0)

                samples, conditionings = self.model.sample_conditional(
                                                        init_x,
                                                        init_x.shape[-2],
                                                        init_x.shape[-1],
                                                        num_samples = num_samples,
                                                        steps = steps,
                                                        sample_times = sample_times,
                                                        use_ema = use_ema)

                all_esm.append(init_x.cpu())
                all_samples.append(samples.cpu())
                all_conditions.append(conditionings.cpu())

        all_esm = torch.cat(all_esm)
        all_samples = torch.cat(all_samples)
        all_conditions = torch.cat(all_conditions)

        if convert_to_xarray:
            all_esm = self.convert_to_xarray(all_esm.numpy())
            all_samples = self.convert_to_xarray(all_samples.numpy())
            all_conditions = self.convert_to_xarray(all_conditions.numpy())

        else:
            all_esm = all_esm
            all_samples = all_samples
            all_conditions = all_conditions

        if inverse_transform:
            all_esm = apply_inverse_transforms(all_esm,
                                               self.training_target,
                                               self.config)
            all_samples = apply_inverse_transforms(all_samples,
                                                   self.training_target,
                                                   self.config)
            all_conditions = apply_inverse_transforms(all_conditions,
                                                   self.training_target,
                                                   self.config)
    
        return {'generated': all_samples, 'conditions': all_conditions, 'esm': all_esm}

    @staticmethod
    def _uncertainty_to_confidence(uncertainty: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        """Convert uncertainty values to [0, 1] confidence heatmaps."""
        uncertainty = uncertainty.float()
        u_min = uncertainty.amin(dim=(-2, -1), keepdim=True)
        u_max = uncertainty.amax(dim=(-2, -1), keepdim=True)
        normalized = (uncertainty - u_min) / (u_max - u_min).clamp_min(eps)
        return (1.0 - normalized).clamp(0.0, 1.0)

    def run_stroke_guidance_uncertainty(
        self,
        esm_dataloader,
        sample_times,
        convert_to_xarray: bool = True,
        inverse_transform: bool = True,
        num_samples: int = 1,
        num_batches: int = 1,
        steps: int = 1,
        use_ema: bool = True,
        n_runs: int = 8,
        uncertainty_metric: str = 'std',
        uncertainty_space: str = 'image',
        seed_offset: int = 0,
        eps: float = 1e-8,
    ) -> dict:
        """
        Generate confidence heatmaps from repeated seeded conditional inference.

        Brighter pixels indicate higher confidence (smaller std/var across runs).
        """
        uncertainty_metric = uncertainty_metric.lower()
        uncertainty_space = uncertainty_space.lower()

        if uncertainty_metric not in {'std', 'var'}:
            raise ValueError("uncertainty_metric must be 'std' or 'var'.")
        if uncertainty_space not in {'image', 'wavelet'}:
            raise ValueError("uncertainty_space must be 'image' or 'wavelet'.")

        supports_wavelet = all(
            hasattr(self.model, attr)
            for attr in ['apply_dwt', 'apply_idwt', '_forward_conditional', 'image_time_product']
        )
        if uncertainty_space == 'wavelet' and not supports_wavelet:
            raise ValueError(
                "Wavelet-space uncertainty requires a wavelet consistency model with "
                "apply_dwt/apply_idwt/_forward_conditional."
            )

        all_esm = []
        all_confidence = []
        all_uncertainty = []

        for x in tqdm(esm_dataloader):
            x = torch.from_numpy(x.values)
            for _ in range(num_batches):
                init_x = x.to(self.device).unsqueeze(0)
                run_outputs = []

                for run_idx in range(n_runs):
                    seed = seed_offset + run_idx
                    torch.manual_seed(seed)
                    if torch.cuda.is_available():
                        torch.cuda.manual_seed_all(seed)

                    if uncertainty_space == 'wavelet':
                        conditioning_dwt = self.model.apply_dwt(init_x)
                        x_sr = self.model.dwsr_ema(conditioning_dwt) if use_ema else self.model.dwsr(conditioning_dwt)
                        model_time = torch.tensor([[self.model.time_max]], device=self.device)

                        noise = torch.randn(
                            (init_x.shape[0], 16, conditioning_dwt.shape[2], conditioning_dwt.shape[3]),
                            device=self.device,
                        )
                        output_dwt = self.model.image_time_product(noise, model_time.squeeze(-1)) + x_sr

                        model_fn = self.model.model_ema if use_ema else self.model.model
                        dwsr_fn = self.model.dwsr_ema if use_ema else self.model.dwsr
                        output_dwt = self.model._forward_conditional(
                            model_fn, dwsr_fn, output_dwt, model_time, conditioning_dwt
                        )

                        if steps > 1:
                            _timesteps = torch.linspace(
                                self.model.bins_max - 1, 0, steps, device=self.device
                            ).long()
                            times_seq = self.model.timesteps_to_times(_timesteps, self.model.bins_max)

                            for step_time in times_seq[1:]:
                                step_time_batch = step_time.expand(init_x.shape[0], 1)
                                step_noise = torch.randn(output_dwt.shape, device=self.device)
                                re_noised = output_dwt + self.model.image_time_product(
                                    step_noise, step_time_batch.squeeze(-1)
                                )
                                output_dwt = self.model._forward_conditional(
                                    model_fn, dwsr_fn, re_noised, step_time_batch, conditioning_dwt
                                )

                        run_outputs.append(output_dwt.detach().cpu())
                    else:
                        samples, _ = self.model.sample_conditional(
                            init_x,
                            init_x.shape[-2],
                            init_x.shape[-1],
                            num_samples=num_samples,
                            steps=steps,
                            sample_times=sample_times,
                            use_ema=use_ema,
                        )
                        run_outputs.append(samples.detach().cpu())

                stacked = torch.stack(run_outputs, dim=0)
                if uncertainty_metric == 'std':
                    uncertainty_map = stacked.std(dim=0, unbiased=False)
                else:
                    uncertainty_map = stacked.var(dim=0, unbiased=False)

                if uncertainty_space == 'wavelet':
                    uncertainty_map = self.model.apply_idwt(
                        uncertainty_map.to(self.device),
                        init_x.shape[-2],
                        init_x.shape[-1],
                    ).cpu()

                confidence_map = self._uncertainty_to_confidence(uncertainty_map, eps=eps)

                all_esm.append(init_x.cpu())
                all_confidence.append(confidence_map)
                all_uncertainty.append(uncertainty_map)

        all_esm = torch.cat(all_esm)
        all_confidence = torch.cat(all_confidence)
        all_uncertainty = torch.cat(all_uncertainty)

        if convert_to_xarray:
            all_esm = self.convert_to_xarray(all_esm.numpy())
            all_confidence = self.convert_to_xarray(all_confidence.numpy())
            all_uncertainty = self.convert_to_xarray(all_uncertainty.numpy())

        if inverse_transform:
            all_esm = apply_inverse_transforms(all_esm, self.training_target, self.config)

        return {
            'generated': all_confidence,
            'uncertainty': all_uncertainty,
            'esm': all_esm,
        }