from dataclasses import dataclass, field
from typing import List
import argparse
import toml
import copy

@dataclass
class InferenceConfig:

    # ── Paths ──────────────────────────────────────────────────────────────────

    # Consistency model checkpoint to evaluate (required)
    consistency_ckpt: str = "path/to/consistency_model.ckpt"
    # consistency_ckpt: str = None
    # Diffusion model checkpoint for comparison (set to None to skip)
    diffusion_ckpt: str = "path/to/diffusion_model.pth"
    # diffusion_ckpt: str = None

    # Test data directories
    lr_path: str = "path/to/low_resolution_data"
    hr_path: str = "path/to/high_resolution_data"

    # Where to write SR images and metric CSVs
    output_dir: str = "path/to/output_directory"

    # ── Sampling ───────────────────────────────────────────────────────────────

    # Consistency sampling steps: 1 = single-step (fastest), >1 = multi-step
    n_consistency_steps: int = 10

    # Whether to use EMA weights for consistency model inference
    use_ema: bool = True

    # Run deterministic DWSR-only baseline from consistency model
    dwsr_consistency: bool = False

    # Run deterministic DWSR-only baseline from diffusion model
    dwsr_diffusion: bool = False

    # Run consistency ensemble and average outputs
    consistency_ensemble: bool = False
    ensemble_runs: int = 8

    # Run consistency uncertainty inference and save 4-channel confidence heatmaps
    uncertainty_backend: str = "consistency"  # "consistency" or "diffusion"
    consistency_uncertainty: bool = False
    uncertainty_runs: int = 8
    uncertainty_metric: str = "std"    # "std" or "var"
    uncertainty_space: str = "image"   # "image" or "wavelet"
    uncertainty_normalization: str = "relative_mean"  # "absolute" or "relative_mean"
    uncertainty_seed_offset: int = 0
    uncertainty_eps: float = 1e-8

    # Standalone uncertainty_eval.py controls
    uncertainty_blend_strength: float = 1.0
    use_confidence_normalization: bool = True
    # Refinement anchor for confidence-guided blending: "robust" or "baseline"
    uncertainty_refine_anchor: str = "robust"
    uncertainty_robust_temperature: float = 1.0
    uncertainty_error_metric: str = "squared"  # "abs" or "squared"
    uncertainty_interval_levels: List[float] = field(default_factory=lambda: [0.5, 0.8, 0.9, 0.95])
    uncertainty_remove_fractions: List[float] = field(default_factory=lambda: [0.0, 0.1, 0.2, 0.3, 0.4, 0.5])

    # Number of DDPM steps for the diffusion model (only used if diffusion_ckpt is set)
    n_diffusion_steps: int = 10
    use_ddim: bool = True       # set True to use DDIM instead of DDPM
    ddim_eta: float = 0.0         # 0.0 = deterministic, 1.0 = stochastic
    # Run diffusion/DDIM ensemble and average outputs across seeded runs.
    diffusion_ensemble: bool = True
    diffusion_ensemble_runs: int = 4
    diffusion_ensemble_seed_offset: int = 1
    diffusion_ensemble_space: str = "wavelet"  # "image" or "wavelet"

    # Run diffusion/DDIM uncertainty inference and save confidence heatmaps.
    diffusion_uncertainty: bool = True
    diffusion_uncertainty_runs: int = 8
    diffusion_uncertainty_metric: str = "std"    # "std" or "var"
    diffusion_uncertainty_space: str = "image"   # "image" or "wavelet"
    diffusion_uncertainty_seed_offset: int = 1
    diffusion_uncertainty_eps: float = 1e-8
    # Reverse-process variance mode for diffusion sampling:
    # - "fixed_large": use beta_t (legacy/default)
    # - "fixed_small": use posterior variance beta_t_tilde
    diffusion_posterior_variance: str = "fixed_small"
    # If True, regenerate a shorter beta schedule at inference time (legacy behavior).
    # Keep False for training-consistent schedule with DDIM step reduction.
    enable_legacy_schedule_resample: bool = False

    # ── Data ───────────────────────────────────────────────────────────────────

    # Upsample LR images by this integer scale factor before feeding to the model
    # (bicubic). Set to None if LR and HR are already at the same spatial size.
    lr_scale: int = None

    batch_size: int = 1
    num_workers: int = 1

    # ── Output ─────────────────────────────────────────────────────────────────

    # Save SR .tif outputs to output_dir/consistency_sr and output_dir/diffusion_sr
    save_images: bool = True
