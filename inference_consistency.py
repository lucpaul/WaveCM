"""
inference_eval.py

Inference and evaluation script to compare the Consistency Model against the
Diffusion Model on a test dataset of .tif files.

Usage:
    python inference_eval.py

Edit the InferenceConfig dataclass below to set your paths and options before running.
"""

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import tifffile
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from tqdm import tqdm

# ── project imports ────────────────────────────────────────────────────────────
from src.consistency_model.model_new import ConsistencyWavelet as Consistency
from config.configuration import Config as config
from config.configuration_inference_default import InferenceConfig
from src.metrics.metrics import calculate_clipscore, calculate_lpips, calculate_psnr, calculate_ssim, tensor2img
from src.data import get_test_dataloader
from src.diffusion_model.data.util import adaptive_instance_normalization
from src.utils.checkpoint_converter import convert_diffusion_to_consistency
from src.utils.convert_checkpoint_darts import convert_checkpoint_for_darts
from src.utils.uncertainty import uncertainty_to_confidence


# ══════════════════════════════════════════════════════════════════════════════
# Model loaders
# ══════════════════════════════════════════════════════════════════════════════

def load_consistency_model(ckpt_path: str, device: torch.device) -> Consistency:
    """Load a ConsistencyWavelet checkpoint."""
    print(f"\nLoading consistency model from:\n  {ckpt_path}")
    torch.serialization.add_safe_globals([config])
    model = Consistency.load_from_checkpoint(
        ckpt_path,
        config=config,
        bins_min=10,
        bins_max=150,
        loss_func="HybridWavelet",
        use_ema=True,
        map_location=device,
        strict=False,
        weights_only=False
    )
    model.eval()
    model.to(device)
    print("  ✓ Consistency model loaded")
    return model


def load_diffusion_model(
    ckpt_path: str,
    device: torch.device,
    n_inference_steps: int = 2000,
    use_ddim: bool = False,
    posterior_variance_type: str = "fixed_large",
    enable_legacy_reschedule: bool = False,
):
    """Load the SR3-style diffusion model from a .pth checkpoint via its JSON config."""
    print(f"\nLoading diffusion model from:\n  {ckpt_path}")

    import json
    from src.diffusion_model import create_model

    json_path = Path(__file__).resolve().parent / 'config' / 'sr_sr3_wave_inference.json'
    with open(json_path, "r") as f:
        config_dict = json.load(f)

    # Always load with the original schedule the model was trained with (2000 steps)
    # Changing n_timestep here causes buffer shape mismatches with the checkpoint
    config_dict['local_rank'] = 0
    config_dict['distributed'] = False
    config_dict['phase'] = 'val'
    config_dict['path']['resume_state'] = ckpt_path.replace('_gen.pth', '')

    model = create_model(config_dict)
    model.load_network()

    posterior_variance_type = posterior_variance_type.lower().strip()
    if posterior_variance_type not in {"fixed_large", "fixed_small"}:
        raise ValueError(
            f"Unsupported posterior_variance_type={posterior_variance_type}. "
            "Use 'fixed_large' or 'fixed_small'."
        )

    # Set before schedule creation so buffers are built with the requested mode.
    netG_for_mode = model.netG.module if isinstance(model.netG, torch.nn.DataParallel) else model.netG
    netG_for_mode.posterior_variance_type = posterior_variance_type

    model.set_new_noise_schedule(
    config_dict['model']['beta_schedule']['val'], schedule_phase='val')

    # Access the diffusion module (unwrap DataParallel if needed).
    netG = model.netG.module if isinstance(model.netG, torch.nn.DataParallel) else model.netG
    print(f"  Diffusion posterior variance mode: {netG.posterior_variance_type}")

    # Keep the training-consistent schedule for DDIM and reduce steps through
    # ddim_steps only. For DDPM, use timestep respacing derived from the
    # training schedule (not a freshly regenerated linear schedule).
    if n_inference_steps != 2000:
        if use_ddim:
            print(
                f"  DDIM mode: keeping 2000-step trained schedule; "
                f"sampling with ddim_steps={n_inference_steps}."
            )
        else:
            base_alphas_cumprod = netG.alphas_cumprod.detach().cpu().numpy().astype(np.float64)
            base_steps = base_alphas_cumprod.shape[0]
            target_steps = int(n_inference_steps)

            if target_steps <= 0:
                raise ValueError(f"n_inference_steps must be > 0, got {target_steps}.")

            if target_steps > base_steps:
                print(
                    f"  Requested {target_steps} DDPM steps exceeds base schedule ({base_steps}). "
                    f"Using {base_steps} steps instead."
                )
                target_steps = base_steps

            if target_steps < base_steps:
                # Uniformly choose indices from the trained trajectory and build
                # equivalent effective betas between retained alpha_bar points.
                selected = np.linspace(0, base_steps - 1, target_steps, dtype=int)
                selected = np.unique(selected)

                respaced_betas = []
                prev_alpha_bar = 1.0
                for idx in selected:
                    alpha_bar = float(base_alphas_cumprod[idx])
                    beta = 1.0 - (alpha_bar / prev_alpha_bar)
                    respaced_betas.append(min(max(beta, 1e-12), 0.999))
                    prev_alpha_bar = alpha_bar

                model.schedule_phase = None
                model.set_new_noise_schedule({'betas': respaced_betas}, schedule_phase='val')
                print(
                    f"  DDPM mode: respaced trained schedule {base_steps} -> "
                    f"{len(respaced_betas)} steps (alpha-trajectory preserving)."
                )

    # Optional legacy behavior for side-by-side experimentation.
    # This intentionally overrides the new default logic above.
    if enable_legacy_reschedule and n_inference_steps != 2000:
        print(f"  [legacy] Resampling noise schedule: 2000 -> {n_inference_steps} steps")
        new_schedule = {
            "schedule": "linear",
            "n_timestep": n_inference_steps,
            "linear_start": 1e-6,
            "linear_end":   1e-2,
        }
        model.schedule_phase = None  # reset guard so the update runs
        model.set_new_noise_schedule(new_schedule, schedule_phase='val')

    # Legacy schedule-resampling block kept for reference (disabled by default):
    # if n_inference_steps != 2000:
    #     print(f"  Resampling noise schedule: 2000 -> {n_inference_steps} steps")
    #     new_schedule = {
    #         "schedule": "linear",
    #         "n_timestep": n_inference_steps,
    #         "linear_start": 1e-6,
    #         "linear_end":   1e-2,
    #     }
    #     model.schedule_phase = None  # reset the guard so the update runs
    #     model.set_new_noise_schedule(new_schedule, schedule_phase='val')

    model.netG.eval()
    model.netG.to(device)
    print(f"  ✓ Diffusion model loaded ({n_inference_steps} inference steps)")
    return model


# ══════════════════════════════════════════════════════════════════════════════
# Per-image metrics
# ══════════════════════════════════════════════════════════════════════════════
def compute_metrics(sr: torch.Tensor, hr: torch.Tensor, label: Optional[str] = None) -> dict:
    """
    sr, hr: (B, C, H, W) float32 tensors, both in [0,1].
    All metrics computed in float — no uint8 conversion, no clamping.
    """
    sr = sr.float().cpu()
    hr = hr.float().cpu()

    # PSNR and SSIM in float [0,1] — skip uint8 conversion entirely
    # calculate_psnr checks dtype and rescales if not uint8, so pass as float
    # but that rescales independently — instead compute directly:
    mse = F.mse_loss(sr, hr).item()
    psnr = 10 * np.log10(1.0 / mse) if mse > 0 else float('inf')

    sr_np = sr.squeeze(0).numpy()   # (C, H, W)
    hr_np = hr.squeeze(0).numpy()

    # SSIM expects (B, H, W, C) with values in a consistent range
    sr_np_hwc = np.transpose(sr_np[np.newaxis], (0, 2, 3, 1))  # (1, H, W, C)
    hr_np_hwc = np.transpose(hr_np[np.newaxis], (0, 2, 3, 1))
    ssim_value = calculate_ssim(
        (sr_np_hwc * 255).round().astype(np.uint8),
        (hr_np_hwc * 255).round().astype(np.uint8)
    )[0]

    mae = F.l1_loss(sr, hr).item()

    # CLIP expects float tensors — [0,1] is fine
    clip_score = calculate_clipscore(sr.squeeze(0), hr.squeeze(0), 'clip-ViT-B/16')

    # LPIPS does its own per-channel normalisation internally — pass [0,1]
    lpips_val = calculate_lpips(sr.squeeze(0), hr.squeeze(0), lpips_model="vgg")

    prefix = f"[{label}] " if label else ""
    print(
        f"{prefix}psnr: {psnr} ssim: {ssim_value} "
        f"mae: {mae} clip: {clip_score} lpips: {lpips_val}"
    )

    return {"psnr": psnr, "ssim": ssim_value, "mae": mae, "clip": clip_score, "lpips": lpips_val}


# ══════════════════════════════════════════════════════════════════════════════
# Inference helpers
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def run_consistency_inference(model: Consistency, lr: torch.Tensor, n_steps: int = 1, use_ema: bool = True):
    """
    Run a single forward pass through the consistency model.

    Args:
        model:   Loaded ConsistencyWavelet instance.
        lr:      LR batch (B, C, H, W) on the correct device.
        n_steps: Number of consistency sampling steps (1 = single-step fast mode).
        use_ema: Whether to use EMA weights for inference.

    Returns:
        sr: Super-resolved tensor (B, C, H, W) on CPU.
        elapsed_s: Wall-clock time in seconds.
    """
    _, _, h, w, = lr.shape

    print("consistency in:", lr.shape)

    # print("shape in pipeline: ", h, w)
    t0 = time.perf_counter()
    sr, _ = model.sample_conditional(
        conditioning=lr,
        x_image_size=h,
        y_image_size=w,
        steps=n_steps,
        use_ema=use_ema,
    )

    print("SR range consistency: ", sr.min(), sr.max())
    
    # sr = model.adaptive_instance_normalization(sr, lr)
    elapsed = time.perf_counter() - t0

    return sr.cpu(), elapsed


@torch.no_grad()
def run_consistency_ensemble(model, lr, n_steps=1, n_runs=8, use_ema=True):
    _, _, h, w = lr.shape
    dwt_outputs = []

    model_fn = model.model_ema if use_ema else model.model
    dwsr_fn  = model.dwsr_ema  if use_ema else model.dwsr

    t0 = time.perf_counter()
    for i in range(n_runs):
        torch.manual_seed(i)

        conditioning_dwt = model.apply_dwt(lr)
        if use_ema:
            x_sr = model.dwsr_ema(conditioning_dwt)
        else:
            x_sr = model.dwsr(conditioning_dwt)

        model_time = torch.tensor([[model.time_max]], device=model.device)
        noise_shape = (lr.shape[0], 16,
                       conditioning_dwt.shape[2],
                       conditioning_dwt.shape[3])
        noise        = torch.randn(noise_shape, device=model.device)
        noisy_residual = model.image_time_product(noise, model_time.squeeze(-1))
        output_dwt   = noisy_residual + x_sr

        # Initial denoising step
        output_dwt = model._forward_conditional(
            model_fn, dwsr_fn, output_dwt, model_time, conditioning_dwt
        )

        # Multi-step refinement loop — this is what n_steps controls
        if n_steps > 1:
            _timesteps = torch.linspace(
                model.bins_max - 1, 0, n_steps, device=model.device
            ).long()
            times_seq = model.timesteps_to_times(_timesteps, model.bins_max)

            for step_time in times_seq[1:]:
                step_time_batch = step_time.expand(lr.shape[0], 1)
                noise      = torch.randn_like(output_dwt)
                re_noised  = output_dwt + model.image_time_product(
                    noise, step_time_batch.squeeze(-1)
                )
                output_dwt = model._forward_conditional(
                    model_fn, dwsr_fn, re_noised, step_time_batch, conditioning_dwt
                )

        dwt_outputs.append(output_dwt)

    # Average in wavelet domain, then convert once to image space
    mean_dwt = torch.stack(dwt_outputs).mean(dim=0)
    sr = model.apply_idwt(mean_dwt, h, w)
    # sr = model.adaptive_instance_normalization(sr, lr)
    elapsed = time.perf_counter() - t0
    return sr.cpu(), elapsed


@torch.no_grad()
def run_consistency_uncertainty(
    model,
    lr,
    n_steps=1,
    n_runs=8,
    use_ema=True,
    uncertainty_metric="std",
    uncertainty_space="image",
    seed_offset=0,
    eps=1e-8,
):
    """
    Estimate predictive uncertainty from multiple seeded runs and return
    4-channel confidence heatmaps.

    Args:
        uncertainty_metric: "std" or "var".
        uncertainty_space: "image" computes stats after IDWT.
                           "wavelet" computes stats in DWT space then IDWT.

    Returns:
        confidence: (B, 4, H, W) tensor in [0, 1], brighter = higher confidence.
        elapsed_s: wall-clock runtime in seconds.
    """
    uncertainty_metric = uncertainty_metric.lower()
    uncertainty_space = uncertainty_space.lower()
    if uncertainty_metric not in {"std", "var"}:
        raise ValueError(f"Unsupported uncertainty metric: {uncertainty_metric}. Use 'std' or 'var'.")
    if uncertainty_space not in {"image", "wavelet"}:
        raise ValueError(f"Unsupported uncertainty space: {uncertainty_space}. Use 'image' or 'wavelet'.")

    _, _, h, w = lr.shape
    dwt_outputs = []
    image_outputs = []

    model_fn = model.model_ema if use_ema else model.model
    dwsr_fn = model.dwsr_ema if use_ema else model.dwsr

    t0 = time.perf_counter()
    for i in range(n_runs):
        generator = torch.Generator(device=model.device)
        generator.manual_seed(seed_offset + i)

        conditioning_dwt = model.apply_dwt(lr)
        x_sr = dwsr_fn(conditioning_dwt)

        model_time = torch.tensor([[model.time_max]], device=model.device)
        noise_shape = (lr.shape[0], 16, conditioning_dwt.shape[2], conditioning_dwt.shape[3])
        noise = torch.randn(noise_shape, device=model.device, generator=generator)
        noisy_residual = model.image_time_product(noise, model_time.squeeze(-1))
        output_dwt = noisy_residual + x_sr

        output_dwt = model._forward_conditional(
            model_fn, dwsr_fn, output_dwt, model_time, conditioning_dwt
        )

        if n_steps > 1:
            _timesteps = torch.linspace(model.bins_max - 1, 0, n_steps, device=model.device).long()
            times_seq = model.timesteps_to_times(_timesteps, model.bins_max)

            for step_time in times_seq[1:]:
                step_time_batch = step_time.expand(lr.shape[0], 1)
                step_noise = torch.randn(output_dwt.shape, device=model.device, generator=generator)
                re_noised = output_dwt + model.image_time_product(
                    step_noise, step_time_batch.squeeze(-1)
                )
                output_dwt = model._forward_conditional(
                    model_fn, dwsr_fn, re_noised, step_time_batch, conditioning_dwt
                )

        dwt_outputs.append(output_dwt)

        if uncertainty_space == "image":
            image_outputs.append(model.apply_idwt(output_dwt, h, w))

    if uncertainty_space == "image":
        stack = torch.stack(image_outputs, dim=0)
    else:
        stack = torch.stack(dwt_outputs, dim=0)

    if uncertainty_metric == "std":
        uncertainty_map = stack.std(dim=0, unbiased=False)
    else:
        uncertainty_map = stack.var(dim=0, unbiased=False)

    if uncertainty_space == "wavelet":
        uncertainty_map = model.apply_idwt(uncertainty_map, h, w)

    confidence = uncertainty_to_confidence(uncertainty_map, eps=eps)
    elapsed = time.perf_counter() - t0
    return confidence.cpu(), elapsed

@torch.no_grad()
def run_diffusion_inference(model, lr: torch.Tensor, n_steps: int = 100, use_ddim: bool = False, ddim_eta: float = 0.0):
    """Run inference using the SR3-style DDPM wrapper."""
    print("diffusion in: ", lr.shape)
    t0 = time.perf_counter()

    # To make it the same as in the diffusion training:
    lr_norm = lr * 2 - 1

    model.feed_data({'SR': lr_norm, 'HR': torch.zeros_like(lr)})

    if use_ddim:
        model.test_ddim(ddim_steps=n_steps, eta=ddim_eta)
    else:
        model.test(continous=False)

    sr = model.get_current_visuals()['SR']  # already on CPU as float
    # lr = (model.get_current_visuals()['LR'] - 1)/2
    # lr = (lr - 1) / 2
    print("Diffusion range", sr.min(), sr.max())

    # sr = adaptive_instance_normalization(sr, lr.detach().cpu())

    elapsed = time.perf_counter() - t0

    return sr, elapsed


@torch.no_grad()
def run_diffusion_ensemble_inference(
    model,
    lr: torch.Tensor,
    n_steps: int = 100,
    n_runs: int = 8,
    use_ddim: bool = False,
    ddim_eta: float = 0.0,
    seed_offset: int = 0,
    ensemble_space: str = "image",
):
    """Run multiple seeded diffusion samples and average in image or wavelet space."""
    print("diffusion ensemble in: ", lr.shape)
    t0 = time.perf_counter()

    ensemble_space = ensemble_space.lower()
    if ensemble_space not in {"image", "wavelet"}:
        raise ValueError(f"Unsupported ensemble space: {ensemble_space}. Use 'image' or 'wavelet'.")

    lr_norm = lr * 2 - 1
    image_outputs = []
    wavelet_outputs = []
    netG = model.netG.module if isinstance(model.netG, torch.nn.DataParallel) else model.netG
    out_h, out_w = lr.shape[2], lr.shape[3]

    for i in range(n_runs):
        seed = seed_offset + i
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        model.feed_data({'SR': lr_norm, 'HR': torch.zeros_like(lr)})

        if use_ddim:
            model.test_ddim(ddim_steps=n_steps, eta=ddim_eta)
        else:
            model.test(continous=False)

        sr_scaled = model.get_current_visuals()['SR'].to(lr.device)

        if ensemble_space == "image":
            image_outputs.append(((sr_scaled + 1) / 2).cpu())
        else:
            wavelet_outputs.append(netG.apply_dwt(sr_scaled).cpu())

    if ensemble_space == "image":
        mean_sr = torch.stack(image_outputs, dim=0).mean(dim=0)
    else:
        mean_dwt = torch.stack(wavelet_outputs, dim=0).mean(dim=0).to(lr.device)
        mean_sr_scaled = netG.apply_idwt(mean_dwt, out_h, out_w)
        mean_sr = ((mean_sr_scaled + 1) / 2).cpu()

    print("Diffusion ensemble range", mean_sr.min(), mean_sr.max())

    elapsed = time.perf_counter() - t0
    return mean_sr, elapsed


@torch.no_grad()
def run_diffusion_uncertainty(
    model,
    lr: torch.Tensor,
    n_steps: int = 100,
    n_runs: int = 8,
    use_ddim: bool = False,
    ddim_eta: float = 0.0,
    uncertainty_metric: str = "std",
    uncertainty_space: str = "image",
    seed_offset: int = 0,
    eps: float = 1e-8,
):
    """Estimate diffusion/DDIM predictive uncertainty from seeded runs."""
    uncertainty_metric = uncertainty_metric.lower()
    uncertainty_space = uncertainty_space.lower()
    if uncertainty_metric not in {"std", "var"}:
        raise ValueError(f"Unsupported uncertainty metric: {uncertainty_metric}. Use 'std' or 'var'.")
    if uncertainty_space not in {"image", "wavelet"}:
        raise ValueError(f"Unsupported uncertainty space: {uncertainty_space}. Use 'image' or 'wavelet'.")

    print("diffusion uncertainty in:", lr.shape)
    t0 = time.perf_counter()

    lr_norm = lr * 2 - 1
    image_outputs = []
    wavelet_outputs = []
    netG = model.netG.module if isinstance(model.netG, torch.nn.DataParallel) else model.netG
    out_h, out_w = lr.shape[2], lr.shape[3]

    for i in range(n_runs):
        seed = seed_offset + i
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        model.feed_data({'SR': lr_norm, 'HR': torch.zeros_like(lr)})

        if use_ddim:
            model.test_ddim(ddim_steps=n_steps, eta=ddim_eta)
        else:
            model.test(continous=False)

        sr_scaled = model.get_current_visuals()['SR'].to(lr.device)

        if uncertainty_space == "image":
            image_outputs.append(((sr_scaled + 1) / 2).cpu())
        else:
            wavelet_outputs.append(netG.apply_dwt(sr_scaled).cpu())

    if uncertainty_space == "image":
        stack = torch.stack(image_outputs, dim=0)
    else:
        stack = torch.stack(wavelet_outputs, dim=0)

    if uncertainty_metric == "std":
        uncertainty_map = stack.std(dim=0, unbiased=False)
    else:
        uncertainty_map = stack.var(dim=0, unbiased=False)

    if uncertainty_space == "wavelet":
        uncertainty_map = netG.apply_idwt(uncertainty_map.to(lr.device), out_h, out_w).cpu()

    confidence = uncertainty_to_confidence(uncertainty_map, eps=eps)
    elapsed = time.perf_counter() - t0
    return confidence.cpu(), elapsed


@torch.no_grad()
def run_dwsr_inference(cons_model: Consistency, lr: torch.Tensor) -> tuple:
    """
    Run only the DWSR auxiliary network — no diffusion or consistency sampling.
    This is the deterministic wavelet SR baseline.

    Args:
        cons_model: Loaded ConsistencyWavelet instance (used to access dwsr_ema
                    and the DWT/IDWT transforms).
        lr:         LR batch (B, C, H, W) in [0, 1], on the correct device.

    Returns:
        sr:         Super-resolved tensor (B, C, H, W) on CPU, in [0, 1].
        elapsed_s:  Wall-clock time in seconds.
    """
    _, _, h, w = lr.shape
    t0 = time.perf_counter()

    # DWT → DWSR → IDWT, exactly as done inside sample_conditional
    conditioning_dwt = cons_model.apply_dwt(lr)
    x_sr_dwt = cons_model.dwsr_ema(conditioning_dwt)
    sr = cons_model.apply_idwt(x_sr_dwt, h, w)

    elapsed = time.perf_counter() - t0

    return sr.cpu(), elapsed

@torch.no_grad()
def run_dwsr_diffusion_inference(diff_model, lr: torch.Tensor) -> tuple:
    """
    Run only the DWSR auxiliary network from the diffusion model.

    The diffusion model's DWSR operates on wavelet-domain LR input in [-1,1],
    so lr is scaled accordingly before entering the network.

    Args:
        diff_model: Loaded DDPM instance.
        lr:         LR batch (B, C, H, W) in [0, 1], on the correct device.

    Returns:
        sr:        Super-resolved tensor (B, C, H, W) on CPU, in [0, 1].
        elapsed_s: Wall-clock time in seconds.
    """
    t0 = time.perf_counter()

    # Diffusion model was trained on [-1,1] input
    lr_scaled = lr * 2 - 1

    # Access the GaussianDiffusion module (unwrap DataParallel if needed)
    netG = diff_model.netG.module if isinstance(diff_model.netG, torch.nn.DataParallel) \
           else diff_model.netG

    lr_dwt = netG.apply_dwt(lr_scaled)
    x_sr_dwt = netG.dwsr(lr_dwt)
    sr_scaled = netG.apply_idwt(x_sr_dwt, lr.shape[2], lr.shape[3])

    # Shift back from [-1,1] to [0,1]
    sr = ((sr_scaled + 1) / 2).cpu()

    elapsed = time.perf_counter() - t0
    return sr, elapsed

# ══════════════════════════════════════════════════════════════════════════════
# Save helpers
# ══════════════════════════════════════════════════════════════════════════════

def save_tif(tensor: torch.Tensor, path: str):
    """Save a (C, H, W) or (B, C, H, W) tensor as a .tif file."""
    arr = tensor.squeeze(0).numpy()  # → (C, H, W)
    arr = arr.transpose(1, 2, 0)     # → (H, W, C)
    if arr.shape[-1] == 1:
        arr = arr[:, :, 0]           # → (H, W) for single-channel
    tifffile.imwrite(path, arr.astype(np.float32))


def build_run_tag(model_type: str, n_steps: int, ensemble_size: int | None = None) -> str:
    """Build a stable suffix for metrics files and image folders."""
    parts = [model_type.lower(), str(n_steps)]
    if ensemble_size is not None:
        parts.append(str(ensemble_size))
    return "_".join(parts)


def build_output_name(prefix: str, model_type: str, n_steps: int, ensemble_size: int | None = None, ext: str = "") -> str:
    tag = build_run_tag(model_type, n_steps, ensemble_size)
    return f"{prefix}_{tag}{ext}"


# ══════════════════════════════════════════════════════════════════════════════
# Main evaluation loop
# ══════════════════════════════════════════════════════════════════════════════

def evaluate(args):

    print("batch size: ", args.batch_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nRunning on: {device}")

    diffusion_model_type = "ddim" if args.use_ddim else "diffusion"
    consistency_tag = build_run_tag("consistency", args.n_consistency_steps)
    consistency_ensemble_tag = (
        build_run_tag("consistency", args.n_consistency_steps, args.ensemble_runs)
        if args.consistency_ensemble else None
    )
    diffusion_tag = build_run_tag(diffusion_model_type, args.n_diffusion_steps)
    diffusion_ensemble_tag = (
        build_run_tag(diffusion_model_type, args.n_diffusion_steps, args.diffusion_ensemble_runs)
        if args.diffusion_ensemble else None
    )
    diffusion_uncertainty_tag = (
        build_run_tag(f"{diffusion_model_type}_uncertainty", args.n_diffusion_steps, args.diffusion_uncertainty_runs)
        if args.diffusion_uncertainty else None
    )
    dwsr_consistency_tag = build_run_tag("dwsr_consistency", args.n_consistency_steps)
    dwsr_diffusion_tag = build_run_tag(f"dwsr_{diffusion_model_type}", args.n_diffusion_steps)
    uncertainty_tag = build_run_tag("consistency_uncertainty", args.n_consistency_steps)

    # ── Output directories ─────────────────────────────────────────────────────
    out_root = Path(args.output_dir)
    cons_out = out_root / consistency_tag
    cons_ens_out = out_root / consistency_ensemble_tag if consistency_ensemble_tag else None
    unc_out = out_root / uncertainty_tag
    diff_out = out_root / diffusion_tag
    diff_ens_out = out_root / diffusion_ensemble_tag if diffusion_ensemble_tag else None
    diff_unc_out = out_root / diffusion_uncertainty_tag if diffusion_uncertainty_tag else None
    dwsr_out = out_root / dwsr_consistency_tag
    dwsr_diff_out = out_root / dwsr_diffusion_tag
    if args.save_images:
        dwsr_out.mkdir(parents=True, exist_ok=True)
        dwsr_diff_out.mkdir(parents=True, exist_ok=True)
        cons_out.mkdir(parents=True, exist_ok=True)
        if diff_ens_out is not None:
            diff_ens_out.mkdir(parents=True, exist_ok=True)
        if diff_unc_out is not None:
            diff_unc_out.mkdir(parents=True, exist_ok=True)
        if args.consistency_uncertainty:
            unc_out.mkdir(parents=True, exist_ok=True)
        if args.diffusion_ckpt:
            diff_out.mkdir(parents=True, exist_ok=True)
        if cons_ens_out is not None:
            cons_ens_out.mkdir(parents=True, exist_ok=True)
    input_dirs = {Path(args.lr_path).resolve(), Path(args.hr_path).resolve()}
    for out_dir in [cons_out, cons_ens_out, diff_out, diff_ens_out, diff_unc_out, dwsr_out, dwsr_diff_out]:
        if out_dir is None:
            continue
        if out_dir.resolve() in input_dirs:
            raise ValueError(f"Output directory {out_dir} points to an input directory — aborting to protect your data.")

    # ── Dataset ────────────────────────────────────────────────────────────────
    loader = get_test_dataloader(
        lr_path=args.lr_path,
        hr_path=args.hr_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pad=0,  # Add padding if needed via config
        crop_size=(192, 192) if hasattr(args, 'crop_size') else None,
        )

    # ── Models ─────────────────────────────────────────────────────────────────
    if args.consistency_ckpt:
        cons_model = load_consistency_model(args.consistency_ckpt, device)
        # diff_model = None
    if args.diffusion_ckpt:
        diff_model = load_diffusion_model(
            args.diffusion_ckpt,
            device,
            n_inference_steps=args.n_diffusion_steps,
            use_ddim=args.use_ddim,
            posterior_variance_type=getattr(args, "diffusion_posterior_variance", "fixed_large"),
            enable_legacy_reschedule=getattr(args, "enable_legacy_schedule_resample", False),
        )

    # ── Per-image result storage ───────────────────────────────────────────────
    records_cons = []
    records_diff = []
    records_dwsr = []
    records_dwsr_diff = []
    records_cons_ens = []
    records_diff_ens = []
    df_cons = None
    df_diff = None
    df_diff_ens = None


    # ── Evaluation loop ────────────────────────────────────────────────────────
    for batch in tqdm(loader, desc="Evaluating"):
        lr        = batch["lr"].to(device)
        hr        = batch["hr"].to(device)
        # filenames = batch["filename"]
        filenames = [Path(f).name for f in batch["filename"]]

        metric_label_suffix = filenames[0] if len(filenames) == 1 else f"batch:{filenames[0]}"

        m_inputs = compute_metrics(lr.cpu(), hr.cpu(), label=f"inputs:{metric_label_suffix}")

        # -- DWSR baseline --
        # Consistency DWSR baseline
        if args.dwsr_consistency:
            sr_dwsr, t_dwsr = run_dwsr_inference(cons_model, lr.to(device))
            m_dwsr = compute_metrics(sr_dwsr.clamp(0, 1), hr.cpu(), label=f"dwsr:{metric_label_suffix}")

            for i, fname in enumerate(filenames):
                records_dwsr.append({
                    "filename": fname,
                    "psnr":   m_dwsr["psnr"],
                    "ssim":   m_dwsr["ssim"],
                    "mae":    m_dwsr["mae"],
                    "clip":   m_dwsr["clip"],
                    "lpips":  m_dwsr["lpips"],
                    "time_s": t_dwsr,
                    "psnr_inputs": m_inputs["psnr"],
                    "ssim_inputs": m_inputs["ssim"],
                    "mae_inputs":  m_inputs["mae"],
                    "clip_inputs": m_inputs["clip"],
                    "lpips_inputs": m_inputs["lpips"],
                })
            if args.save_images:
                save_tif(sr_dwsr[i], str(dwsr_out / fname))



        # Diffusion DWSR baseline
        if args.dwsr_diffusion:
            sr_dwsr_diff, t_dwsr_diff = run_dwsr_diffusion_inference(diff_model, lr.to(device))
            m_dwsr_diff = compute_metrics(
                sr_dwsr_diff.clamp(0, 1),
                hr.cpu(),
                label=f"dwsr_diffusion:{metric_label_suffix}"
            )

            for i, fname in enumerate(filenames):
                records_dwsr_diff.append({
                    "filename": fname,
                    "psnr":   m_dwsr_diff["psnr"],
                    "ssim":   m_dwsr_diff["ssim"],
                    "mae":    m_dwsr_diff["mae"],
                    "clip":   m_dwsr_diff["clip"],
                    "lpips":  m_dwsr_diff["lpips"],
                    "time_s": t_dwsr_diff,
                    "psnr_inputs": m_inputs["psnr"],
                    "ssim_inputs": m_inputs["ssim"],
                    "mae_inputs":  m_inputs["mae"],
                    "clip_inputs": m_inputs["clip"],
                    "lpips_inputs": m_inputs["lpips"],
                })
                if args.save_images:
                    save_tif(sr_dwsr_diff[i], str(dwsr_diff_out / fname))



        # -- Consistency model --
        if args.consistency_ckpt:
            sr_cons, t_cons = run_consistency_inference(
                cons_model,
                lr,
                n_steps=args.n_consistency_steps,
                use_ema=args.use_ema,
            )

        # -- Consistency ensemble --
        # sr_cons, t_cons = run_consistency_ensemble(
        #     cons_model,
        #     lr,
        #     n_steps=args.n_consistency_steps,
        #     n_runs=20,
        #     use_ema=args.use_ema,
        # )

            m_cons = compute_metrics(
                sr_cons.clamp(0,1),
                hr.cpu(),
                label=f"consistency:{metric_label_suffix}"
            )
            # m_inputs = compute_metrics(lr.cpu(), hr.cpu())

            for i, fname in enumerate(filenames):
                records_cons.append(
                    {
                        "filename": fname,
                        "psnr": m_cons["psnr"],
                        "ssim": m_cons["ssim"],
                        "mae":  m_cons["mae"],
                        "clip": m_cons["clip"],
                        "lpips": m_cons["lpips"],
                        "time_s": t_cons / len(filenames),
                        "psnr_inputs": m_inputs["psnr"],
                        "ssim_inputs": m_inputs["ssim"],
                        "mae_inputs":  m_inputs["mae"],
                        "clip_inputs": m_inputs["clip"],
                        "lpips_inputs": m_inputs["lpips"],

                    }
                )
                if args.save_images:
                    save_tif(sr_cons[i], str(cons_out / fname))



        # -- Consistency ensemble --
        if args.consistency_ckpt and args.consistency_ensemble:
            sr_cons_ens, t_cons_ens = run_consistency_ensemble(
                cons_model,
                lr,
                n_steps=args.n_consistency_steps,
                n_runs=args.ensemble_runs,
                use_ema=args.use_ema,
                )

            m_cons_ens = compute_metrics(
                sr_cons_ens.clamp(0,1),
                hr.cpu(),
                label=f"consistency_ensemble:{metric_label_suffix}"
            )
            # m_inputs = compute_metrics(lr.cpu(), hr.cpu())

            for i, fname in enumerate(filenames):
                records_cons_ens.append(
                    {
                        "filename": fname,
                        "psnr": m_cons_ens["psnr"],
                        "ssim": m_cons_ens["ssim"],
                        "mae":  m_cons_ens["mae"],
                        "clip": m_cons_ens["clip"],
                        "lpips": m_cons_ens["lpips"],
                        "time_s": t_cons_ens / len(filenames),
                        "psnr_inputs": m_inputs["psnr"],
                        "ssim_inputs": m_inputs["ssim"],
                        "mae_inputs":  m_inputs["mae"],
                        "clip_inputs": m_inputs["clip"],
                        "lpips_inputs": m_inputs["lpips"],

                    }
                )
                if args.save_images:
                    save_tif(sr_cons_ens[i], str(cons_ens_out / fname))



        # -- Consistency uncertainty heatmaps --
        if args.consistency_ckpt and args.consistency_uncertainty:
            sr_unc, t_unc = run_consistency_uncertainty(
                cons_model,
                lr,
                n_steps=args.n_consistency_steps,
                n_runs=args.uncertainty_runs,
                use_ema=args.use_ema,
                uncertainty_metric=args.uncertainty_metric,
                uncertainty_space=args.uncertainty_space,
                seed_offset=args.uncertainty_seed_offset,
                eps=args.uncertainty_eps,
            )

            print(
                "Saved uncertainty heatmaps "
                f"({args.uncertainty_metric}, {args.uncertainty_space}) "
                f"for {len(filenames)} images in {t_unc:.3f}s"
            )

            for i, fname in enumerate(filenames):
                if args.save_images:
                    save_tif(sr_unc[i], str(unc_out / fname))


        # -- Diffusion model (optional) --
        # if diff_model is not None:
        if args.diffusion_ckpt:
            sr_diff, t_diff = run_diffusion_inference(
                diff_model,
                lr,
                n_steps=args.n_diffusion_steps,
                use_ddim=args.use_ddim,
                ddim_eta=args.ddim_eta,
            )

            m_diff = compute_metrics(
                (sr_diff + 1) / 2,
                hr.cpu(),
                label=f"diffusion:{metric_label_suffix}"
            )

            for i, fname in enumerate(filenames):
                records_diff.append(
                    {
                        "filename": fname,
                        "psnr": m_diff["psnr"],
                        "ssim": m_diff["ssim"],
                        "mae":  m_diff["mae"],
                        "clip": m_diff["clip"],
                        "lpips": m_diff["lpips"],
                        "time_s": t_diff / len(filenames),
                        "psnr_inputs": m_inputs["psnr"],
                        "ssim_inputs": m_inputs["ssim"],
                        "mae_inputs":  m_inputs["mae"],
                        "clip_inputs": m_inputs["clip"],
                        "lpips_inputs": m_inputs["lpips"],
                    }
                )
                if args.save_images:
                    save_tif(sr_diff[i], str(diff_out / fname))
                    print("Diffusion saved: ", str(diff_out / fname))

        # -- Diffusion/DDIM ensemble --
        if args.diffusion_ckpt and args.diffusion_ensemble:
            sr_diff_ens, t_diff_ens = run_diffusion_ensemble_inference(
                diff_model,
                lr,
                n_steps=args.n_diffusion_steps,
                n_runs=args.diffusion_ensemble_runs,
                use_ddim=args.use_ddim,
                ddim_eta=args.ddim_eta,
                seed_offset=args.diffusion_ensemble_seed_offset,
                ensemble_space=args.diffusion_ensemble_space,
            )

            m_diff_ens = compute_metrics(
                sr_diff_ens,
                hr.cpu(),
                label=f"{diffusion_model_type}_ensemble:{metric_label_suffix}"
            )

            for i, fname in enumerate(filenames):
                records_diff_ens.append(
                    {
                        "filename": fname,
                        "psnr": m_diff_ens["psnr"],
                        "ssim": m_diff_ens["ssim"],
                        "mae":  m_diff_ens["mae"],
                        "clip": m_diff_ens["clip"],
                        "lpips": m_diff_ens["lpips"],
                        "time_s": t_diff_ens / len(filenames),
                        "psnr_inputs": m_inputs["psnr"],
                        "ssim_inputs": m_inputs["ssim"],
                        "mae_inputs":  m_inputs["mae"],
                        "clip_inputs": m_inputs["clip"],
                        "lpips_inputs": m_inputs["lpips"],
                    }
                )
                if args.save_images:
                    save_tif(sr_diff_ens[i], str(diff_ens_out / fname))
                    print("Diffusion ensemble saved: ", str(diff_ens_out / fname))

        # -- Diffusion/DDIM uncertainty heatmaps --
        if args.diffusion_ckpt and args.diffusion_uncertainty:
            sr_diff_unc, t_diff_unc = run_diffusion_uncertainty(
                diff_model,
                lr,
                n_steps=args.n_diffusion_steps,
                n_runs=args.diffusion_uncertainty_runs,
                use_ddim=args.use_ddim,
                ddim_eta=args.ddim_eta,
                uncertainty_metric=args.diffusion_uncertainty_metric,
                uncertainty_space=args.diffusion_uncertainty_space,
                seed_offset=args.diffusion_uncertainty_seed_offset,
                eps=args.diffusion_uncertainty_eps,
            )

            print(
                "Saved diffusion uncertainty heatmaps "
                f"({args.diffusion_uncertainty_metric}, {args.diffusion_uncertainty_space}) "
                f"for {len(filenames)} images in {t_diff_unc:.3f}s"
            )

            for i, fname in enumerate(filenames):
                if args.save_images:
                    save_tif(sr_diff_unc[i], str(diff_unc_out / fname))

    # ── Aggregate and print ────────────────────────────────────────────────────
    if records_dwsr:
        df_dwsr = pd.DataFrame(records_dwsr)
        print_summary("DWSR only", df_dwsr, n_steps=1)
        df_dwsr.to_csv(
            out_root / build_output_name("metrics_dwsr", "consistency", args.n_consistency_steps, ext=".csv"),
            index=False,
        )

    if records_dwsr_diff:
        df_dwsr_diff = pd.DataFrame(records_dwsr_diff)
        print_summary("DWSR only (diffusion)", df_dwsr_diff, n_steps=1)
        df_dwsr_diff.to_csv(
            out_root / build_output_name("metrics_dwsr", diffusion_model_type, args.n_diffusion_steps, ext=".csv"),
            index=False,
        )

    if records_cons:
        df_cons = pd.DataFrame(records_cons)
        print_summary("Consistency Model", df_cons, args.n_consistency_steps)
        df_cons.to_csv(
            out_root / build_output_name("metrics", "consistency", args.n_consistency_steps, ext=".csv"),
            index=False,
        )

    if records_cons_ens:
        df_cons_ens = pd.DataFrame(records_cons_ens)
        print_summary("Consistency Model ensemble", df_cons_ens, args.n_consistency_steps)
        df_cons_ens.to_csv(
            out_root / build_output_name("metrics", "consistency", args.n_consistency_steps, args.ensemble_runs, ext=".csv"),
            index=False,
        )

    if records_diff:
        df_diff = pd.DataFrame(records_diff)
        print_summary("DDIM Model" if args.use_ddim else "Diffusion Model", df_diff, args.n_diffusion_steps)
        df_diff.to_csv(
            out_root / build_output_name("metrics", diffusion_model_type, args.n_diffusion_steps, ext=".csv"),
            index=False,
        )

    if records_diff_ens:
        df_diff_ens = pd.DataFrame(records_diff_ens)
        print_summary(
            "DDIM Model ensemble" if args.use_ddim else "Diffusion Model ensemble",
            df_diff_ens,
            args.n_diffusion_steps,
        )
        df_diff_ens.to_csv(
            out_root / build_output_name("metrics", diffusion_model_type, args.n_diffusion_steps, args.diffusion_ensemble_runs, ext=".csv"),
            index=False,
        )

        # Side-by-side comparison summary
    if df_cons is not None and df_diff is not None:
        print_comparison(df_cons, df_diff)

    return df_cons, df_diff if df_cons is not None and df_diff is not None else None


# ══════════════════════════════════════════════════════════════════════════════
# Reporting helpers
# ══════════════════════════════════════════════════════════════════════════════

def print_summary(name: str, df: pd.DataFrame, n_steps: int):
    print(f"\n{'─'*55}")
    print(f"  {name}  ({n_steps} steps)")
    print(f"{'─'*55}")
    for col in ["psnr", "ssim", "mae", "clip", "lpips", "time_s"]:
        print(col)
        print(f"  {col:10s}  mean={df[col].mean():.4f}   std={df[col].std():.4f}")
    print(f"{'─'*55}")


def print_comparison(df_cons: pd.DataFrame, df_diff: pd.DataFrame):
    print(f"\n{'═'*55}")
    print("  COMPARISON: Consistency vs Diffusion")
    print(f"{'═'*55}")
    header = f"  {'Metric':10s}  {'Consistency':>14s}  {'Diffusion':>14s}  {'Δ (C−D)':>10s}"
    print(header)
    print(f"{'─'*55}")
    for col in ["psnr", "ssim", "mae", "clip", "lpips", "time_s"]:
        c_mean = df_cons[col].mean()
        d_mean = df_diff[col].mean()
        delta  = c_mean - d_mean
        print(f"  {col:10s}  {c_mean:>14.4f}  {d_mean:>14.4f}  {delta:>+10.4f}")
    print(f"{'═'*55}")


if __name__ == "__main__":
    cfg = InferenceConfig()
    evaluate(cfg)