"""
uncertainty_eval.py

Standalone runner for uncertainty calibration and uncertainty-aware refinement.

This script is separate from inference_consistency.py so the regular inference
pipeline remains focused on generation, while this file handles:
- uncertainty calibration metrics,
- uncertainty ranking analysis,
- uncertainty-guided refinement,
- optional saving of analysis outputs.
"""

from pathlib import Path
from typing import Iterable
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

from inference_consistency import load_consistency_model, load_diffusion_model, save_tif
from config.configuration_inference_default import InferenceConfig
from src.data import get_test_dataloader
from src.utils.uncertainty import (
    evaluate_uncertainty_calibration,
    normalize_uncertainty_map,
    refine_prediction_with_uncertainty,
    robust_ensemble_average,
    uncertainty_to_confidence,
)


def _as_float_list(values: Iterable[float], fallback):
    if values is None:
        return list(fallback)
    return [float(v) for v in values]


def _make_unique_output_dir(base_dir: Path, run_tag: str) -> Path:
    """Create a run-specific output directory without overwriting previous runs."""
    stem = f"uncertainty_eval_{run_tag}"
    candidate = base_dir / stem
    if not candidate.exists():
        return candidate

    suffix = 2
    while True:
        candidate = base_dir / f"{stem}_{suffix}"
        if not candidate.exists():
            return candidate
        suffix += 1


def _fast_metrics(sr: torch.Tensor, hr: torch.Tensor, eps: float = 1e-12) -> dict:
    """Compute lightweight image metrics for quick comparison."""
    sr = sr.float()
    hr = hr.float()

    diff = sr - hr
    mae = float(diff.abs().mean())
    rmse = float(torch.sqrt(diff.pow(2).mean().clamp_min(eps)))
    mse = float(diff.pow(2).mean())
    psnr = float(10.0 * np.log10(1.0 / max(mse, eps)))

    mae_ch = diff.abs().mean(dim=(0, 2, 3))
    rmse_ch = torch.sqrt(diff.pow(2).mean(dim=(0, 2, 3)).clamp_min(eps))
    mse_ch = diff.pow(2).mean(dim=(0, 2, 3))
    psnr_ch = 10.0 * torch.log10(1.0 / mse_ch.clamp_min(eps))

    return {
        "mae": mae,
        "rmse": rmse,
        "psnr": psnr,
        "mae_ch": [float(v) for v in mae_ch],
        "rmse_ch": [float(v) for v in rmse_ch],
        "psnr_ch": [float(v) for v in psnr_ch],
    }


@torch.no_grad()
def run_consistency_ensemble_outputs(
    model,
    lr: torch.Tensor,
    n_steps: int = 1,
    n_runs: int = 8,
    use_ema: bool = True,
    uncertainty_metric: str = "std",
    uncertainty_space: str = "image",
    uncertainty_normalization: str = "absolute",
    seed_offset: int = 0,
    eps: float = 1e-8,
):
    """
    Generate ensemble outputs and uncertainty maps.

    uncertainty_space="image": uncertainty from image-space ensemble.
    uncertainty_space="wavelet": uncertainty from wavelet-space ensemble,
    then mapped to image space for analysis/saving.

    Returns:
        dict with image-space outputs for evaluation and optional wavelet-space
        statistics for wavelet-domain refinement.
        elapsed_s: runtime in seconds
    """
    uncertainty_metric = uncertainty_metric.lower()
    uncertainty_space = uncertainty_space.lower()
    if uncertainty_metric not in {"std", "var"}:
        raise ValueError("uncertainty_metric must be 'std' or 'var'.")
    if uncertainty_space not in {"image", "wavelet"}:
        raise ValueError("uncertainty_space must be 'image' or 'wavelet'.")
    if uncertainty_normalization.lower() not in {"absolute", "relative_mean"}:
        raise ValueError("uncertainty_normalization must be 'absolute' or 'relative_mean'.")

    supports_wavelet = all(
        hasattr(model, attr)
        for attr in ["apply_dwt", "apply_idwt", "_forward_conditional", "image_time_product"]
    )
    if uncertainty_space == "wavelet" and not supports_wavelet:
        raise ValueError(
            "Wavelet-space uncertainty requires a wavelet consistency model with "
            "apply_dwt/apply_idwt/_forward_conditional."
        )

    _, _, h, w = lr.shape
    image_samples = []
    wavelet_samples = []

    t0 = time.perf_counter()

    model_fn = model.model_ema if use_ema else model.model
    dwsr_fn = model.dwsr_ema if use_ema else model.dwsr

    for i in range(n_runs):
        generator = torch.Generator(device=model.device)
        generator.manual_seed(seed_offset + i)

        if uncertainty_space == "image":
            sr, _ = model.sample_conditional(
                conditioning=lr,
                x_image_size=h,
                y_image_size=w,
                steps=n_steps,
                generator=generator,
                use_ema=use_ema,
            )
            image_samples.append(sr)
            continue

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

        wavelet_samples.append(output_dwt)
        image_samples.append(model.apply_idwt(output_dwt, h, w))

    image_stack = torch.stack(image_samples, dim=0)
    mean_dwt = None
    uncertainty_dwt = None

    # Always compute image-space moments for calibration metrics.
    mean_pred = image_stack.mean(dim=0)
    std_pred = image_stack.std(dim=0, unbiased=False)
    var_pred = image_stack.var(dim=0, unbiased=False)

    if uncertainty_space == "wavelet":
        wavelet_stack = torch.stack(wavelet_samples, dim=0)
        mean_dwt = wavelet_stack.mean(dim=0)
        std_wavelet = wavelet_stack.std(dim=0, unbiased=False)
        var_wavelet = wavelet_stack.var(dim=0, unbiased=False)
        uncertainty_dwt = std_wavelet if uncertainty_metric == "std" else var_wavelet

    uncertainty_map = std_pred if uncertainty_metric == "std" else var_pred
    uncertainty_map = normalize_uncertainty_map(
        uncertainty_map,
        mean_prediction=mean_pred,
        uncertainty_metric=uncertainty_metric,
        mode=uncertainty_normalization,
        eps=eps,
    )

    if uncertainty_dwt is not None and mean_dwt is not None:
        uncertainty_dwt = normalize_uncertainty_map(
            uncertainty_dwt,
            mean_prediction=mean_dwt,
            uncertainty_metric=uncertainty_metric,
            mode=uncertainty_normalization,
            eps=eps,
        )

    confidence_map = uncertainty_to_confidence(uncertainty_map, eps=eps)

    elapsed = time.perf_counter() - t0
    return {
        "stack": image_stack,
        "mean": mean_pred,
        "std": std_pred,
        "var": var_pred,
        "uncertainty": uncertainty_map,
        "confidence": confidence_map,
        "mean_dwt": mean_dwt,
        "uncertainty_dwt": uncertainty_dwt,
        "elapsed_s": elapsed,
    }


@torch.no_grad()
def run_diffusion_ensemble_outputs(
    model,
    lr: torch.Tensor,
    n_steps: int = 10,
    n_runs: int = 8,
    use_ddim: bool = True,
    ddim_eta: float = 0.0,
    uncertainty_metric: str = "std",
    uncertainty_space: str = "image",
    uncertainty_normalization: str = "absolute",
    seed_offset: int = 0,
    eps: float = 1e-8,
):
    """Generate diffusion/DDIM ensemble outputs and uncertainty maps."""
    uncertainty_metric = uncertainty_metric.lower()
    uncertainty_space = uncertainty_space.lower()
    if uncertainty_metric not in {"std", "var"}:
        raise ValueError("uncertainty_metric must be 'std' or 'var'.")
    if uncertainty_space not in {"image", "wavelet"}:
        raise ValueError("uncertainty_space must be 'image' or 'wavelet'.")
    if uncertainty_normalization.lower() not in {"absolute", "relative_mean"}:
        raise ValueError("uncertainty_normalization must be 'absolute' or 'relative_mean'.")

    netG = model.netG.module if isinstance(model.netG, torch.nn.DataParallel) else model.netG
    lr_norm = lr * 2.0 - 1.0
    _, _, h, w = lr.shape

    image_samples = []
    wavelet_samples = []
    t0 = time.perf_counter()

    for i in range(n_runs):
        seed = seed_offset + i
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        model.feed_data({"SR": lr_norm, "HR": torch.zeros_like(lr)})
        if use_ddim:
            model.test_ddim(ddim_steps=n_steps, eta=ddim_eta)
        else:
            model.test(continous=False)

        sr_scaled = model.get_current_visuals()["SR"].to(lr.device)
        sr_image = ((sr_scaled + 1.0) / 2.0).clamp(0.0, 1.0)

        image_samples.append(sr_image)
        if uncertainty_space == "wavelet":
            wavelet_samples.append(netG.apply_dwt(sr_scaled))

    image_stack = torch.stack(image_samples, dim=0)
    mean_dwt = None
    uncertainty_dwt = None

    # Always compute image-space moments for calibration metrics.
    mean_pred = image_stack.mean(dim=0)
    std_pred = image_stack.std(dim=0, unbiased=False)
    var_pred = image_stack.var(dim=0, unbiased=False)

    if uncertainty_space == "wavelet":
        wavelet_stack = torch.stack(wavelet_samples, dim=0)
        mean_dwt = wavelet_stack.mean(dim=0)
        std_wavelet = wavelet_stack.std(dim=0, unbiased=False)
        var_wavelet = wavelet_stack.var(dim=0, unbiased=False)
        uncertainty_dwt = std_wavelet if uncertainty_metric == "std" else var_wavelet

    uncertainty_map = std_pred if uncertainty_metric == "std" else var_pred
    uncertainty_map = normalize_uncertainty_map(
        uncertainty_map,
        mean_prediction=mean_pred,
        uncertainty_metric=uncertainty_metric,
        mode=uncertainty_normalization,
        eps=eps,
    )

    if uncertainty_dwt is not None and mean_dwt is not None:
        # Convert wavelet-space uncertainty from [-1,1]-scaled image domain to [0,1].
        uncertainty_dwt_img = netG.apply_idwt(uncertainty_dwt, h, w)
        if uncertainty_metric == "std":
            uncertainty_dwt_img = uncertainty_dwt_img / 2.0
        else:
            uncertainty_dwt_img = uncertainty_dwt_img / 4.0

        uncertainty_dwt = normalize_uncertainty_map(
            uncertainty_dwt_img,
            mean_prediction=mean_pred,
            uncertainty_metric=uncertainty_metric,
            mode=uncertainty_normalization,
            eps=eps,
        )

    confidence_map = uncertainty_to_confidence(uncertainty_map, eps=eps)

    elapsed = time.perf_counter() - t0
    return {
        "stack": image_stack,
        "mean": mean_pred,
        "std": std_pred,
        "var": var_pred,
        "uncertainty": uncertainty_map,
        "confidence": confidence_map,
        "mean_dwt": mean_dwt,
        "uncertainty_dwt": uncertainty_dwt,
        "elapsed_s": elapsed,
    }


def evaluate_uncertainty(cfg: InferenceConfig):
    uncertainty_backend = str(getattr(cfg, "uncertainty_backend", "consistency")).lower()
    if uncertainty_backend not in {"consistency", "diffusion"}:
        raise ValueError("uncertainty_backend must be 'consistency' or 'diffusion'.")

    if uncertainty_backend == "consistency" and not cfg.consistency_ckpt:
        raise ValueError("consistency_ckpt must be set for consistency uncertainty evaluation.")
    if uncertainty_backend == "diffusion" and not cfg.diffusion_ckpt:
        raise ValueError("diffusion_ckpt must be set for diffusion uncertainty evaluation.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running uncertainty evaluation on: {device}")

    if uncertainty_backend == "consistency":
        n_steps = int(getattr(cfg, "n_consistency_steps", 1))
        n_runs = int(getattr(cfg, "uncertainty_runs", 8))
        seed_offset = int(getattr(cfg, "uncertainty_seed_offset", 0))
        uncertainty_metric = str(getattr(cfg, "uncertainty_metric", "std"))
        uncertainty_space = str(getattr(cfg, "uncertainty_space", "image"))
        use_ema = bool(getattr(cfg, "use_ema", True))
        use_ddim = False
        ddim_eta = 0.0
    else:
        n_steps = int(getattr(cfg, "n_diffusion_steps", 10))
        n_runs = int(getattr(cfg, "diffusion_uncertainty_runs", 8))
        seed_offset = int(getattr(cfg, "diffusion_uncertainty_seed_offset", 0))
        uncertainty_metric = str(getattr(cfg, "diffusion_uncertainty_metric", "std"))
        uncertainty_space = str(getattr(cfg, "diffusion_uncertainty_space", "image"))
        use_ema = False
        use_ddim = bool(getattr(cfg, "use_ddim", True))
        ddim_eta = float(getattr(cfg, "ddim_eta", 0.0))
    uncertainty_normalization = str(getattr(cfg, "uncertainty_normalization", "absolute"))
    if uncertainty_backend == "consistency":
        uncertainty_eps = float(getattr(cfg, "uncertainty_eps", 1e-8))
    else:
        uncertainty_eps = float(getattr(cfg, "diffusion_uncertainty_eps", 1e-8))

    print(
        f"Uncertainty backend={uncertainty_backend} settings: metric={uncertainty_metric}, "
        f"space={uncertainty_space}, norm={uncertainty_normalization}, runs={n_runs}"
    )

    blend_strength = float(getattr(cfg, "uncertainty_blend_strength", 1.0))
    use_confidence_normalization = bool(getattr(cfg, "use_confidence_normalization", True))
    refine_anchor = str(getattr(cfg, "uncertainty_refine_anchor", "robust")).lower()
    robust_temperature = float(getattr(cfg, "uncertainty_robust_temperature", 1.0))
    error_metric = str(getattr(cfg, "uncertainty_error_metric", "abs"))

    if refine_anchor not in {"robust", "baseline"}:
        raise ValueError("uncertainty_refine_anchor must be 'robust' or 'baseline'.")

    interval_levels = _as_float_list(
        getattr(cfg, "uncertainty_interval_levels", None),
        fallback=(0.5, 0.8, 0.9, 0.95),
    )
    remove_fractions = _as_float_list(
        getattr(cfg, "uncertainty_remove_fractions", None),
        fallback=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5),
    )

    run_tag = (
        f"backend{uncertainty_backend}_space{uncertainty_space}_metric{uncertainty_metric}"
        f"_norm{uncertainty_normalization}"
        f"_steps{n_steps}_runs{n_runs}_seed{seed_offset}"
    )
    out_root = _make_unique_output_dir(Path(cfg.output_dir), run_tag)
    out_root.mkdir(parents=True, exist_ok=True)

    mean_out = out_root / "mean_sr"
    refined_out = out_root / "refined_sr"
    robust_out = out_root / "robust_sr"
    uncertainty_out = out_root / "uncertainty_map"
    confidence_out = out_root / "confidence_map"
    baseline_out = out_root / "baseline_sr"

    save_images = bool(getattr(cfg, "save_images", True))
    if save_images:
        for p in [mean_out, refined_out, robust_out, uncertainty_out, confidence_out, baseline_out]:
            p.mkdir(parents=True, exist_ok=True)

    if uncertainty_backend == "consistency":
        model = load_consistency_model(cfg.consistency_ckpt, device)
    else:
        model = load_diffusion_model(
            cfg.diffusion_ckpt,
            device=device,
            n_inference_steps=n_steps,
            use_ddim=use_ddim,
            posterior_variance_type=str(getattr(cfg, "diffusion_posterior_variance", "fixed_large")),
            enable_legacy_reschedule=bool(getattr(cfg, "enable_legacy_schedule_resample", False)),
        )

    loader = get_test_dataloader(
        lr_path=cfg.lr_path,
        hr_path=cfg.hr_path,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        pad=0,
        crop_size=(192, 192) if hasattr(cfg, "crop_size") else None,
    )

    quality_records = []
    calibration_summary_records = []
    coverage_records = []
    sparsification_records = []

    for batch in tqdm(loader, desc="Uncertainty eval"):
        lr = batch["lr"].to(device)
        hr = batch["hr"].to(device)
        filenames = [Path(f).name for f in batch["filename"]]

        if uncertainty_backend == "consistency":
            ensemble = run_consistency_ensemble_outputs(
                model,
                lr,
                n_steps=n_steps,
                n_runs=n_runs,
                use_ema=use_ema,
                uncertainty_metric=uncertainty_metric,
                uncertainty_space=uncertainty_space,
                uncertainty_normalization=uncertainty_normalization,
                seed_offset=seed_offset,
                eps=uncertainty_eps,
            )
        else:
            ensemble = run_diffusion_ensemble_outputs(
                model,
                lr,
                n_steps=n_steps,
                n_runs=n_runs,
                use_ddim=use_ddim,
                ddim_eta=ddim_eta,
                uncertainty_metric=uncertainty_metric,
                uncertainty_space=uncertainty_space,
                uncertainty_normalization=uncertainty_normalization,
                seed_offset=seed_offset,
                eps=uncertainty_eps,
            )
        stack = ensemble["stack"]
        ensemble_time = ensemble["elapsed_s"]
        mean_pred = ensemble["mean"]
        std_pred = ensemble["std"]
        var_pred = ensemble["var"]
        uncertainty_map = ensemble["uncertainty"]
        confidence_map = ensemble["confidence"]

        baseline = lr
        if baseline.shape[-2:] != mean_pred.shape[-2:]:
            baseline = F.interpolate(
                baseline,
                size=mean_pred.shape[-2:],
                mode="bicubic",
                align_corners=False,
            )

        robust_pred = robust_ensemble_average(
            stack,
            temperature=robust_temperature,
        )

        if uncertainty_space.lower() == "wavelet":
            mean_dwt = ensemble["mean_dwt"]
            uncertainty_dwt = ensemble["uncertainty_dwt"]
            if mean_dwt is None or uncertainty_dwt is None:
                raise RuntimeError("Wavelet refinement requested, but wavelet outputs are missing.")

            refine_anchor_image = robust_pred if refine_anchor == "robust" else baseline
            if uncertainty_backend == "diffusion":
                netG = model.netG.module if isinstance(model.netG, torch.nn.DataParallel) else model.netG
                refine_anchor_scaled = refine_anchor_image * 2.0 - 1.0
                baseline_dwt = netG.apply_dwt(refine_anchor_scaled)
            else:
                baseline_dwt = model.apply_dwt(refine_anchor_image)
            refined_dwt = refine_prediction_with_uncertainty(
                mean_prediction=mean_dwt,
                baseline_prediction=baseline_dwt,
                uncertainty_map=uncertainty_dwt,
                blend_strength=blend_strength,
                use_confidence_normalization=use_confidence_normalization,
                ensemble_runs=n_runs,
                eps=uncertainty_eps,
            )
            if uncertainty_backend == "diffusion":
                netG = model.netG.module if isinstance(model.netG, torch.nn.DataParallel) else model.netG
                refined_scaled = netG.apply_idwt(refined_dwt, mean_pred.shape[-2], mean_pred.shape[-1])
                refined_pred = ((refined_scaled + 1.0) / 2.0).clamp(0.0, 1.0)
            else:
                refined_pred = model.apply_idwt(refined_dwt, mean_pred.shape[-2], mean_pred.shape[-1])
        else:
            refine_anchor_image = robust_pred if refine_anchor == "robust" else baseline
            refined_pred = refine_prediction_with_uncertainty(
                mean_prediction=mean_pred,
                baseline_prediction=refine_anchor_image,
                uncertainty_map=uncertainty_map,
                blend_strength=blend_strength,
                use_confidence_normalization=use_confidence_normalization,
                ensemble_runs=n_runs,
                eps=uncertainty_eps,
            )

        for i, fname in enumerate(filenames):
            mean_i = mean_pred[i : i + 1].detach().cpu()
            refined_i = refined_pred[i : i + 1].detach().cpu()
            robust_i = robust_pred[i : i + 1].detach().cpu()
            baseline_i = baseline[i : i + 1].detach().cpu()
            hr_i = hr[i : i + 1].detach().cpu()

            uncertainty_i = uncertainty_map[i : i + 1].detach().cpu()
            confidence_i = confidence_map[i : i + 1].detach().cpu()
            std_i = std_pred[i : i + 1].detach().cpu()
            var_i = var_pred[i : i + 1].detach().cpu()

            for variant, pred_i in [
                ("mean", mean_i),
                ("refined", refined_i),
                ("robust", robust_i),
                ("baseline", baseline_i),
            ]:
                m = _fast_metrics(pred_i, hr_i)
                row = {
                    "filename": fname,
                    "variant": variant,
                    "psnr": m["psnr"],
                    "mae": m["mae"],
                    "rmse": m["rmse"],
                    "uncertainty_backend": uncertainty_backend,
                    "n_steps": n_steps,
                    "uncertainty_runs": n_runs,
                    "uncertainty_metric": uncertainty_metric,
                    "uncertainty_space": uncertainty_space,
                    "uncertainty_normalization": uncertainty_normalization,
                    "ensemble_time_s": ensemble_time / max(len(filenames), 1),
                }
                for ch_idx, ch_val in enumerate(m["mae_ch"]):
                    row[f"mae_ch{ch_idx}"] = ch_val
                for ch_idx, ch_val in enumerate(m["rmse_ch"]):
                    row[f"rmse_ch{ch_idx}"] = ch_val
                for ch_idx, ch_val in enumerate(m["psnr_ch"]):
                    row[f"psnr_ch{ch_idx}"] = ch_val

                quality_records.append(row)

            calib = evaluate_uncertainty_calibration(
                mean_prediction=mean_i,
                hr=hr_i,
                uncertainty_map=uncertainty_i,
                std_prediction=std_i,
                var_prediction=var_i,
                interval_levels=interval_levels,
                error_metric=error_metric,
                remove_fractions=remove_fractions,
            )

            corr = calib["correlation"]
            calibration_summary_records.append(
                {
                    "filename": fname,
                    "uncertainty_backend": uncertainty_backend,
                    "pearson_uncertainty_error": corr["pearson"],
                    "spearman_uncertainty_error": corr["spearman"],
                    "mean_nll": calib["gaussian_nll"]["mean_nll"],
                    "uncertainty_metric": uncertainty_metric,
                    "uncertainty_space": uncertainty_space,
                    "uncertainty_normalization": uncertainty_normalization,
                }
            )

            for row in calib["coverage"]["coverage"]:
                cov_row = {
                    "filename": fname,
                    "uncertainty_backend": uncertainty_backend,
                    "interval": row["interval"],
                    "z_value": row["z_value"],
                    "empirical": row["empirical"],
                    "gap": row["gap"],
                    "uncertainty_metric": uncertainty_metric,
                    "uncertainty_space": uncertainty_space,
                    "uncertainty_normalization": uncertainty_normalization,
                }
                for ch_idx, ch_cov in enumerate(row["per_channel"]):
                    cov_row[f"channel{ch_idx}"] = ch_cov
                coverage_records.append(cov_row)

            for row in calib["sparsification"]["curve"]:
                sparsification_records.append(
                    {
                        "filename": fname,
                        "uncertainty_backend": uncertainty_backend,
                        "remove_fraction": row["remove_fraction"],
                        "keep_fraction": row["keep_fraction"],
                        "num_kept_pixels": row["num_kept_pixels"],
                        "mae": row["mae"],
                        "rmse": row["rmse"],
                        "uncertainty_metric": uncertainty_metric,
                        "uncertainty_space": uncertainty_space,
                        "uncertainty_normalization": uncertainty_normalization,
                    }
                )

            if save_images:
                save_tif(mean_i[0], str(mean_out / fname))
                save_tif(refined_i[0], str(refined_out / fname))
                save_tif(robust_i[0], str(robust_out / fname))
                save_tif(baseline_i[0], str(baseline_out / fname))
                save_tif(uncertainty_i[0], str(uncertainty_out / fname))
                save_tif(confidence_i[0], str(confidence_out / fname))

    df_quality = pd.DataFrame(quality_records)
    df_calib = pd.DataFrame(calibration_summary_records)
    df_cov = pd.DataFrame(coverage_records)
    df_sparse = pd.DataFrame(sparsification_records)

    df_quality.to_csv(out_root / "prediction_quality.csv", index=False)
    df_calib.to_csv(out_root / "calibration_summary.csv", index=False)
    df_cov.to_csv(out_root / "coverage.csv", index=False)
    df_sparse.to_csv(out_root / "sparsification.csv", index=False)

    print("\nSaved uncertainty evaluation artifacts to:")
    print(f"  {out_root}")

    if not df_quality.empty:
        summary = df_quality.groupby("variant")[["psnr", "mae", "rmse"]].mean().reset_index()
        print("\nPrediction quality summary (mean over images):")
        print(summary.to_string(index=False))

    if not df_calib.empty:
        print("\nCalibration summary (mean over images):")
        print(
            df_calib[["pearson_uncertainty_error", "spearman_uncertainty_error", "mean_nll"]]
            .mean()
            .to_string()
        )

    return out_root


if __name__ == "__main__":
    cfg = InferenceConfig()
    evaluate_uncertainty(cfg)
