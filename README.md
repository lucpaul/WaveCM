# Consistency Model Distillation for SR3

## Project abstract

This project trains a consistency model for single-image super-resolution by distilling a pretrained SR3-style diffusion model. The workflow starts from a diffusion checkpoint, converts it into a consistency-compatible initialization, and then trains a wavelet-based consistency network to approximate the diffusion generator with much faster inference. The repository also includes a side-by-side evaluation pipeline that runs both the diffusion and consistency models on the same test set and reports image-quality metrics such as PSNR, SSIM, MAE, LPIPS, and CLIP score.

The overall goal is to preserve the quality of the diffusion baseline while making inference substantially faster and more practical for repeated SR evaluation.

---

## 1. Training from a pretrained diffusion checkpoint

The training entry point is `train_consistency.py`. The script checks `config/configuration.py` for:

- `diffusion_checkpoint_path`: path to the pretrained diffusion model checkpoint
- `checkpoint_path`: where consistency checkpoints are written
- `data_path`, `data_path_lowres`, `data_path_test`: dataset locations
- `resume_ckpt_path`: optional path to resume training from a saved consistency checkpoint

### Recommended setup

1. Edit `config/configuration.py` and define your data and checkpoint paths.
2. Set `diffusion_checkpoint_path` to your pretrained diffusion checkpoint, for example:

```python
DiffusionConfig.diffusion_checkpoint_path = "/path/to/your/diffusion_model.pth"
```

3. Optionally set `resume_ckpt_path` if you want to continue a previous run.
4. Run the training script from the repository root:

```bash
python train_consistency.py
```

### What the script does automatically

When `resume_ckpt_path` is empty and a diffusion checkpoint is set, the script does the following:

- loads the diffusion checkpoint,
- converts it into a consistency-compatible checkpoint,
- initializes the consistency model from that converted checkpoint,
- trains the model with Lightning,
- saves checkpoints to `config.checkpoint_path` and a final model artifact.

---

## 2. Inference and diffusion vs consistency comparison

The comparison pipeline is implemented in `inference_consistency.py`. This script loads a trained consistency model and a diffusion model, runs them on the same LR test images, and compares their outputs with common SR metrics.

### Configuration

Edit `config/configuration_inference_default.py` before running inference:

- `consistency_ckpt`: checkpoint for the trained consistency model
- `diffusion_ckpt`: checkpoint for the pretrained diffusion model
- `lr_path` and `hr_path`: low-resolution and high-resolution test data paths
- `output_dir`: directory where generated SR outputs and CSV metric tables are written
- `n_consistency_steps`: number of consistency sampling steps (for example `1` or `10`)
- `n_diffusion_steps`: number of diffusion/DDIM steps
- `use_ddim`: set to `True` to use DDIM, `False` for standard diffusion
- `save_images`: whether to save generated `.tif` outputs to `consistency_sr` and `diffusion_sr`

An example configuration is already present in the file and uses a pretrained diffusion checkpoint by default.

### Run inference

From the project root:

```bash
python inference_consistency.py
```

### What the inference script compares

The script can evaluate and compare several variants, including:

- single-step consistency model inference,
- multi-step consistency model inference,
- diffusion or DDIM inference,
- DWSR-only baselines,
- ensemble average outputs,
- uncertainty evaluation if enabled.

For each evaluated model, it writes metric summaries and saves SR image outputs in directories such as:

- `consistency_sr/`
- `diffusion_sr/`
- output CSV files with metric scores

Typical comparison metrics include:

- PSNR
- SSIM
- MAE
- LPIPS
- CLIP score

This makes it straightforward to compare the speed/quality trade-off between the diffusion baseline and the distilled consistency model.

---

## 3. Typical end-to-end workflow

```bash
# 1. Edit configs
# - config/configuration.py
# - config/configuration_inference_default.py

# 2. Train consistency model from a pretrained diffusion checkpoint
./venv/bin/python train_consistency.py

# 3. Run evaluation against the diffusion model
./venv/bin/python inference_consistency.py
```

This is the recommended full pipeline: initialize the consistency model from a pretrained diffusion checkpoint, train it on the SR dataset, then evaluate the trained consistency model against the original diffusion baseline on the same test set.

---

## 4. Notes

- The training pipeline expects a valid pretrained diffusion checkpoint and dataset path layout.
- The conversion step is handled internally by `src/utils/checkpoint_converter.py`.
- For resuming training, set `resume_ckpt_path` in `config/configuration.py` and rerun `train_consistency.py`.

