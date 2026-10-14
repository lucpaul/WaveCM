from collections.abc import Sequence
from typing import Tuple, Union

import torch.nn.functional as F
import torch
from torch import nn
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity


class PerceptualLoss(nn.Module):
    def __init__(
        self,
        *,
        net_type: Union[str, Tuple[str, ...]] = "vgg",
        l1_weight: float = 0.0,
        perceptual_weight: float = 1.0,
        **kwargs):
        """ Loss combining LPIPS and L1 norm.

        Args:
            net_type: Network type used for LPIPS.
            l1_weight: Weight factor of the L1 norm in the loss.
        """

        super().__init__()

        available_net_types = ("vgg", "alex", "squeeze")

        def _append_net_type(net_type: str):
            if net_type in available_net_types:
                self.lpips_losses.append(
                    LearnedPerceptualImagePatchSimilarity(net_type)
                )
            else:
                raise TypeError(
                    f"'net_type' should be on of {available_net_types}, got {net_type}"
                )

        self.lpips_losses = nn.ModuleList()

        if isinstance(net_type, str):
            _append_net_type(net_type)

        elif isinstance(net_type, Sequence):
            for _net_type in sorted(net_type):
                _append_net_type(_net_type)

        self.lpips_losses.requires_grad_(False)

        self.l1_weight = l1_weight
        self.perceptual_weight = perceptual_weight


    def forward(self, input: torch.tensor, target: torch.tensor):
        """ 
        Args:
            input: Network prediction.
            target: Ground truth.
        
        """
        input =  input.clamp(-1.0, 1.0)
        target =  target.clamp(-1.0, 1.0)

        upscaled_input = F.interpolate(input, (224, 224), mode="bilinear")
        upscaled_target = F.interpolate(target, (224, 224), mode="bilinear")

        if upscaled_input.shape[1] == 1 and upscaled_target.shape[1] == 1:
            upscaled_input = torch.concat([upscaled_input,upscaled_input,upscaled_input], dim=1)
            upscaled_target = torch.concat([upscaled_target,upscaled_target,upscaled_target], dim=1)

            lpips_loss_total = sum(
            _lpips_loss(upscaled_input, upscaled_target)
            for _lpips_loss in self.lpips_losses
        )
        
        elif upscaled_input.shape[1] == 4 and upscaled_target.shape[1] == 4:
            upscaled_input_1 = upscaled_input[:, 0:3, :, :]
            upscaled_input_2 = upscaled_input[:, 1:, :, :]
            upscaled_target_1 = upscaled_target[:, 0:3, :, :]
            upscaled_target_2 = upscaled_target[:, 1:, :, :]

            lpips_loss_1 = sum(
            _lpips_loss(upscaled_input_1, upscaled_target_1)
            for _lpips_loss in self.lpips_losses
            )

            lpips_loss_2 = sum(
            _lpips_loss(upscaled_input_2, upscaled_target_2)
            for _lpips_loss in self.lpips_losses
            )

            lpips_loss_total = self.perceptual_weight * ((lpips_loss_1 + lpips_loss_2) / 2)
        
        if self.l1_weight > 0.0:
            l1_loss = self.l1_weight * F.l1_loss(input, target)
            self.last_sub_losses = {
                'lpips_loss': lpips_loss_total.detach(),
                'L1_loss': l1_loss.detach()
                }
            return lpips_loss_total + l1_loss
        
        else:
            return lpips_loss_total

    def get_sub_losses(self):
        """Return the last computed sub-losses as a dictionary"""
        return self.last_sub_losses.copy()

LPIPSLoss = PerceptualLoss

class WeightedWaveletLoss(nn.Module):
    """Weighted loss that accounts for different importance of wavelet coefficients"""
    def __init__(self, ll_weight=1.0, lh_weight=3.0, hl_weight=3.0, hh_weight=6.0):
        super().__init__()
        self.ll_weight = ll_weight
        self.lh_weight = lh_weight  
        self.hl_weight = hl_weight
        self.hh_weight = hh_weight
        self.last_sub_losses = {}
        
    def forward(self, pred, target):
        # Split into wavelet subbands (assuming 16 channels from 4-channel images)
        pred_ll = pred[:, 0:4, :, :]    # Low-low (most important)
        pred_lh = pred[:, 4:8, :, :]    # Low-high (horizontal details)
        pred_hl = pred[:, 8:12, :, :]   # High-low (vertical details) 
        pred_hh = pred[:, 12:16, :, :]  # High-high (diagonal details)
        
        target_ll = target[:, 0:4, :, :]
        target_lh = target[:, 4:8, :, :]
        target_hl = target[:, 8:12, :, :]
        target_hh = target[:, 12:16, :, :]
        
        # # Compute weighted losses
        # loss_ll = F.mse_loss(pred_ll, target_ll) * self.ll_weight
        # loss_lh = F.mse_loss(pred_lh, target_lh) * self.lh_weight
        # loss_hl = F.mse_loss(pred_hl, target_hl) * self.hl_weight
        # loss_hh = F.mse_loss(pred_hh, target_hh) * self.hh_weight

        # Compute weighted losses
        loss_ll = F.l1_loss(pred_ll, target_ll) * self.ll_weight
        loss_lh = F.l1_loss(pred_lh, target_lh) * self.lh_weight
        loss_hl = F.l1_loss(pred_hl, target_hl) * self.hl_weight
        loss_hh = F.l1_loss(pred_hh, target_hh) * self.hh_weight

        self.last_sub_losses = {
            'wavelet_ll': loss_ll.detach(),
            'wavelet_lh': loss_lh.detach(),
            'wavelet_hl': loss_hl.detach(),
            'wavelet_hh': loss_hh.detach(),
        }
        
        # weights = self.ll_weight + self.lh_weight + self.hl_weight + self.hh_weight
        loss = loss_ll + loss_lh + loss_hl + loss_hh
        
        return loss

    def get_sub_losses(self):
        """Return the last computed sub-losses as a dictionary"""
        return self.last_sub_losses.copy()

class AdaptiveWaveletLoss(nn.Module):
    """Adaptive loss that weights based on coefficient magnitudes"""
    def __init__(self, base_weight=1.0, adaptive_factor=0.1):
        super().__init__()
        self.base_weight = base_weight
        self.adaptive_factor = adaptive_factor
        
    def forward(self, pred, target):
        # Calculate adaptive weights based on target magnitude
        with torch.no_grad():
            # Split into subbands
            target_ll = target[:, 0:4, :, :] 
            target_high = target[:, 4:16, :, :]
            
            # Calculate relative importance
            ll_magnitude = torch.mean(torch.abs(target_ll))
            high_magnitude = torch.mean(torch.abs(target_high))
            
            # Avoid division by zero
            total_magnitude = ll_magnitude + high_magnitude + 1e-8
            ll_weight = ll_magnitude / total_magnitude + self.base_weight
            high_weight = high_magnitude / total_magnitude + self.base_weight * self.adaptive_factor
        
        # Apply weighted loss
        pred_ll = pred[:, 0:4, :, :]
        pred_high = pred[:, 4:16, :, :]
        target_ll = target[:, 0:4, :, :]
        target_high = target[:, 4:16, :, :]
        
        # loss_ll = F.mse_loss(pred_ll, target_ll) * ll_weight
        # loss_high = F.mse_loss(pred_high, target_high) * high_weight

        loss_ll = F.l1_loss(pred_ll, target_ll) * ll_weight
        loss_high = F.l1_loss(pred_high, target_high) * high_weight
        
        return loss_ll + loss_high

# Alternative: Image domain loss for better perceptual quality
class HybridWaveletImageLoss(nn.Module):
    """Combines wavelet domain loss with image domain loss"""
    def __init__(self, wavelet_weight=0.3, image_weight=0.7, perceptual_weight=1.0, l1_weight=0.0, xfm=None, ifm=None):
        super().__init__()
        self.wavelet_weight = wavelet_weight
        self.image_weight = image_weight
        self.wavelet_loss = WeightedWaveletLoss()
        self.xfm = xfm  # DWT transform
        self.ifm = ifm  # IDWT transform
        self.last_sub_losses = {}

        self.perceptual_loss = PerceptualLoss(net_type="vgg", l1_weight=l1_weight)
        
    def wavelet_to_image(self, wavelet_coeffs, target_h, target_w):
        """Convert wavelet coefficients back to images"""
        sr_images_LL = wavelet_coeffs[:, 0:4, :, :]
        sr_images_HL = wavelet_coeffs[:, 4:8, :, :].unsqueeze(2)
        sr_images_LH = wavelet_coeffs[:, 8:12, :, :].unsqueeze(2)  
        sr_images_HH = wavelet_coeffs[:, 12:16, :, :].unsqueeze(2)
        
        sr_HFreqs = torch.cat([sr_images_HL, sr_images_LH, sr_images_HH], 2)
        sr_images = self.ifm((sr_images_LL, [sr_HFreqs]))

        ### Below should only be necessary if images were not upsampled before ###
        # sr_images = F.interpolate(sr_images, size=(target_h, target_w), mode='bicubic')
        return sr_images
        
    def forward(self, pred_wavelet, target_wavelet, target_h, target_w):
        # Wavelet domain loss
        # wavelet_loss = F.mse_loss(pred_wavelet, target_wavelet)
        # wavelet_loss = F.l1_loss(pred_wavelet, target_wavelet)
        # image_loss = PerceptualLoss()
        wavelet_loss = self.wavelet_loss(pred_wavelet, target_wavelet)
        # print("Wavelet loss", wavelet_loss)
        # Convert to image domain for image loss
        pred_image = self.wavelet_to_image(pred_wavelet, target_h, target_w)
        target_image = self.wavelet_to_image(target_wavelet, target_h, target_w)
        # image_loss = image_loss(pred_image, target_image)
        # image_loss = F.mse_loss(pred_image, target_image)
        # image_loss = F.l1_loss(pred_image, target_image)
        image_loss = self.perceptual_loss(pred_image, target_image)

        # Store sub-losses (detached)
        self.last_sub_losses = {
            'wavelet_total': wavelet_loss.detach(),
            'image_total': image_loss.detach(),
            'wavelet_weighted': (self.wavelet_weight * wavelet_loss).detach(),
            'image_weighted': (self.image_weight * image_loss).detach(),
        }
        if hasattr(image_loss, "last_sub_losses"):
            perceptual_sub_losses = self.image_loss.get_sub_losses()
            self.last_sub_losses.update(perceptual_sub_losses)

        if hasattr(self.wavelet_loss, 'last_sub_losses'):
            wavelet_sub_losses = self.wavelet_loss.get_sub_losses()
            self.last_sub_losses.update(wavelet_sub_losses)
        
        return self.wavelet_weight * wavelet_loss + self.image_weight * image_loss

    def get_sub_losses(self):
        """Return the last computed sub-losses as a dictionary"""
        return self.last_sub_losses.copy()

class TotalVariationLoss(nn.Module):
    """Penalizes checkerboard by encouraging smooth gradients"""
    def __init__(self, weight=0.01):
        super().__init__()
        self.weight = weight
    
    def forward(self, image):
        # Compute pixel-wise differences
        diff_y = image[..., 1:, :] - image[..., :-1, :]
        diff_x = image[..., :, 1:] - image[..., :, :-1]
        
        # L2 norm (can also use L1 for stronger smoothing)
        tv_loss = torch.mean(diff_y**2) + torch.mean(diff_x**2)
        
        return self.weight * tv_loss
