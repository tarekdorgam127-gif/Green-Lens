from __future__ import annotations

from typing import Optional, Tuple

import matplotlib.cm as cm
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps

from greenlens.data.preprocessing import get_eval_transforms


class GradCAM:
    """Gradient-weighted Class Activation Mapping (Grad-CAM) for ResNet-18."""

    def __init__(self, model: torch.nn.Module, target_layer: Optional[torch.nn.Module] = None):
        self.model = model
        self.model.eval()
        self.device = next(model.parameters()).device

        # Default target layer is the last residual block of layer4 in ResNet-18
        self.target_layer = target_layer or self.model.layer4[-1]

        self.gradients = None
        self.activations = None

        # Register forward and backward hooks
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0]

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(
        self,
        image: Image.Image,
        target_class_idx: Optional[int] = None,
    ) -> Tuple[np.ndarray, Image.Image, int]:
        """Generate Grad-CAM heatmap and blended overlay image for a given input PIL Image."""
        # Ensure image is in RGB and oriented correctly
        image_clean = ImageOps.exif_transpose(image).convert("RGB")
        orig_size = image_clean.size

        # Preprocess
        transform = get_eval_transforms()
        input_tensor = transform(image_clean).unsqueeze(0).to(self.device)

        # Forward pass with gradient tracking enabled
        self.model.zero_grad()
        output = self.model(input_tensor)

        if target_class_idx is None:
            target_class_idx = int(output.argmax(dim=1).item())

        # Backward pass on the score for the target class
        score = output[0, target_class_idx]
        score.backward()

        # Pool the gradients across the channels
        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])

        # Weight the activations by computed gradients
        activations = self.activations[0]
        for i in range(activations.size(0)):
            activations[i, :, :] *= pooled_gradients[i]

        # Channel-wise mean of the weighted activations
        heatmap = torch.mean(activations, dim=0).squeeze()
        heatmap = F.relu(heatmap).detach().cpu().numpy()

        # Normalize heatmap between 0 and 1
        if np.max(heatmap) > 0:
            heatmap = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap) + 1e-8)
        else:
            heatmap = np.zeros_like(heatmap)

        # Resize heatmap to original image size
        heatmap_resized = Image.fromarray(np.uint8(255 * heatmap)).resize(
            orig_size, resample=Image.Resampling.BILINEAR
        )
        heatmap_np = np.array(heatmap_resized) / 255.0

        # Apply Jet Color Map
        try:
            import matplotlib
            colormap = matplotlib.colormaps["jet"]
        except (AttributeError, KeyError):
            colormap = cm.get_cmap("jet")

        colored_heatmap = colormap(heatmap_np)[:, :, :3]  # Drop alpha
        colored_heatmap = np.uint8(colored_heatmap * 255)

        # Superimpose the heatmap on input image (alpha blend 0.5)
        orig_np = np.array(image_clean)
        overlay = np.uint8(0.55 * orig_np + 0.45 * colored_heatmap)
        overlay_img = Image.fromarray(overlay)

        return heatmap_np, overlay_img, target_class_idx
