"""
CURE: Concept Unlearning via Orthogonal Representation Editing

Main implementation of the Spectral Eraser algorithm for Stable Diffusion.
"""

import torch
from typing import List, Optional, Union
from diffusers import StableDiffusionPipeline
from PIL import Image

from .spectral import compute_discriminative_projector
from .attention import get_cross_attention_layers, apply_weight_update, count_cross_attention_layers


class CURE:
    """
    CURE (Concept Unlearning via Orthogonal Representation Editing) for Stable Diffusion.

    This class implements the Spectral Eraser algorithm that removes concepts from
    diffusion models by modifying cross-attention weights. The method is:
    - Training-free (no fine-tuning required)
    - Fast (~2 seconds for concept erasure)
    - Preserves unrelated concepts

    Example:
        >>> pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")
        >>> cure = CURE(pipe)
        >>> cure.erase_concept(["car", "automobile", "vehicle"])
        >>> image = cure.generate("a red car on the street")  # Won't generate car
    """

    def __init__(
        self,
        pipe: StableDiffusionPipeline,
        device: Optional[str] = None
    ):
        """
        Initialize CURE with a Stable Diffusion pipeline.

        Args:
            pipe: A StableDiffusionPipeline instance
            device: Device to use ("cuda", "cpu", or None for auto-detect)
        """
        self.pipe = pipe

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        # Move pipeline to device
        self.pipe = self.pipe.to(self.device)

        # Store original weights for potential restoration
        self._original_weights = None

    def get_text_embeddings(self, prompts: List[str]) -> torch.Tensor:
        """
        Convert text prompts to CLIP token embeddings for subspace computation.

        Each prompt produces 77 token embeddings via CLIP ViT-L/14. We preserve
        all token embeddings (reshape, not aggregate) to match the cross-attention
        input distribution and give SVD more data points.

        Args:
            prompts: List of text prompts (each describing a concept)

        Returns:
            Embeddings tensor of shape [n_prompts * 77, hidden_dim]
        """
        tokenizer = self.pipe.tokenizer
        text_encoder = self.pipe.text_encoder

        # Tokenize
        tokens = tokenizer(
            prompts,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt"
        )

        input_ids = tokens.input_ids.to(self.device)

        # Get embeddings
        with torch.no_grad():
            outputs = text_encoder(input_ids)
            # Use the hidden states (before final layer norm for richer representation)
            embeddings = outputs.last_hidden_state # [batch, seq_len, hidden_dim]
            

        # Reshape to preserve all 77 token embeddings per prompt
        # This respects CLIP ViT-L/14's training distribution and gives SVD
        # more data points for robust subspace estimation. Padding tokens
        # naturally get near-zero singular values and are deprioritized.
        batch_size, seq_len, hidden_dim = embeddings.shape
        embeddings = embeddings.reshape(batch_size * seq_len, hidden_dim).float()  # [batch*77, 768], float32 for SVD

        return embeddings

    def compute_spectral_eraser(
        self,
        forget_embeddings: torch.Tensor,
        retain_embeddings: Optional[torch.Tensor],
        alpha: float
    ) -> torch.Tensor:
        """
        Compute the Spectral Eraser (discriminative projector).

        The algorithm:
        1. Perform SVD on forget/retain embeddings
        2. Build energy-scaled projectors Pf, Pr using spectral expansion
        3. Compute discriminative projector Pdis = Pf - Pf @ Pr

        Args:
            forget_embeddings: Embeddings of concepts to forget [n_forget, hidden_dim]
            retain_embeddings: Embeddings of concepts to retain [n_retain, hidden_dim]
            alpha: Spectral expansion parameter (2 for objects, 5 for NSFW)

        Returns:
            Discriminative projector [hidden_dim, hidden_dim]
        """
        return compute_discriminative_projector(
            forget_embeddings=forget_embeddings,
            retain_embeddings=retain_embeddings,
            alpha=alpha
        )

    def save_original_weights(self) -> None:
        """Save original cross-attention weights for later restoration."""
        self._original_weights = []
        for layer in get_cross_attention_layers(self.pipe.unet):
            self._original_weights.append({
                'Wk': layer.to_k.weight.data.clone(),
                'Wv': layer.to_v.weight.data.clone()
            })

    def restore_original_weights(self) -> None:
        """Restore original cross-attention weights."""
        if self._original_weights is None:
            raise ValueError("No original weights saved. Call save_original_weights() first.")

        for layer, weights in zip(get_cross_attention_layers(self.pipe.unet),
                                   self._original_weights):
            layer.to_k.weight.data = weights['Wk'].clone()
            layer.to_v.weight.data = weights['Wv'].clone()

    def erase_concept(
        self,
        forget_prompts: List[str],
        retain_prompts: Optional[List[str]] = None,
        alpha: float = 5.0,
        save_original: bool = True
    ) -> None:
        """
        Erase a concept from the model.

        This is the main entry point for concept removal. It:
        1. Gets embeddings for forget/retain concepts
        2. Computes the Spectral Eraser
        3. Applies it to all cross-attention Wk, Wv weights

        Args:
            forget_prompts: List of prompts describing concepts to forget
                           e.g., ["car", "automobile", "vehicle", "sedan"]
            retain_prompts: Optional list of prompts for concepts to preserve
                           e.g., ["bicycle", "motorcycle"] (related but different)
            alpha: Spectral expansion parameter
                   - Use 2 for object/artist removal
                   - Use 5 for NSFW content (stronger erasure)
            save_original: If True, save original weights for potential restoration
        """
        if save_original and self._original_weights is None:
            self.save_original_weights()

        # Get embeddings for forget concepts
        forget_embeddings = self.get_text_embeddings(forget_prompts)

        # Get embeddings for retain concepts (if provided)
        retain_embeddings = None
        if retain_prompts is not None and len(retain_prompts) > 0:
            retain_embeddings = self.get_text_embeddings(retain_prompts)

        # Compute the discriminative projector
        projector = self.compute_spectral_eraser(
            forget_embeddings=forget_embeddings,
            retain_embeddings=retain_embeddings,
            alpha=alpha
        )

        # Apply to all cross-attention layers
        n_layers = 0
        for layer in get_cross_attention_layers(self.pipe.unet):
            apply_weight_update(layer, projector, device=self.device)
            n_layers += 1

        print(f"Applied Spectral Eraser to {n_layers} cross-attention layers")

    def generate(
        self,
        prompt: Union[str, List[str]],
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        generator: Optional[torch.Generator] = None,
        **kwargs
    ) -> List[Image.Image]:
        """
        Generate images with the modified model.

        Args:
            prompt: Text prompt(s) for generation
            negative_prompt: Negative prompt(s) to avoid
            num_inference_steps: Number of denoising steps
            guidance_scale: Classifier-free guidance scale
            generator: Random number generator for reproducibility
            **kwargs: Additional arguments passed to the pipeline

        Returns:
            List of generated PIL Images
        """
        output = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
            **kwargs
        )

        return output.images

    def __repr__(self) -> str:
        n_layers = count_cross_attention_layers(self.pipe.unet)
        return f"CURE(device={self.device}, cross_attention_layers={n_layers})"
