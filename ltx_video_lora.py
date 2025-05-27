from typing import Dict, List, Optional, Union
import random
import torch
import torch.nn as nn
from accelerate.logging import get_logger
# from diffusers import  FlowMatchEulerDiscreteScheduler
from ltx_video.schedulers.rf import RectifiedFlowScheduler
from diffusers.utils import logging
from transformers import T5EncoderModel, T5Tokenizer
from PIL import Image
from diffusers.utils.torch_utils import randn_tensor
from einops import rearrange

from ltx_video.models.autoencoders.causal_video_autoencoder import (
    CausalVideoAutoencoder,
)
from ltx_video.models.autoencoders.vae_encode import (
    get_vae_size_scale_factor,
    latent_to_pixel_coords,
    vae_decode,
    vae_encode,
)
from ltx_video.models.transformers.transformer3d import Transformer3DModel
from ltx_video.pipelines.pipeline_ltx_video import ConditioningItem, LTXVideoPipeline


logger = get_logger("ltx_train")  # pylint: disable=invalid-name
MODEL_ID = "./pretrained/ltx-video-2b-v0.9.5.safetensors"
text_encoder_id = "PixArt-alpha/PixArt-XL-2-1024-MS"
# MODEL_ID = "./data/fused"

def load_condition_models(
    model_id: str = text_encoder_id,
    text_encoder_dtype: torch.dtype = torch.bfloat16,
    revision: Optional[str] = None,
    cache_dir: Optional[str] = None,
    **kwargs,
) -> Dict[str, nn.Module]:
    tokenizer = T5Tokenizer.from_pretrained(model_id, subfolder="tokenizer", revision=revision, cache_dir=cache_dir)
    text_encoder = T5EncoderModel.from_pretrained(
        model_id, subfolder="text_encoder", torch_dtype=text_encoder_dtype, revision=revision, cache_dir=cache_dir
    )
    return {"tokenizer": tokenizer, "text_encoder": text_encoder}


def load_latent_models(
    model_id: str = MODEL_ID,
    vae_dtype: torch.dtype = torch.bfloat16,
    revision: Optional[str] = None,
    cache_dir: Optional[str] = None,
    **kwargs,
) -> Dict[str, nn.Module]:
    # vae = AutoencoderKLLTXVideo.from_pretrained(
    #     model_id, subfolder="vae", torch_dtype=vae_dtype, revision=revision, cache_dir=cache_dir
    # )
    vae = CausalVideoAutoencoder.from_pretrained(model_id, torch_dtype=vae_dtype, cache_dir=cache_dir)
    return {"vae": vae}


def load_diffusion_models(
    model_id: str = MODEL_ID,
    transformer_dtype: torch.dtype = torch.bfloat16,
    revision: Optional[str] = None,
    cache_dir: Optional[str] = None,
    **kwargs,
) -> Dict[str, nn.Module]:
    transformer = Transformer3DModel.from_pretrained(
        model_id, subfolder="transformer", torch_dtype=transformer_dtype, revision=revision, cache_dir=cache_dir,
    )
    # scheduler = FlowMatchEulerDiscreteScheduler()
    scheduler = RectifiedFlowScheduler()
    return {"transformer": transformer, "scheduler": scheduler}


def initialize_pipeline(
    model_id: str = MODEL_ID,
    text_encoder_dtype: torch.dtype = torch.bfloat16,
    transformer_dtype: torch.dtype = torch.bfloat16,
    vae_dtype: torch.dtype = torch.bfloat16,
    tokenizer: Optional[T5Tokenizer] = None,
    text_encoder: Optional[T5EncoderModel] = None,
    transformer: Optional[Transformer3DModel] = None,
    vae: Optional[CausalVideoAutoencoder] = None,
    scheduler: Optional[RectifiedFlowScheduler] = None,
    device: Optional[torch.device] = None,
    revision: Optional[str] = None,
    cache_dir: Optional[str] = None,
    enable_slicing: bool = False,
    enable_tiling: bool = False,
    enable_model_cpu_offload: bool = False,
    **kwargs,
) -> LTXVideoPipeline:
    component_name_pairs = [
        ("tokenizer", tokenizer),
        ("text_encoder", text_encoder),
        ("transformer", transformer),
        ("vae", vae),
        ("scheduler", scheduler),
    ]
    components = {}
    for name, component in component_name_pairs:
        if component is not None:
            components[name] = component

    pipe = LTXVideoPipeline.from_pretrained(model_id, **components, revision=revision, cache_dir=cache_dir)
    pipe.text_encoder = pipe.text_encoder.to(dtype=text_encoder_dtype)
    pipe.transformer = pipe.transformer.to(dtype=transformer_dtype)
    pipe.vae = pipe.vae.to(dtype=vae_dtype)

    if enable_slicing:
        pipe.vae.enable_slicing()
    if enable_tiling:
        pipe.vae.enable_tiling()

    if enable_model_cpu_offload:
        pipe.enable_model_cpu_offload(device=device)
    else:
        pipe.to(device=device)

    return pipe


def prepare_conditions(
    tokenizer: T5Tokenizer,
    text_encoder: T5EncoderModel,
    prompt: Union[str, List[str]],
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    max_sequence_length: int = 128,
    **kwargs,
) -> torch.Tensor:
    device = device or text_encoder.device
    dtype = dtype or text_encoder.dtype

    if isinstance(prompt, str):
        prompt = [prompt]

    return _encode_prompt_t5(tokenizer, text_encoder, prompt, device, dtype, max_sequence_length)

"""
!!! TODO: should use vae_decode, vae_encode,
instead of vae.encode().latent_dist.sample to match the inference code of LTX 0.9.5
"""
def prepare_latents(
    vae: CausalVideoAutoencoder,
    image_or_video: torch.Tensor,
    patch_size: int = 1,
    patch_size_t: int = 1,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    generator: Optional[torch.Generator] = None,
    precompute: bool = False,
) -> torch.Tensor:
    device = device or vae.device

    if image_or_video.ndim == 4:
        image_or_video = image_or_video.unsqueeze(2)
    assert image_or_video.ndim == 5, f"Expected 5D tensor, got {image_or_video.ndim}D tensor"

    image_or_video = image_or_video.to(device=device, dtype=vae.dtype)
    image_or_video = image_or_video.permute(0, 2, 1, 3, 4).contiguous()  # [B, C, F, H, W] -> [B, F, C, H, W]
    if not precompute:
        latents = vae.encode(image_or_video).latent_dist.sample(generator=generator)
        latents = latents.to(dtype=dtype)
        _, _, num_frames, height, width = latents.shape
        latents = _normalize_latents(latents, vae.mean_of_means, vae.std_of_means)
        latents = pack_latents(latents, patch_size, patch_size_t)
        return {"latents": latents, "num_frames": num_frames, "height": height, "width": width}
    else:
        if vae.use_slicing and image_or_video.shape[0] > 1:
            encoded_slices = [vae._encode(x_slice) for x_slice in image_or_video.split(1)]
            h = torch.cat(encoded_slices)
        else:
            h = vae._encode(image_or_video)
        _, _, num_frames, height, width = h.shape

        # TODO(aryan): This is very stupid that we might possibly be storing the latents_mean and latents_std in every file
        # if precomputation is enabled. We should probably have a single file where re-usable properties like this are stored
        # so as to reduce the disk memory requirements of the precomputed files.
        return {
            "latents": h,
            "num_frames": num_frames,
            "height": height,
            "width": width,
            "latents_mean": vae.mean_of_means,
            "latents_std": vae.std_of_means,
        }


def post_latent_preparation(
    latents: torch.Tensor,
    latents_mean: torch.Tensor,
    latents_std: torch.Tensor,
    num_frames: int,
    height: int,
    width: int,
    patch_size: int = 1,
    patch_size_t: int = 1,
) -> torch.Tensor:
    latents = _normalize_latents(latents, latents_mean, latents_std)
    latents = pack_latents(latents, patch_size, patch_size_t)
    return {"latents": latents, "num_frames": num_frames, "height": height, "width": width}


def collate_fn_t2v(batch: List[List[Dict[str, torch.Tensor]]]) -> Dict[str, torch.Tensor]:
    return {
        "prompts": [x["prompt"] for x in batch[0]],
        "videos": torch.stack([x["video"] for x in batch[0]]),
    }


def get_latent_coords(
    latent_num_frames, latent_height, latent_width, batch_size, device,
    _patch_size=(1, 1, 1)
):
    """
    Return a tensor of shape [batch_size, 3, num_patches] containing the
        top-left corner latent coordinates of each latent patch.
    The tensor is repeated for each batch element.
    """
    latent_sample_coords = torch.meshgrid(
        torch.arange(0, latent_num_frames, _patch_size[0], device=device),
        torch.arange(0, latent_height, _patch_size[1], device=device),
        torch.arange(0, latent_width, _patch_size[2], device=device),
    )
    latent_sample_coords = torch.stack(latent_sample_coords, dim=0)
    latent_coords = latent_sample_coords.unsqueeze(0).repeat(batch_size, 1, 1, 1, 1)
    latent_coords = rearrange(
        latent_coords, "b c f h w -> b c (f h w)", b=batch_size
    )
    return latent_coords

def forward_pass(
    transformer: Transformer3DModel,
    vae: CausalVideoAutoencoder,
    prompt_embeds: torch.Tensor,
    prompt_attention_mask: torch.Tensor,
    latents: torch.Tensor,
    noisy_latents: torch.Tensor,
    timesteps: torch.LongTensor,
    num_frames: int,
    height: int,
    width: int,
) -> torch.Tensor:
    frame_rate = 25
    bsz =  latents.size(0)
    latent_coords = get_latent_coords(num_frames, height, width, bsz, latents.device)
    pixel_coords = latent_to_pixel_coords(latent_coords, vae)
    
    # num_conds = 1
    # pixel_coords = torch.cat([pixel_coords] * num_conds)
    fractional_coords = pixel_coords.to(torch.float32)
    fractional_coords[:, 0] = fractional_coords[:, 0] * (1.0 / frame_rate)
    
    noise_pred = transformer(
        noisy_latents, #.to(latents.dtype),
        indices_grid=fractional_coords,
        encoder_hidden_states=prompt_embeds, # .to(latents.dtype),
        encoder_attention_mask=prompt_attention_mask,
        timestep=timesteps,
        return_dict=False,
    )[0]

    return {"latents": noise_pred}


def validation(
    pipeline: LTXVideoPipeline,
    prompt: str,
    image: Optional[Image.Image] = None,
    video: Optional[List[Image.Image]] = None,
    height: Optional[int] = None,
    width: Optional[int] = None,
    num_frames: Optional[int] = None,
    frame_rate: int = 25,
    num_videos_per_prompt: int = 1,
    generator: Optional[torch.Generator] = None,
    **kwargs,
):
    generation_kwargs = {
        "prompt": prompt,
        "height": height,
        "width": width,
        "num_frames": num_frames,
        "frame_rate": frame_rate,
        "num_videos_per_prompt": num_videos_per_prompt,
        "generator": generator,
        "return_dict": True,
        "output_type": "pil",
    }
    generation_kwargs = {k: v for k, v in generation_kwargs.items() if v is not None}
    video = pipeline(**generation_kwargs).frames[0]
    return [("video", video)]


def _encode_prompt_t5(
    tokenizer: T5Tokenizer,
    text_encoder: T5EncoderModel,
    prompt: List[str],
    device: torch.device,
    dtype: torch.dtype,
    max_sequence_length,
) -> torch.Tensor:
    batch_size = len(prompt)

    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=max_sequence_length,
        truncation=True,
        add_special_tokens=True,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids
    prompt_attention_mask = text_inputs.attention_mask
    prompt_attention_mask = prompt_attention_mask.bool().to(device)

    prompt_embeds = text_encoder(text_input_ids.to(device))[0]
    prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)
    prompt_attention_mask = prompt_attention_mask.view(batch_size, -1)

    return {"prompt_embeds": prompt_embeds, "prompt_attention_mask": prompt_attention_mask}


def _normalize_latents(
    latents: torch.Tensor, latents_mean: torch.Tensor, latents_std: torch.Tensor, scaling_factor: float = 1.0,
    reverse=False,
) -> torch.Tensor:
    # Normalize latents across the channel dimension [B, C, F, H, W]
    latents_mean = latents_mean.view(1, -1, 1, 1, 1).to(latents.device, latents.dtype)
    latents_std = latents_std.view(1, -1, 1, 1, 1).to(latents.device, latents.dtype)
    if not reverse:
        latents = (latents - latents_mean) * scaling_factor / latents_std
    else:
        latents = latents * latents_std / scaling_factor + latents_mean
    return latents

def unpack_latents(
        latents: torch.Tensor, num_frames: int, height: int, width: int, patch_size: int = 1, patch_size_t: int = 1
    ) -> torch.Tensor:
    # Packed latents of shape [B, S, D] (S is the effective video sequence length, D is the effective feature dimensions)
    # are unpacked and reshaped into a video tensor of shape [B, C, F, H, W]. This is the inverse operation of
    # what happens in the `pack_latents` method.
    batch_size = latents.size(0)
    latents = latents.reshape(batch_size, num_frames, height, width, -1, patch_size_t, patch_size, patch_size)
    latents = latents.permute(0, 4, 1, 5, 2, 6, 3, 7).flatten(6, 7).flatten(4, 5).flatten(2, 3)
    return latents

def pack_latents(latents: torch.Tensor, patch_size: int = 1, patch_size_t: int = 1) -> torch.Tensor:
    # Unpacked latents of shape are [B, C, F, H, W] are patched into tokens of shape [B, C, F // p_t, p_t, H // p, p, W // p, p].
    # The patch dimensions are then permuted and collapsed into the channel dimension of shape:
    # [B, F // p_t * H // p * W // p, C * p_t * p * p] (an ndim=3 tensor).
    # dim=0 is the batch size, dim=1 is the effective video sequence length, dim=2 is the effective number of input features
    batch_size, num_channels, num_frames, height, width = latents.shape
    post_patch_num_frames = num_frames // patch_size_t
    post_patch_height = height // patch_size
    post_patch_width = width // patch_size
    latents = latents.reshape(
        batch_size,
        -1,
        post_patch_num_frames,
        patch_size_t,
        post_patch_height,
        patch_size,
        post_patch_width,
        patch_size,
    )
    latents = latents.permute(0, 2, 4, 6, 1, 3, 5, 7).flatten(4, 7).flatten(1, 3)
    return latents


"""
Paper: Temporal Regularization Makes Your Video Generator Stronger
https://arxiv.org/abs/2503.15417
"""
def shuffle_latent_frames(latent, num_frames: int, height: int, width: int, shuffle_num=2):
    # now the shape should be: (B, C, F, H, W)
    latent = unpack_latents(latent, num_frames, height, width)
    new_latent = latent.clone()

    # Select N random indices along the F dimension (dim=2)
    indices = torch.randperm(num_frames)[:shuffle_num]  # Randomly permute indices and select the first 3

    # Shuffle the selected indices
    shuffled_indices = indices[torch.randperm(shuffle_num)]  # Shuffle the 3 indices

    #Assign the shuffled values back to the new tensor
    new_latent[:, :, indices, :, :] = new_latent[:, :, shuffled_indices, :, :]

    return pack_latents(new_latent)


# in i2v task, we need to init noise from the first frame of the video
def gen_noise_from_first_frame_latent(frame_latent, latent_num_frames, 
                                     latent_height=18, latent_width=32, 
                                     vae_spatial_compression_ratio=32,
                                     vae_temporal_compression_ratio=8, 
                                     generator=None, return_static=False, noise_to_first_frame=0.05):
    num_channels_latents = frame_latent.size(-1) # 128
    batch_size = frame_latent.size(0)
    # latent_num_frames = (num_frames - 1) // vae_temporal_compression_ratio + 1

    shape = (batch_size, num_channels_latents, latent_num_frames, latent_height, latent_width)
    mask_shape = (batch_size, 1, latent_num_frames, latent_height, latent_width)
    # 因为最开始生成数据集时不小心pack了第一帧所以现在多做这一步
    init_latents = unpack_latents(frame_latent, 1, latent_height, latent_width)
    init_latents = init_latents.repeat(1, 1, latent_num_frames, 1, 1)
    conditioning_mask = torch.zeros(mask_shape, device=frame_latent.device, dtype=frame_latent.dtype)
    conditioning_mask[:, :, 0] = 1.0

    rand_noise_ff = random.random() * noise_to_first_frame

    first_frame_mask = conditioning_mask.clone() 
    first_frame_mask[:, :, 0] = 1.0 - rand_noise_ff

    noise = randn_tensor(shape, generator=generator, device=frame_latent.device, dtype=frame_latent.dtype)
    latents = init_latents * first_frame_mask + noise * (1 - first_frame_mask)

    conditioning_mask = pack_latents(conditioning_mask).squeeze(-1)
    latents = pack_latents(latents)
    if return_static:
        static_latent = pack_latents(init_latents)
        return latents, conditioning_mask, static_latent

    return latents, conditioning_mask




LTX_VIDEO_T2V_LORA_CONFIG = {
    "pipeline_cls": LTXVideoPipeline,
    "load_condition_models": load_condition_models,
    "load_latent_models": load_latent_models,
    "load_diffusion_models": load_diffusion_models,
    "initialize_pipeline": initialize_pipeline,
    "prepare_conditions": prepare_conditions,
    "prepare_latents": prepare_latents,
    "post_latent_preparation": post_latent_preparation,
    "collate_fn": collate_fn_t2v,
    "forward_pass": forward_pass,
    "validation": validation,
}
