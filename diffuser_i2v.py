import torch
from diffusers.pipelines.ltx.pipeline_ltx_condition import LTXConditionPipeline, LTXVideoCondition
from diffusers.utils import export_to_video, load_video, load_image

pipe = LTXConditionPipeline.from_pretrained("Lightricks/LTX-Video-0.9.5", torch_dtype=torch.bfloat16)
pipe.to("cuda")

# Load input image and video
# video = load_video(
#     "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/cosmos/cosmos-video2world-input-vid.mp4"
# )
image = load_image(
    "data/portraits/1721558728737_73.jpg"
)

# Create conditioning objects
condition1 = LTXVideoCondition(
    image=image,
    frame_index=0,
)
# condition2 = LTXVideoCondition(
#     video=video,
#     frame_index=80,
# )

prompt = "a chinese woman lying on a field of lotus, she is blinking her eyes, light wind blowing, no camera movement"
negative_prompt = "worst quality, inconsistent motion, blurry, jittery, distorted"

# Generate video
generator = torch.Generator("cuda").manual_seed(11)
video = pipe(
    conditions=[condition1],
    prompt=prompt,
    negative_prompt=negative_prompt,
    width=768,
    height=512,
    num_frames=97,
    num_inference_steps=40,
    generator=generator,
).frames[0]

export_to_video(video, "data/diffusers_output.mp4", fps=24)