
import os, random

import torch
from diffusers.utils import export_to_video, load_image
from inference_base import infer, create_ltx_video_pipeline, prepare_args, load_lora_weights, load_diffuser_lora_weights

args = prepare_args()

LORA_WEIGHT = 0.9
guidance = 3

# diffuser_lora_path = "/home/eisneim/www/ml/_video_gen/ltx_training/data_flow/flow_assist/flow_assist_v1/checkpoint-29000/pytorch_lora_weights.safetensors"
diffuser_lora_path = "/home/eisneim/www/ml/_video_gen/ltx_training/data_256/mixed/checkpoint-8000/pytorch_lora_weights.safetensors"
lora_path = "/home/eisneim/www/ml/_video_gen/LTX-Video-3-22/data/lora_256/checkpoint-10"
# lora_path = "/home/eisneim/www/ml/_video_gen/ltx_training/data_256/mixed/checkpoint-57000"

prefix = "freeze time, camera orbit left, " 
# negative_prompt = "worst quality, static, noisy, inconsistent motion, blurry, jittery, distorted"
negative_prompt = "worst quality, static no camera movement, noisy, inconsistent motion, blurry, jittery, distorted"
enhance_prompt = False

args.negative_prompt = negative_prompt
args.guidance_scale = guidance
args.width = 1280
args.height = 704
args.num_frames = 49
args.seed = 1324
args.conditioning_start_frames = [0]
args.offload_to_cpu = True
args.output_path = "data/outputs/3-25"
args.device = "cuda"
# images_dir = "data/images/ltx_actions/"
images_dir = "data/images/snapshots/"

print(args)

if not os.path.exists(args.output_path):
    os.makedirs(args.output_path)

pairs = []
for ii in os.listdir(images_dir):
    if ii.endswith(".jpg") and ii[0] != ".":
        file = os.path.join(images_dir, ii)
        prompt = open(os.path.join(images_dir, ii.rsplit(".", 1)[0] + ".txt"), "r").read()
        prompt = prompt.replace("The image", "The video")
        pairs.append((file, prompt))

print("loaded images", len(pairs))

pipeline = create_ltx_video_pipeline(
    ckpt_path=args.ckpt_path,
    precision=args.precision,
    text_encoder_model_name_or_path=args.text_encoder_model_name_or_path,
    sampler=args.sampler,
    device=args.device,
    enhance_prompt=enhance_prompt,
    prompt_enhancer_image_caption_model_name_or_path=args.prompt_enhancer_image_caption_model_name_or_path,
    prompt_enhancer_llm_model_name_or_path=args.prompt_enhancer_llm_model_name_or_path,
)

pipeline.transformer = load_lora_weights(pipeline.transformer, lora_path, lora_weight=1.0, name="orbit")
load_diffuser_lora_weights(pipeline.transformer, diffuser_lora_path, "orbit", weight=LORA_WEIGHT)

# for idx in range(3):
for idx in range(len(pairs)):
    file, prompt = pairs[idx]
    print(f"[{idx}/{len(pairs)}] {file}")
    prompt = prefix + prompt
    
    fname = f"_29k_{LORA_WEIGHT}_{args.num_frames}x{args.width}x{args.height}.mp4"
    dest = os.path.join(args.output_path, os.path.basename(file).replace(".jpg", "") + fname)
    # image = load_image(file).resize((width, height))

    args.prompt = prompt
    print(args.prompt)
    args.conditioning_media_paths = [file]

    infer(pipeline, output_filename=dest, **vars(args))

    # export_to_video(video, dest, fps=24)

print("done!")