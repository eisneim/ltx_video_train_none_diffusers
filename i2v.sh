#!/bin/bash


python inference.py --ckpt_path './pretrained/ltx-video-2b-v0.9.5.safetensors' \
	--prompt "a chinese woman lying on a field of lotus, she is blinking her eyes, light wind blowing, no camera movement" \
	--conditioning_media_paths "data/images/portraits/1721558728737_73.jpg" \
	--height 480 --width 704 \
	--num_frames 49 --seed 0 \
	--prompt_enhancement_words_threshold 10 \
	--output_path "data/outputs/" \
	--offload_to_cpu --conditioning_start_frames 0



