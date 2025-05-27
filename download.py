from huggingface_hub import hf_hub_download

model_dir = './pretrained'   # The local directory to save downloaded checkpoint
hf_hub_download(repo_id="Lightricks/LTX-Video", filename="ltx-video-2b-v0.9.5.safetensors", 
    local_dir=model_dir, local_dir_use_symlinks=False, repo_type='model')