import subprocess
import psutil
import time
import gc

# sudo sync; sudo sysctl -w vm.drop_caches=3
# command = "accelerate launch --config_file ./config.yaml  batch_2x.py "
command = "CUDA_VISIBLE_DEVICES=1 python precompute.py"

for ii in range(50):
	print("run time:", ii)
	result = subprocess.run([ command ], shell=True, capture_output=False, text=True)
	print(result.stdout)
	
	# Clear cached memory
	# subprocess.run(["sudo", "sysctl", "-w", "vm.drop_caches=3"])

	# Monitor memory usage
	memory_info = psutil.virtual_memory()
	print(f"Available Memory after cleanup: {memory_info.available / (1024 ** 3):.2f} GB")

	time.sleep(15)
	# Force garbage collection
	gc.collect()
	