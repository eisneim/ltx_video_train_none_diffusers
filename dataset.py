import os, random, math
import torch
# import numpy as np
# from tqdm import tqdm
# from pathlib import Path
# from torchvision import transforms
# from PIL import Image


class PrecomputedDataset(torch.utils.data.Dataset):
    def __init__(
        self, 
        data_dir: str,
    ):
        super().__init__()        
        
        self.data_dir = data_dir

        self.data = []
        # load from cache
        for root, dirs, files in os.walk(self.data_dir):
            for file in files:
                if file.endswith('.pt') and file[0] != ".":
                    self.data.append(os.path.join(root, file))
        print(f"load cached videos: {len(self.data)}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        dpath = self.data[idx]
        try:
            # print("==", dpath)
            data = torch.load(dpath, weights_only=True)
        except:
            print(">>> error file:", dpath)
            data = torch.load(self.data[idx + 1])

        index = idx = random.randint(0, len(data["captions"]) - 1)
        latent, embeds = data["latents"][index], data["embedds"][index]
        assert not torch.isnan(latent.flatten()[0]), f"latent nan, {dpath}"
        assert not torch.isnan(embeds.flatten()[0]), f"embeds nan, {dpath}"
        first_frame = data["first_frames"][index].squeeze(0)
        
        return latent, first_frame, embeds, data["masks"][index], data["captions"][index], data["meta_info"][index]

"""
train lora to generate optical flow, then use the output to guide normal video
"""
class PrecomWithFlowDataset(PrecomputedDataset):
    def __init__(
        self, 
        data_dir: str,
        flow_dir: str,
    ):
        super().__init__(data_dir)
        self.flow_dir = flow_dir
        self.flow_pair = []
        self.flows = []

        for root, dirs, files in os.walk(self.flow_dir):
            for file in files:
                if file.endswith('.pt') and file[0] != ".":
                    flow_path = os.path.join(root, file)
                    self.flows.append(flow_path)

                    latent_file = file.replace("_flow", "")
                    latent_file_path = os.path.join(data_dir, latent_file)
                    # checck if we have this file already 
                    if latent_file_path in self.data:
                        self.flow_pair.append((flow_path, latent_file_path))
        
        print(">> found flow video pairs:", len(self.flow_pair), "flow files:", len(self.flows))

    def __len__(self):
        return len(self.flow_pair)
    
    def __getitem__(self, idx):
        fpath, dpath = self.flow_pair[idx]

        try:
            flow_data = torch.load(fpath, weights_only=True)
            data = torch.load(dpath, weights_only=True)
        except:
            print(">>> error file:", dpath, fpath)
            fpath, dpath = self.flow_pair[idx + 1]
            flow_data = torch.load(fpath, weights_only=True)
            data = torch.load(dpath, weights_only=True)

        index = idx = random.randint(0, len(flow_data["flow_latents"]) - 1)
        latent = data["latents"][index]
        latent_flow = flow_data["flow_latents"][index]
        embeds = data["embedds"][index]
        # assert not torch.isnan(latent.flatten()[0]), f"latent nan, {dpath}"
        # assert not torch.isnan(embeds.flatten()[0]), f"embeds nan, {dpath}"

        first_frame = flow_data["first_frames"][index].squeeze(0)

        return latent, first_frame, embeds, data["masks"][index], data["captions"][index], data["meta_info"][index], latent_flow


class BucketSampler(torch.utils.data.Sampler):
    def __init__(self):
        pass


class MultiDatasetWraper(torch.utils.data.Dataset):
    def __init__(self, datasets):
        super().__init__()
        self.data = datasets
        self.lenlist = [len(dd) for dd in datasets]
        print("self.lenlist", self.lenlist)
        
        
    def __len__(self):
        return sum(self.lenlist)

    def __getitem__(self, idx):
        acc = 0
        for dindex, length in enumerate(self.lenlist):
            # if idx == 0:
            #     return self.data[0][0]

            if idx <= length - 1 + acc and idx >= acc:
                real_index = idx - acc
                return self.data[dindex][real_index]    
            acc += length
            
        print("error!!, this should not be printed", idx)
        return self.data[0][0]

    
    @staticmethod
    def calc_index_for_multi_datasets(list_of_datasets):
        dataset_indices = []
        idxAccumulate = 0
        for dd in list_of_datasets:
            idxrange = [idxAccumulate + ii for ii in range(len(dd))]
            dataset_indices.append(idxrange)
            idxAccumulate += len(dd)
        return dataset_indices

class MixedBatchSampler(torch.utils.data.Sampler):
    def __init__(self, dataset_indices, batch_size):
        self.dataset_indices = dataset_indices
        self.batch_size = batch_size
        self.counts = sum([ len(dd) for dd in self.dataset_indices])

    def __iter__(self):
        batch = []
        num_batch = self.counts // self.batch_size
        for _ in range(num_batch):
            idxs = random.choice(self.dataset_indices)
            batch = random.sample(idxs, self.batch_size)
            # print("batch", batch)
            yield batch
            batch = []
        
    def __len__(self):
        return self.counts // self.batch_size

"""
print("use mixed datasets", self.args.precomputed_datasets)

list_of_datasets = [ PrecomputedDataset(data_root) for data_root in self.args.precomputed_datasets]
dataset_indices = MultiDatasetWraper.calc_index_for_multi_datasets(list_of_datasets)

mixed_batch_sampler = MixedBatchSampler(dataset_indices, batch_size=self.args.batch_size)

self.dataset = MultiDatasetWraper(list_of_datasets)
self.dataloader = torch.utils.data.DataLoader(
    dataset=self.dataset,
    batch_sampler=mixed_batch_sampler,
    # collate_fn=self.model_config.get("collate_fn"),
    num_workers=self.args.dataloader_num_workers,
    pin_memory=self.args.pin_memory,
)


"""