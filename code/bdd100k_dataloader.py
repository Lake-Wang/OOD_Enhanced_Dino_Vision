import gc
import json
import os
import subprocess
from typing import Callable, List, NamedTuple

import numpy as np
import torch
import torch.distributed as dist
import torchvision
import torchvision.transforms.functional as tvF
from decord import VideoReader, cpu
from torch.utils import data as torchdata
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.io import VideoReader as TVVideoReader
from torchvision.transforms import InterpolationMode

import logging 
log = logging.getLogger(__name__)


def get_rank():
    if not dist.is_available() or not dist.is_initialized():
        return 0 
    else:
        return dist.get_rank()

class FFProbeResult(NamedTuple):
    return_code: int
    json: str
    error: str


def ffprobe(file_path) -> FFProbeResult:
    command_array = ["ffprobe",
                     "-v", "quiet",
                     "-print_format", "json",
                     "-show_format",
                     "-show_streams",
                     file_path]
    result = subprocess.run(command_array, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
    return FFProbeResult(return_code=result.returncode,
                         json=json.loads(result.stdout),
                         error=result.stderr)

class BDD100KDataset(Dataset):
    def __init__(self,
                 data_dir: str,
                 transform: Callable,
                 mode: str = "train",
                 meta_info_file: str = None,
                 repeat_sample: int = None,
                 backend='decord',
                 ):
        
        assert mode in ["train", "val", "test"]
        
        self.data_dir = data_dir
        self.mode = mode
        self.backend = backend
        self.repeat_sample = repeat_sample or 1
        
        assert backend in ['decord', 'torchvision', 'torchvision-videoreader'], f"backend must be one of ['decord', 'torchvision', 'torchvision-videoreader'], got {backend}"
        if backend == 'torchvision-videoreader':
            torchvision.set_video_backend('video_reader')
        else:
            torchvision.set_video_backend('pyav')

        video_dir = os.path.join(data_dir, mode)
        video_paths = [os.path.join(video_dir, f) for f in os.listdir(video_dir)]
        if meta_info_file is not None:
            meta_info = np.load(meta_info_file, allow_pickle=True).item()
            self.video_paths = []
            self.video_metadata = []
            for p in video_paths:
                info = meta_info.get(p)
                if info is None:
                    continue
                length = info.get('length', 0)
                if length < self.repeat_sample:
                    continue
                self.video_paths.append(p)
                self.video_metadata.append(info)
        else:
            self.video_paths = video_paths
            self.video_metadata = [None for _ in video_paths]
        self._dataset_len = len(self.video_paths)
        self.transform = transform
       
    def __len__(self):
        return self._dataset_len
    
    def __getitem__(self, idx):
        worker_info = torch.utils.data.get_worker_info()
        cpuid = 0 if worker_info == None else int(get_rank() * worker_info.num_workers + (worker_info.id))
        if self.backend == 'decord':
            vr = VideoReader(self.video_paths[idx], num_threads=0, ctx=cpu(cpuid))
            vr_len = len(vr)
        elif 'torchvision' in self.backend:
            vr = TVVideoReader(self.video_paths[idx], "video")
            # conda-base FFMPEG does not preserve rotations properly, must read manually
            if self.video_metadata[idx] is not None and 'rotation' in self.video_metadata[idx]:
                vr_rotation = int(self.video_metadata[idx]['rotation'])
            else:
                try:
                    vr_rotation = -int(ffprobe(self.video_paths[idx]).json['streams'][0]['side_data_list'][0].get('rotation', '0'))
                except:
                    vr_rotation = 0
            vr_md = vr.get_metadata()['video']
            vr_len = int(vr_md['duration'][0] * vr_md['fps'][0]) - 1

        i_s = np.random.randint(0, vr_len, size=self.repeat_sample)
        sort_indexes = np.argsort(i_s).astype(np.int32)
        unsort_indexes = np.argsort(sort_indexes).astype(np.int32)

        if self.backend == 'decord':
            imgs = vr.get_batch(list(i_s[sort_indexes])).asnumpy()[unsort_indexes]
        elif 'torchvision' in self.backend:
            res = []
            i_s_ = [x / vr_md['fps'][0] for x in i_s[sort_indexes]]
            for i_ in i_s_:
                vr.seek(i_)
                count = 0
                while True:
                    try:
                        res.append(next(vr)['data'])
                        break
                    except StopIteration:
                        log.warning(f"StopIteration at {i_ + (count * 1/vr_md['fps'][0])} for {self.video_paths[idx]}")
                        count += 1
                        if count < 3:
                            vr.seek(i_ + count * 1/vr_md['fps'][0])
                        else:
                            log.warning(f"Failed to read frame for 3rd iteration, resorting to keyframe from {i_-(count-2)/vr_md['fps'][0]}")
                            vr.seek(i_-(count-2)/vr_md['fps'][0], keyframes_only=True)
            imgs = torch.stack(res, axis=0)
            if vr_rotation != 0:
                imgs = torch.rot90(imgs, k=-vr_rotation//90, dims=[2, 3])
            imgs = imgs.permute(0, 2, 3, 1).numpy()[unsort_indexes]
        del vr; gc.collect()

        ls = []
        for i in range(self.repeat_sample):
            img = tvF.to_pil_image(imgs[i])
            aug_img = self.transform(img)
            if len(ls) == 0:
                ls = [[] for _ in range(len(aug_img))]
            for j in range(len(aug_img)):
                ls[j].append(aug_img[j])

        return [torch.stack(l, dim=0) for l in ls]
