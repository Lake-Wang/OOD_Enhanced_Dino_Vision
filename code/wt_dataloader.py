from bisect import bisect
import gc
import glob
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

# from line_profiler import profile
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

class WalkingToursDataset(Dataset):
    def __init__(self,
                 data_dir: str,
                 transform: Callable,
                 mode: str = "train",
                 repeat_sample: int = None,
                 chunk_ratio: int = 4,
                 backend='torchvision-videoreader',
                 ):
        assert mode in ["train", "val", "test"]
        
        self.data_dir = data_dir
        self.transform = transform
        self.mode = mode
        self.repeat_sample = repeat_sample
        self.backend = backend
        
        assert backend in ['decord', 'torchvision', 'torchvision-videoreader'], f"backend must be one of ['decord', 'torchvision', 'torchvision-videoreader'], got {backend}"
        if backend == 'torchvision-videoreader':
            torchvision.set_video_backend('video_reader')
        else:
            torchvision.set_video_backend('pyav')

        video_paths = glob.glob(os.path.join(data_dir, '**', "*.mp4"), recursive=True)

        self.video_paths = video_paths
        self.video_idx_start = []
        self.video_md = []
        self.video_rotations = []
        self.sample_indices = []
        idx = 0
        for video_path in video_paths:
            _, vr_len, vr_md, vr_rotation = self.create_video_reader(video_path, 0)
            self.video_frames = vr_len
            num_samples = (vr_len // (chunk_ratio * repeat_sample)) * chunk_ratio
            self.video_idx_start.append(idx)
            idx += num_samples
            self.video_md.append(vr_md)
            self.video_rotations.append(vr_rotation)
            chunks = np.split(np.arange((num_samples // chunk_ratio) * chunk_ratio * repeat_sample), num_samples // chunk_ratio)
            for chunk in chunks:
                self.sample_indices.extend(np.split(np.random.permutation(chunk), chunk_ratio))
        
        self._dataset_len = idx
    
    def create_video_reader(self, video_path, cpuid):
        if self.backend == 'decord':
            vr = VideoReader(video_path, num_threads=0, ctx=cpu(cpuid))
            vr_len = len(vr)
            vr_md = None
            vr_rotation = None
        elif 'torchvision' in self.backend:
            vr = TVVideoReader(video_path, "video")
            # conda-base FFMPEG does not preserve rotations properly, must read manually
            try:
                vr_rotation = -int(ffprobe(video_path).json['streams'][0]['side_data_list'][0].get('rotation', '0'))
            except:
                vr_rotation = 0
            vr_md = vr.get_metadata()['video']
            vr_len = int(vr_md['duration'][0] * vr_md['fps'][0]) - 1
        return vr, vr_len, vr_md, vr_rotation

    def __len__(self):
        return self._dataset_len
    
    # @profile
    def __getitem__(self, idx):
        worker_info = torch.utils.data.get_worker_info()
        cpuid = 0 if worker_info == None else int(get_rank() * worker_info.num_workers + (worker_info.id))
        video_idx = bisect(self.video_idx_start, idx) - 1
        vr, _, vr_md, vr_rotation = self.create_video_reader(self.video_paths[video_idx], cpuid)
        i_s = self.sample_indices[idx]
        sort_indexes = np.argsort(i_s).astype(np.int32)
        unsort_indexes = np.argsort(sort_indexes).astype(np.int32)
        if self.backend == 'decord':
            imgs = vr.get_batch(list(i_s[sort_indexes])).asnumpy()[unsort_indexes]
            vr.seek(0)
        elif 'torchvision' in self.backend:
            vr.seek(0)
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
                            log.warning(f"Failed to read frame for 3rd iteration, resorting to keyframe from {i_-1/vr_md['fps'][0]}")
                            vr.seek(i_-1/vr_md['fps'][0], keyframes_only=True)
                            decode_res = next(vr)
                            log.warn(f"Keyframe from {i_} read succesfully at {decode_res['pts']}")
                            res.append(decode_res['data'])
                            break
            imgs = torch.stack(res, axis=0)
            if vr_rotation != 0:
                imgs = torch.rot90(imgs, k=-vr_rotation//90, dims=[2, 3])
            imgs = imgs.permute(0, 2, 3, 1).numpy()[unsort_indexes]
        del vr; gc.collect()

        ls = []
        for i in range(self.repeat_sample):
            img = tvF.to_pil_image(imgs[i])
            aug_img = self.transform(img)
            ls = [[] for _ in range(len(aug_img))]
            for j in range(len(aug_img)):
                ls[j].append(aug_img[j])

        return [torch.stack(l, dim=0) for l in ls]

