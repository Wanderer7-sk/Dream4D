import os
import numpy as np

import torch
import torchvision


def tensor_to_mp4(video, savepath, fps, rescale=True, nrow=None):
    """
    video: torch.Tensor, b,c,t,h,w, 0-1
    if -1~1, enable rescale=True
    """
    n = video.shape[0]
    video = video.permute(2, 0, 1, 3, 4) # t,n,c,h,w
    nrow = int(np.sqrt(n)) if nrow is None else nrow
    frame_grids = [torchvision.utils.make_grid(framesheet, nrow=nrow, padding=0) for framesheet in video] # [3, grid_h, grid_w]
    grid = torch.stack(frame_grids, dim=0) # stack in temporal dim [T, 3, grid_h, grid_w]
    grid = torch.clamp(grid.float(), -1., 1.)
    if rescale:
        grid = (grid + 1.0) / 2.0
    grid = (grid * 255).to(torch.uint8).permute(0, 2, 3, 1) # [T, 3, grid_h, grid_w] -> [T, grid_h, grid_w, 3]
    torchvision.io.write_video(savepath, grid, fps=fps, video_codec='h264', options={'crf': '10'})
