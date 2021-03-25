import numpy as np
import torch
import h5py
import matplotlib.pyplot as plt
from PIL import Image
import time
import random

from mvn.utils.img import image_batch_to_numpy, to_numpy, denormalize_image, resize_image
from mvn.utils.vis import draw_2d_pose
from mvn.utils.multiview import Camera, find_rotation_matrices, compare_rotations, IDXS

import kornia

DATA_PATH = './data/human36m/processed/S1/Directions-1/annot.h5'
PROJ_PATH = './proj.npy'
IMG1_PATH  = './data/human36m/processed/S1/Directions-1/imageSequence/60457274/img_000001.jpg'
IMG2_PATH  = './data/human36m/processed/S1/Directions-1/imageSequence/55011271/img_000001.jpg'
LABELS_PATH = './data/human36m/extra/human36m-multiview-labels-GTbboxes.npy'

DEVICE='cuda'


if __name__ == '__main__':
    #while True:
    labels = np.load(LABELS_PATH, allow_pickle=True).item()
    X = torch.tensor(labels['table']['keypoints'][0], device=DEVICE, dtype=torch.float32)

    frame = labels['table'][0]
    camera_labels = labels['cameras'][frame['subject_idx']]
    cameras = [Camera(camera_labels['R'][x], camera_labels['t'][x], camera_labels['K'][x], camera_labels['dist'][x], str(x)) for x in range(4)]

    Ps = torch.stack([torch.tensor(x.projection, device=DEVICE, dtype=torch.float32) for x in cameras], dim=0)
    Ks = torch.stack([torch.tensor(x.K, device=DEVICE, dtype=torch.float32) for x in cameras], dim=0)
    Rs = torch.stack([torch.tensor(x.R, device=DEVICE, dtype=torch.float32) for x in cameras], dim=0)
    ts = torch.stack([torch.tensor(x.t, device=DEVICE, dtype=torch.float32) for x in cameras], dim=0)
    extrs = torch.stack([torch.tensor(x.extrinsics, device=DEVICE, dtype=torch.float32) for x in cameras])

    img1 = Image.open(IMG1_PATH)
    img2 = Image.open(IMG2_PATH)

    fig, axs = plt.subplots(nrows=3, ncols=2, figsize=(25, 25))

    
    axs[0][0].imshow(img1)
    axs[0][1].imshow(img2)
    axs[1][0].imshow(img1)
    axs[1][1].imshow(img2)
    axs[2][0].imshow(img1)
    axs[2][1].imshow(img2)
    

    X = torch.cat((X, torch.ones((X.shape[0], 1), device=DEVICE)), dim=1)

    # NOTE: Can also decompose extrinsics to R|t.
    uv1 = X @ torch.transpose(Ks[IDXS[0]] @ extrs[IDXS[0]], 0, 1)
    uv1 = (uv1 / uv1[:, 2].reshape(uv1.shape[0], 1))[:, :2]
    uv2 = X @ torch.transpose(Ks[IDXS[1]] @ extrs[IDXS[1]], 0, 1)
    uv2 = (uv2 / uv2[:, 2].reshape(uv2.shape[0], 1))[:, :2]

    points = torch.stack((uv1, uv2), dim=0)

    draw_2d_pose(points[0].cpu(), axs[0][0], kind='human36m')
    draw_2d_pose(points[1].cpu(), axs[0][1], kind='human36m')

    total_error = 0.
    for p_idx in range(points.shape[1] - 5):
        noise1 = (1. if torch.rand(1) < 0.5 else -1.) * torch.normal(mean=torch.tensor(0.), std=torch.tensor(0.5))
        #noise2 = (1. if torch.rand(1) < 0.5 else -1.) * torch.normal(mean=torch.tensor(0.), std=torch.tensor(1.))
        #noise1 = torch.clamp(noise1, -0.5, 0.5)
        #noise2 = torch.clamp(noise2, -0.005, 0.005)
        total_error += torch.abs(noise1) + torch.abs(noise1)
        points[0][p_idx][0] += noise1
        points[0][p_idx][1] += noise1

    draw_2d_pose(points[0].cpu(), axs[1][0], kind='human36m')
    draw_2d_pose(points[1].cpu(), axs[1][1], kind='human36m')

    with torch.no_grad():
        Ks = torch.unsqueeze(Ks, dim=0)
        Rs = torch.unsqueeze(Rs, dim=0)

        start_time = time.time()
        R_est = find_rotation_matrices(points, None, Ks)
        rot_similarity, min_index = compare_rotations(Rs, R_est)

    R1_est = Rs[0][IDXS[0]]
    t1_est = ts[IDXS[0]]
    R2_est = R_est[min_index][0] @ Rs[0][IDXS[0]]
    #R2_est = Rs[0][IDXS[1]]
    t2_est = ts[IDXS[1]]

    extr1_est = torch.cat((R1_est, t1_est), dim=1)
    extr2_est = torch.cat((R2_est, t2_est), dim=1)

    uv1_est = X @ torch.transpose(Ks[0][IDXS[0]] @ extr1_est, 0, 1)
    uv1_est = (uv1_est / uv1_est[:, 2].reshape(uv1_est.shape[0], 1))[:, :2]
    uv2_est = X @ torch.transpose(Ks[0][IDXS[1]] @ extr2_est, 0, 1)
    uv2_est = (uv2_est / uv2_est[:, 2].reshape(uv2_est.shape[0], 1))[:, :2]
    
    draw_2d_pose(uv1_est.cpu(), axs[2][0], kind='human36m')
    draw_2d_pose(uv2_est.cpu(), axs[2][1], kind='human36m')

    plt.savefig('fig.png')

    print(f'({total_error}, {torch.sum(torch.abs(uv2 - uv2_est))}, {rot_similarity})')
