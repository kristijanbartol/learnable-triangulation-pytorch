import numpy as np
import torch
import h5py
import matplotlib.pyplot as plt
from PIL import Image
import time
import random
import cv2
import os

from mvn.utils.img import image_batch_to_numpy, to_numpy, denormalize_image, resize_image
from mvn.utils.vis import draw_2d_pose
from mvn.utils.multiview import Camera, find_rotation_matrices, compare_rotations, create_fundamental_matrix, IDXS

import kornia

DATA_PATH = './data/human36m/processed/S1/Directions-1/annot.h5'
PROJ_PATH = './proj.npy'

IMG_DIR  = './data/human36m/processed/S1/Directions-1/imageSequence/'
CAM1_NAME = '60457274'
CAM2_NAME = '55011271'
IMG1_PATH  = './data/human36m/processed/S1/Directions-1/imageSequence/60457274/img_000001.jpg'
IMG2_PATH  = './data/human36m/processed/S1/Directions-1/imageSequence/55011271/img_000001.jpg'

LABELS_PATH = './data/human36m/extra/human36m-multiview-labels-GTbboxes.npy'

DEVICE='cuda'


def draw_epipolar_lines(points, F, img, name):
    # lines ... (a, b, c) ... ax + by + c = 0
    lines = kornia.geometry.compute_correspond_epilines(points, F)[0].cpu().numpy()

    start_points = np.zeros((points.shape[1], 2), dtype=np.float32)
    end_points = np.zeros((points.shape[1], 2), dtype=np.float32)

    start_points[:, 0] = 0.
    # (a=0) ... y = -c/b
    start_points[:, 1] = -lines[:, 2] / lines[:, 1]
    end_points[:, 0] = img.shape[0]
    # y = -(c + ax) / b
    end_points[:, 1] = -(lines[:, 2] + lines[:, 0] * end_points[:, 0]) / lines[:, 1]

    for p_idx in range(start_points.shape[0]):
        cv2.line(img, tuple(start_points[p_idx]), tuple(end_points[p_idx]), color=(0, 255, 0))

    cv2.imwrite(f'{name}.png', img)

    return lines


if __name__ == '__main__':
    labels = np.load(LABELS_PATH, allow_pickle=True).item()
    #X = torch.tensor(labels['table']['keypoints'][0], device=DEVICE, dtype=torch.float32)

    frame = labels['table'][0]
    camera_labels = labels['cameras'][frame['subject_idx']]
    cameras = [Camera(camera_labels['R'][x], camera_labels['t'][x], camera_labels['K'][x], camera_labels['dist'][x], str(x)) for x in range(4)]

    Ps = torch.stack([torch.tensor(x.projection, device=DEVICE, dtype=torch.float32) for x in cameras], dim=0)
    Ks = torch.stack([torch.tensor(x.K, device=DEVICE, dtype=torch.float32) for x in cameras], dim=0)
    Rs = torch.stack([torch.tensor(x.R, device=DEVICE, dtype=torch.float32) for x in cameras], dim=0)
    ts = torch.stack([torch.tensor(x.t, device=DEVICE, dtype=torch.float32) for x in cameras], dim=0)
    extrs = torch.stack([torch.tensor(x.extrinsics, device=DEVICE, dtype=torch.float32) for x in cameras])

    Ks = torch.unsqueeze(Ks, dim=0)
    Rs = torch.unsqueeze(Rs, dim=0)
    ts = torch.unsqueeze(ts, dim=0)

    #fig, axs = plt.subplots(nrows=3, ncols=2, figsize=(25, 25))

    for img_idx, img_name in enumerate(os.listdir(os.path.join(IMG_DIR, CAM1_NAME))):
        img1 = Image.open(os.path.join(IMG_DIR, CAM1_NAME, img_name))
        img2 = Image.open(os.path.join(IMG_DIR, CAM2_NAME, img_name))

        X = torch.tensor(labels['table']['keypoints'][img_idx], device=DEVICE, dtype=torch.float32)

        #img1 = Image.open(IMG1_PATH)
        #img2 = Image.open(IMG2_PATH)

        '''
        axs[0][0].imshow(img1)
        axs[0][1].imshow(img2)
        axs[1][0].imshow(img1)
        axs[1][1].imshow(img2)
        axs[2][0].imshow(img1)
        axs[2][1].imshow(img2)
        '''

        X = torch.cat((X, torch.ones((X.shape[0], 1), device=DEVICE)), dim=1)

        # NOTE: Can also decompose extrinsics to R|t.
        uv1 = X @ torch.transpose(Ks[0, IDXS[0]] @ extrs[IDXS[0]], 0, 1)
        uv1 = (uv1 / uv1[:, 2].reshape(uv1.shape[0], 1))
        uv2 = X @ torch.transpose(Ks[0, IDXS[1]] @ extrs[IDXS[1]], 0, 1)
        uv2 = (uv2 / uv2[:, 2].reshape(uv2.shape[0], 1))

        # Unhomogenous coordinates.
        uv1 = uv1[:, :2]
        uv2 = uv2[:, :2]

        points = torch.transpose(torch.stack((uv1, uv2), dim=0), 0, 1)

#        draw_2d_pose(points[0].cpu(), axs[0][0], kind='human36m')
#        draw_2d_pose(points[1].cpu(), axs[0][1], kind='human36m')

        with torch.no_grad():
            #points = torch.unsqueeze(points, dim=0)

            F_created = create_fundamental_matrix(Ks, Rs, ts)

            start_time = time.time()
            R_est1, R_est2, F = find_rotation_matrices(points, None, Ks)
            rot_similarity, min_index = compare_rotations(Rs, (R_est1, R_est2))
            R_est = R_est1 if min_index == 0 else R_est2

        R1_est = Rs[0][IDXS[0]]
        t1_est = ts[0][IDXS[0]]
        R_est_rel = R_est[0] @ Rs[0][IDXS[0]]
        t2_est = ts[0][IDXS[1]]

        extr1_est = torch.cat((R1_est, t1_est), dim=1)
        extr2_est = torch.cat((R_est_rel, t2_est), dim=1)

        uv1_est = X @ torch.transpose(Ks[0][IDXS[0]] @ extr1_est, 0, 1)
        uv1_est = (uv1_est / uv1_est[:, 2].reshape(uv1_est.shape[0], 1))[:, :2]
        uv2_est = X @ torch.transpose(Ks[0][IDXS[1]] @ extr2_est, 0, 1)
        uv2_est = (uv2_est / uv2_est[:, 2].reshape(uv2_est.shape[0], 1))[:, :2]
        
        #draw_2d_pose(uv1_est.cpu(), axs[2][0], kind='human36m')
        #draw_2d_pose(uv2_est.cpu(), axs[2][1], kind='human36m')

        uv1_est = torch.unsqueeze(uv1_est, dim=0)
        uv2_est = torch.unsqueeze(uv2_est, dim=0)

        _ = draw_epipolar_lines(uv2_est, torch.transpose(F_created, 1, 2), np.array(img1), 'view1')
        epipolar_lines = draw_epipolar_lines(uv1_est, F_created, np.array(img2), 'view2')

        uv2_est = uv2_est.cpu().numpy()

        dists = np.abs(epipolar_lines[:, 0] * uv2_est[0, :, 0] + epipolar_lines[:, 1] * uv2_est[0, :, 1] + epipolar_lines[:, 2]) \
            / np.sqrt(epipolar_lines[:, 0] ** 2 + epipolar_lines[:, 1] ** 2)

        print(dists.sum())

        #plt.savefig('fig.png')
