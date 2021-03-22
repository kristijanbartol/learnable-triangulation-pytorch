import numpy as np
import torch
import h5py
import matplotlib.pyplot as plt
from PIL import Image
import time

from mvn.utils.img import image_batch_to_numpy, to_numpy, denormalize_image, resize_image
from mvn.utils.vis import draw_2d_pose
from mvn.utils.multiview import Camera

import kornia

DATA_PATH = './data/human36m/processed/S1/Directions-1/annot.h5'
PROJ_PATH = './proj.npy'
IMG1_PATH  = './data/human36m/processed/S1/Directions-1/imageSequence/60457274/img_000001.jpg'
IMG2_PATH  = './data/human36m/processed/S1/Directions-1/imageSequence/55011271/img_000001.jpg'
LABELS_PATH = './data/human36m/extra/human36m-multiview-labels-GTbboxes.npy'

DEVICE='cuda'


def find_rotation_matrices(points1, points2, K1, K2):
    conf = torch.ones(points1.shape[:2], device=points1.device)
    conf[0][5] = 0.01
    conf[0][:9] = 0.
    F = kornia.find_fundamental(points1, points2, conf)
    E = kornia.essential_from_fundamental(F, K1, K2)
    R1, R2, t = kornia.decompose_essential_matrix(E)

    return R1, R2


def compare_rotations(rot_mat1, rot_mat2, est_rot):
    rel_rot = rot_mat2 @ torch.inverse(rot_mat1)

    rel_rot_quat = kornia.geometry.rotation_matrix_to_quaternion(rel_rot)
    est_rot1_quat = kornia.geometry.rotation_matrix_to_quaternion(est_rot[0])
    est_rot2_quat = kornia.geometry.rotation_matrix_to_quaternion(est_rot[1])

    rot_mat2_quat = kornia.geometry.rotation_matrix_to_quaternion(rot_mat2)

    diff1 = torch.norm(est_rot1_quat - rel_rot_quat)
    diff2 = torch.norm(est_rot2_quat - rel_rot_quat)

    return min(diff1, diff2), np.argmin([diff1, diff2])


if __name__ == '__main__':
    labels = np.load(LABELS_PATH, allow_pickle=True).item()
    X = torch.tensor(labels['table']['keypoints'][0], device=DEVICE, dtype=torch.float32)

    frame = labels['table'][0]
    camera_labels1 = labels['cameras'][frame['subject_idx'], 3]
    camera_labels2 = labels['cameras'][frame['subject_idx'], 1]

    camera1 = Camera(camera_labels1['R'], camera_labels1['t'], camera_labels1['K'], camera_labels1['dist'], '3')
    camera2 = Camera(camera_labels2['R'], camera_labels2['t'], camera_labels2['K'], camera_labels2['dist'], '1')

    P1 = torch.tensor(camera1.projection, device=DEVICE, dtype=torch.float32)
    P2 = torch.tensor(camera2.projection, device=DEVICE, dtype=torch.float32)

    R1 = torch.tensor(camera1.R, device=DEVICE, dtype=torch.float32)
    R2 = torch.tensor(camera2.R, device=DEVICE, dtype=torch.float32)

    t1 = torch.tensor(camera1.t, device=DEVICE, dtype=torch.float32)
    t2 = torch.tensor(camera2.t, device=DEVICE, dtype=torch.float32)

    extr1 = torch.tensor(camera1.extrinsics, device=DEVICE, dtype=torch.float32)
    extr2 = torch.tensor(camera2.extrinsics, device=DEVICE, dtype=torch.float32)

    K1 = torch.tensor(camera1.K, device=DEVICE, dtype=torch.float32)
    K2 = torch.tensor(camera2.K, device=DEVICE, dtype=torch.float32)

    img1 = Image.open(IMG1_PATH)
    img2 = Image.open(IMG2_PATH)

    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(25, 25))

    axs[0][0].imshow(img1)
    axs[0][1].imshow(img2)
    #axs[1][0].imshow(img1)
    #axs[1][1].imshow(img2)
    axs[1][0].imshow(img1)
    axs[1][1].imshow(img2)

    X = torch.cat((X, torch.ones((X.shape[0], 1), device=DEVICE)), dim=1)

    # NOTE: Can also decompose extrinsics to R|t.
    uv1 = X @ torch.transpose(K1 @ extr1, 0, 1)
    uv1 = (uv1 / uv1[:, 2].reshape(uv1.shape[0], 1))[:, :2]
    uv2 = X @ torch.transpose(K2 @ extr2, 0, 1)
    uv2 = (uv2 / uv2[:, 2].reshape(uv2.shape[0], 1))[:, :2]

    draw_2d_pose(uv1.cpu(), axs[0][0], kind='human36m')
    draw_2d_pose(uv2.cpu(), axs[0][1], kind='human36m')

    uv1[5] += 15.

    #draw_2d_pose(uv1.cpu(), axs[1][0], kind='human36m')
    #draw_2d_pose(uv2.cpu(), axs[1][1], kind='human36m')

    with torch.no_grad():
        uv1 = torch.unsqueeze(uv1, dim=0)
        uv2 = torch.unsqueeze(uv2, dim=0)

        K1 = torch.unsqueeze(K1, dim=0)
        K2 = torch.unsqueeze(K2, dim=0)

        start_time = time.time()
        R_est = find_rotation_matrices(uv1, uv2, K1, K2)
        rot_similarity, min_index = compare_rotations(R1, R2, R_est)
        print(rot_similarity)
        print(f'estimation time: {time.time() - start_time}')

    R1_est = R1
    t1_est = t1
    R2_est = R_est[min_index][0] @ R1
    t2_est = t2

    extr1_est = torch.cat((R1_est, t1_est), dim=1)
    extr2_est = torch.cat((R2_est, t2_est), dim=1)

    uv1 = X @ torch.transpose(K1[0] @ extr1_est, 0, 1)
    uv1 = (uv1 / uv1[:, 2].reshape(uv1.shape[0], 1))[:, :2]
    uv2 = X @ torch.transpose(K2[0] @ extr2_est, 0, 1)
    uv2 = (uv2 / uv2[:, 2].reshape(uv2.shape[0], 1))[:, :2]
    
    draw_2d_pose(uv1.cpu(), axs[1][0], kind='human36m')
    draw_2d_pose(uv2.cpu(), axs[1][1], kind='human36m')

    plt.savefig('fig.png')
