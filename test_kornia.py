import numpy as np
import torch
import h5py
import matplotlib.pyplot as plt
from PIL import Image

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

    img1 = Image.open(IMG1_PATH)
    img2 = Image.open(IMG2_PATH)

    fig, axs = plt.subplots(2)

    axs[0].imshow(img1)
    axs[1].imshow(img2)

    X = torch.cat((X, torch.ones((X.shape[0], 1), device=DEVICE)), dim=1)

    uv1 = X @ torch.transpose(P1, 0, 1)
    uv1 = (uv1 / uv1[:, 2].reshape(uv1.shape[0], 1))[:, :2]
    uv2 = X @ torch.transpose(P2, 0, 1)
    uv2 = (uv2 / uv2[:, 2].reshape(uv2.shape[0], 1))[:, :2]

    draw_2d_pose(uv1.cpu(), axs[0], kind='human36m')
    draw_2d_pose(uv2.cpu(), axs[1], kind='human36m')

    plt.savefig('fig.png')
