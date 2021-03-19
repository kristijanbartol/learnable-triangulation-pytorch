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
IMG_PATH  = './data/human36m/processed/S1/Directions-1/imageSequence/54138969/img_000001.jpg'
LABELS_PATH = './data/human36m/extra/human36m-multiview-labels-GTbboxes.npy'

DEVICE='cuda'


if __name__ == '__main__':
    mode = '3d'

    labels = np.load(LABELS_PATH, allow_pickle=True).item()
    with h5py.File(DATA_PATH, "r") as f:
        if mode == '2d':
            uv1 = torch.tensor(f['pose']['2d'][0], device=DEVICE, dtype=torch.float32)
        else:
            X = torch.tensor(labels['table']['keypoints'][0], device=DEVICE, dtype=torch.float32)

    frame = labels['table'][0]
    camera_labels1 = labels['cameras'][frame['subject_idx'], 0]
    camera_labels2 = labels['cameras'][frame['subject_idx'], 1]

    camera1 = Camera(camera_labels1['R'], camera_labels1['t'], camera_labels1['K'], camera_labels1['dist'], '0')
    camera2 = Camera(camera_labels2['R'], camera_labels2['t'], camera_labels2['K'], camera_labels2['dist'], '0')

    P1 = torch.tensor(camera1.projection, device=DEVICE, dtype=torch.float32)
    P2 = torch.tensor(camera2.projection, device=DEVICE, dtype=torch.float32)

    img = Image.open(IMG_PATH)
    ax = plt.subplot()
    ax.imshow(img)

    if mode != '2d':
        X = torch.cat((X, torch.ones((X.shape[0], 1), device=DEVICE)), dim=1)

        uv1 = X @ torch.transpose(P1, 0, 1)
        uv1 = (uv1 / uv1[:, 2].reshape(uv1.shape[0], 1))[:, :2]
        uv2 = X @ torch.transpose(P2, 0, 1)
        uv2 = (uv2 / uv2[:, 2].reshape(uv2.shape[0], 1))[:, :2]

    draw_2d_pose(uv1.cpu(), ax, kind='human36m')

    plt.savefig('fig.png')
