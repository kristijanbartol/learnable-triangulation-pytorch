import numpy as np
import torch
import kornia

from mvn.utils.multiview import create_fundamental_matrix, IDXS, find_rotation_matrices, compare_rotations


IMGS_PATH = 'all_images.npy'
PRED_PATH = 'all_2d_preds.npy'
K_PATH = 'Ks.npy'
R_PATH = 'Rs.npy'
T_PATH = 'ts.npy'
BBOX_PATH = 'all_bboxes.npy'

M = 100             # number of frames
J = 17              # number of joints
P = M * J           # total number of point correspondences    
N = 100000           # trials
eps = 0.6           # outlier probability
S = 30              # sample size
I = (1 - eps) * P   # number of inliers condition


if __name__ == '__main__':
    with torch.no_grad():
        #all_images = np.load(IMGS_PATH)
        all_2d_preds = np.load(PRED_PATH)[:M, IDXS]
        Ks = np.load(K_PATH)
        Rs = np.load(R_PATH)
        ts = np.load(T_PATH)
        bboxes = np.load(BBOX_PATH)[:M, IDXS]

        # Unbbox keypoints.
        # TODO: Update magic number 384.
        bbox_height = np.abs(bboxes[:, :, 0, 0] - bboxes[:, :, 1, 0])
        all_2d_preds *= np.expand_dims(np.expand_dims(bbox_height / 384., axis=-1), axis=-1)
        all_2d_preds += np.expand_dims(bboxes[:, :, 0, :], axis=2)

        point_corresponds = np.concatenate(np.split(all_2d_preds, all_2d_preds.shape[0], axis=0), axis=2)[0]

        Ks = torch.unsqueeze(torch.tensor(Ks, device='cuda', dtype=torch.float32), dim=0)
        Rs = torch.unsqueeze(torch.tensor(Rs, device='cuda', dtype=torch.float32), dim=0)
        ts = torch.unsqueeze(torch.tensor(ts, device='cuda', dtype=torch.float32), dim=0)

        all_2d_preds = torch.tensor(all_2d_preds, device='cuda', dtype=torch.float32)

        counter = 0
        for i in range(N):
            selected_idxs = np.random.choice(np.arange(point_corresponds.shape[1]), size=S)
            selected = torch.tensor(point_corresponds[:, selected_idxs], device='cuda', dtype=torch.float32)

            F_gt = create_fundamental_matrix(Ks, Rs, ts)
            R_est1, R_est2, F = find_rotation_matrices(selected, None, Ks)

            lines = kornia.geometry.compute_correspond_epilines(all_2d_preds[:, 0], F)
            lines_gt = kornia.geometry.compute_correspond_epilines(all_2d_preds[:, 0], F_gt)

            dists = torch.abs(lines[:, :, 0] * all_2d_preds[:, 1, :, 0] + lines[:, :, 1] * all_2d_preds[:, 1, :, 1] + lines[:, :, 2]) \
                / torch.sqrt(lines[:, :, 0] ** 2 + lines[:, :, 1] ** 2)
            dists = dists.view(-1)
            dists_gt = torch.abs(lines_gt[:, :, 0] * all_2d_preds[:, 1, :, 0] + lines_gt[:, :, 1] * all_2d_preds[:, 1, :, 1] + lines_gt[:, :, 2]) \
                / torch.sqrt(lines_gt[:, :, 0] ** 2 + lines_gt[:, :, 1] ** 2)
            dists_gt = dists_gt.view(-1)

            num_inliers = (dists < 1.).sum()
            print(f'Number of inliers: {num_inliers} ({P})')

            if num_inliers > 100:
                counter += 1

                R_sim, m_idx = compare_rotations(Rs, (R_est1, R_est2))
                print(f'Quaternion similarity between the rotations: {R_sim:.5f}')

                # TODO: Estimate camera parameters using all inliers.
                # TODO: Evaluate using 3D GT.

                #kpts_3d_gt = keypoints_3d_batch_gt[batch_index]
                #kpts_2d_projs = evaluate_projection(kpts_3d_gt, Ks[0], Rs[0], ts[0], R_est1[0] if m_idx == 0 else R_est2[0])

                #evaluate_reconstruction(kpts_3d_gt, kpts_2d_projs, Ks[0], Rs[0], ts[0], R_est1[0] if m_idx == 0 else R_est2[0])
