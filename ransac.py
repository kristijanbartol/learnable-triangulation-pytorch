import numpy as np
import torch
import kornia
import cv2
import time
import itertools
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random

from mvn.utils.multiview import create_fundamental_matrix, IDXS, find_rotation_matrices, compare_rotations, \
    evaluate_projection, evaluate_reconstruction, distance_between_projections
from mvn.utils.vis import draw_2d_pose_cv2, draw_images, draw_epipolar_lines
from mvn.utils.img import denormalize_image, image_batch_to_numpy


IMGS_PATH = 'all_images.npy'
PRED_PATH = 'all_2d_preds.npy'
GT_PATH = 'all_3d_gt.npy'
KS_BBOX_PATH = 'Ks_bboxed.npy'
K_PATH = 'Ks.npy'
R_PATH = 'Rs.npy'
T_PATH = 'ts.npy'
BBOX_PATH = 'all_bboxes.npy'

M = 50             # number of frames
J = 17              # number of joints
P = M * J           # total number of point correspondences    
N = 100           # trials
eps = 0.75           # outlier probability
S = 100              # sample size
#I = (1 - eps) * P  # number of inliers condition
I = 20
D = .5              # distance criterion


if __name__ == '__main__':
    print(f'Number of inliers condition: {I} ({P})')

    with torch.no_grad():
        #all_images = np.load(IMGS_PATH)[:M]
        all_2d_preds = np.load(PRED_PATH)
        all_3d_gt = torch.tensor(np.load(GT_PATH), device='cuda', dtype=torch.float32)
        #Ks_bboxed = torch.tensor(np.load(KS_BBOX_PATH), device='cuda', dtype=torch.float32)
        Ks = torch.unsqueeze(torch.tensor(np.load(K_PATH), device='cuda', dtype=torch.float32), dim=0)[:, IDXS]
        Rs = torch.unsqueeze(torch.tensor(np.load(R_PATH), device='cuda', dtype=torch.float32), dim=0)[:, IDXS]
        ts = torch.unsqueeze(torch.tensor(np.load(T_PATH), device='cuda', dtype=torch.float32), dim=0)[:, IDXS]
        bboxes = np.load(BBOX_PATH)

        frame_selection = np.random.choice(np.arange(all_2d_preds.shape[0]), size=M)
        all_2d_preds = all_2d_preds[frame_selection][:, IDXS]
        all_3d_gt = all_3d_gt[frame_selection]
        #Ks_bboxed = Ks_bboxed[frame_selection]
        bboxes = bboxes[frame_selection][:, IDXS]

        #all_right_imgs = image_batch_to_numpy(all_images[:, 1])
        #img = denormalize_image(all_right_imgs[0]).astype(np.uint8)
        #img = img[..., ::-1]  # bgr -> rgb

        #all_2d_preds_bboxed = torch.tensor(all_2d_preds, device='cuda', dtype=torch.float32)

        # Unbbox keypoints.
        # TODO: Update magic number 384.
        # TODO: Write in Torch.
        bbox_height = np.abs(bboxes[:, :, 0, 0] - bboxes[:, :, 1, 0])
        all_2d_preds *= np.expand_dims(np.expand_dims(bbox_height / 384., axis=-1), axis=-1)
        all_2d_preds += np.expand_dims(bboxes[:, :, 0, :], axis=2)

        # All points stacked along a single dimension.
        point_corresponds = torch.tensor(np.concatenate(np.split(all_2d_preds, all_2d_preds.shape[0], axis=0), axis=2)[0], 
            device='cuda', dtype=torch.float32).transpose(0, 1)

        all_2d_preds = torch.tensor(all_2d_preds, device='cuda', dtype=torch.float32)

        ########### GT ###########
        dists = distance_between_projections(point_corresponds[:, 0], point_corresponds[:, 1], Ks[0], Rs[0], ts[0])
        condition = dists < 1.
        num_inliers = (condition).sum()
        print(f'Number of inliers (GT): {num_inliers} ({P})')
        print(f'Mean distances between corresponding lines (GT): {dists.mean()}')

        def evaluate(condition):
            # Evaluate fundamental matrix.
            #inliers_gt_refine = point_corresponds[condition_gt_initial][:S]
            inliers = point_corresponds[condition]
            R_gt1, R_gt2, _ = find_rotation_matrices(inliers, None, Ks)
            R_sim, m_idx = compare_rotations(Rs, (R_gt1, R_gt2))

            kpts_2d_projs, _ = evaluate_projection(all_3d_gt, Ks[0], Rs[0], ts[0], R_gt1[0] if m_idx == 0 else R_gt2[0])
            error_3d = evaluate_reconstruction(all_3d_gt, kpts_2d_projs, Ks[0], Rs[0], ts[0], R_gt1[0] if m_idx == 0 else R_gt2[0])

            return R_sim, error_3d

        _, error_3d = evaluate(condition)
        print(f'3D error (GT): {error_3d}')
        #####################################################################

        counter = 0
        line_dist_error_pairs = torch.empty((0, 4), device='cuda', dtype=torch.float32)
        for i in range(N):
            selected_idxs = torch.tensor(np.random.choice(np.arange(point_corresponds.shape[0]), size=S), device='cuda')

            R_initial1, R_initial2, _ = find_rotation_matrices(point_corresponds[selected_idxs], None, Ks)
            R_sim, m_idx = compare_rotations(Rs, (R_initial1, R_initial2))

            R2_est = ((R_initial1 if m_idx == 0 else R_initial2) @ Rs[0, 0])[0]

            line_dists_initial = distance_between_projections(
                    point_corresponds[:, 0], point_corresponds[:, 1], 
                    Ks[0], torch.stack((Rs[0, 0], R2_est), dim=0), ts[0])

            condition_initial = line_dists_initial < 1.
            num_inliers_initial = (condition_initial).sum()

            if num_inliers_initial > I:
                # Evaluate 2D projections and 3D reprojections (triangulation).
                kpts_2d_projs, error_2d = evaluate_projection(all_3d_gt, Ks[0], Rs[0], ts[0], R_initial1[0] if m_idx == 0 else R_initial2[0])
                error_3d = evaluate_reconstruction(all_3d_gt, kpts_2d_projs, Ks[0], Rs[0], ts[0], R_initial1[0] if m_idx == 0 else R_initial2[0])

                line_dists_inlier = distance_between_projections(
                    point_corresponds[condition_initial][:, 0], point_corresponds[condition_initial][:, 1], Ks[0], 
                    torch.stack((Rs[0, 0], R2_est), dim=0), ts[0])
                line_dists_all = distance_between_projections(
                    point_corresponds[:, 0], point_corresponds[:, 1], Ks[0], 
                    torch.stack((Rs[0, 0], R2_est), dim=0), ts[0])
                
                line_dist_error_pair = torch.unsqueeze(torch.cat(
                    (torch.unsqueeze(num_inliers_initial, dim=0), 
                    torch.unsqueeze(line_dists_inlier.mean(), dim=0), 
                    torch.unsqueeze(line_dists_all.mean(), dim=0),
                    torch.unsqueeze(error_3d, dim=0)), dim=0), dim=0)
                line_dist_error_pairs = torch.cat((line_dist_error_pairs, line_dist_error_pair), dim=0)
                #print(f'{counter}. ({quaternion} -> ({line_dists_inlier.mean():.3f}, {R_sim:.2f}, {error_3d:.2f})')
                print(f'{counter}. ({num_inliers_initial}, {line_dists_inlier.mean():.3f}, {line_dists_all.mean():.3f}) -> {error_3d:.2f}')

                counter += 1

        print(f'Estimated best (num inliers): {line_dist_error_pairs[line_dist_error_pairs[:, 0].argmax()]}')
        print(f'Estimated best (inlier distances): {line_dist_error_pairs[line_dist_error_pairs[:, 1].argmin()]}')
        print(f'Estimated best (all distances): {line_dist_error_pairs[line_dist_error_pairs[:, 2].argmin()]}')
        print(f'Actual best: {line_dist_error_pairs[line_dist_error_pairs[:, 3].argmin()]}')

        '''
        quat_error_pairs_np = quat_error_pairs.cpu().numpy()
        np.save('quat_error_pairs.npy', quat_error_pairs_np)

        quat_error_median = np.median(quat_error_pairs_np[:, :4], axis=0)
        median_idx = np.linalg.norm(quat_error_pairs_np[:, :4] - quat_error_median, axis=1).argmin()
        print(median_idx)
        print(quat_error_pairs_np[median_idx])

        quat_error_mean = np.mean(quat_error_pairs_np[:, :4], axis=0)
        mean_idx = np.linalg.norm(quat_error_pairs_np[:, :4] - quat_error_mean, axis=1).argmin()
        print(mean_idx)
        print(quat_error_pairs_np[mean_idx])

        camera_inliers = quat_error_pairs_np[:, 4] < 20.
        combinations2d = list(itertools.combinations(range(4), 2))

        for idx1, idx2 in combinations2d:
            plt.clf()
            plt.scatter(quat_error_pairs_np[~camera_inliers, idx1], quat_error_pairs_np[~camera_inliers, idx2], c='blue')
            plt.scatter(quat_error_pairs_np[camera_inliers, idx1], quat_error_pairs_np[camera_inliers, idx2], c='red')
            plt.scatter(quat_error_pairs_np[median_idx, idx1], quat_error_pairs_np[median_idx, idx2], c='green')
            plt.scatter(quat_error_pairs_np[mean_idx, idx1], quat_error_pairs_np[mean_idx, idx2], c='yellow')
            plt.savefig(f'quat_distro_{idx1}{idx2}.png')

        combinations3d = list(itertools.combinations(range(4), 3))

        fig = plt.figure()
        ax = Axes3D(fig)

        for idx1, idx2, idx3 in combinations3d:
            ax.cla()
            ax.scatter(quat_error_pairs_np[~camera_inliers, idx1], quat_error_pairs_np[~camera_inliers, idx2], \
                quat_error_pairs_np[~camera_inliers, idx3], c='blue')
            ax.scatter(quat_error_pairs_np[camera_inliers, idx1], quat_error_pairs_np[camera_inliers, idx2], \
                quat_error_pairs_np[camera_inliers, idx3], c='red')
            plt.savefig(f'quat_distro_{idx1}{idx2}{idx3}.png')

        print(f'Number of camera inliers (total): {np.sum(camera_inliers)}')
        '''
