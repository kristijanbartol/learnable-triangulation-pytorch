import numpy as np
import torch
import kornia
import cv2
import time
import itertools
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from mvn.utils.multiview import create_fundamental_matrix, IDXS, find_rotation_matrices, compare_rotations, \
    evaluate_projection, evaluate_reconstruction
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

M = 15             # number of frames
J = 17              # number of joints
P = M * J           # total number of point correspondences    
N = 500           # trials
eps = 0.75           # outlier probability
S = 50              # sample size
#I = (1 - eps) * P  # number of inliers condition
I = 0
D = .5              # distance criterion


if __name__ == '__main__':
    print(f'Number of inliers condition: {I} ({P})')

    with torch.no_grad():
        #all_images = np.load(IMGS_PATH)[:M]
        all_2d_preds = np.load(PRED_PATH)
        all_3d_gt = torch.tensor(np.load(GT_PATH), device='cuda', dtype=torch.float32)
        #Ks_bboxed = torch.tensor(np.load(KS_BBOX_PATH), device='cuda', dtype=torch.float32)
        Ks = torch.unsqueeze(torch.tensor(np.load(K_PATH), device='cuda', dtype=torch.float32), dim=0)
        Rs = torch.unsqueeze(torch.tensor(np.load(R_PATH), device='cuda', dtype=torch.float32), dim=0)
        ts = torch.unsqueeze(torch.tensor(np.load(T_PATH), device='cuda', dtype=torch.float32), dim=0)
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
        # Using unbboxed Ks to calculate all distances.
        F_gt_initial = create_fundamental_matrix(Ks, Rs, ts)
        lines_gt_initial = kornia.geometry.compute_correspond_epilines(torch.unsqueeze(point_corresponds[:, 0], dim=0), F_gt_initial)[0]
        dists_gt_initial = torch.abs(lines_gt_initial[:, 0] * point_corresponds[:, 1, 0] + \
            lines_gt_initial[:, 1] * point_corresponds[:, 1, 1] + lines_gt_initial[:, 2]) \
            / torch.sqrt(lines_gt_initial[:, 0] ** 2 + lines_gt_initial[:, 1] ** 2)
        dists_gt_initial = dists_gt_initial.view(-1)

        condition_gt_initial = dists_gt_initial < D
        num_inliers_gt_initial = (condition_gt_initial).sum()
        print(f'Number of inliers (initial GT): {num_inliers_gt_initial} ({P})')

        # Evaluate fundamental matrix.
        #inliers_gt_refine = point_corresponds[condition_gt_initial][:S]
        inliers_gt_refine = point_corresponds[condition_gt_initial]
        R_gt_refine1, R_gt_refine2, F_gt_refine = find_rotation_matrices(inliers_gt_refine, None, Ks)
        R_sim, m_idx = compare_rotations(Rs, (R_gt_refine1, R_gt_refine2))
        
        lines_gt_refine = kornia.geometry.compute_correspond_epilines(torch.unsqueeze(point_corresponds[:, 0], dim=0), F_gt_refine)[0]
        dists_gt_refine = torch.abs(lines_gt_refine[:, 0] * point_corresponds[:, 1, 0] + \
            lines_gt_refine[:, 1] * point_corresponds[:, 1, 1] + lines_gt_refine[:, 2]) \
            / torch.sqrt(lines_gt_refine[:, 0] ** 2 + lines_gt_refine[:, 1] ** 2)
        dists_gt_refine = dists_gt_refine.view(-1)

        condition_gt_refine = dists_gt_refine < D
        num_inliers_gt_refine = (condition_gt_refine).sum()
        print(f'Number of inliers (refine GT): {num_inliers_gt_refine} ({P})')

        kpts_2d_projs = evaluate_projection(all_3d_gt, Ks[0], Rs[0], ts[0], R_gt_refine1[0] if m_idx == 0 else R_gt_refine2[0])
        error_3d = evaluate_reconstruction(all_3d_gt, kpts_2d_projs, Ks[0], Rs[0], ts[0], R_gt_refine1[0] if m_idx == 0 else R_gt_refine2[0])

        print(f'3D error (refine GT): {error_3d}')
        #####################################################################

        counter = 0
        quat_error_pairs = torch.empty((0, 5), device='cuda', dtype=torch.float32)
        for i in range(N):
            selected_idxs = torch.tensor(np.random.choice(np.arange(point_corresponds.shape[0]), size=S), device='cuda')
            selected = point_corresponds[selected_idxs]

            R_initial1, R_initial2, F_initial = find_rotation_matrices(selected, None, Ks)
            _, m_idx = compare_rotations(Rs, (R_initial1, R_initial2))

            lines_initial = kornia.geometry.compute_correspond_epilines(torch.unsqueeze(point_corresponds[:, 0], dim=0), F_initial)[0]
            dists_initial = torch.abs(lines_initial[:, 0] * point_corresponds[:, 1, 0] + lines_initial[:, 1] * point_corresponds[:, 1, 1] \
                + lines_initial[:, 2]) / torch.sqrt(lines_initial[:, 0] ** 2 + lines_initial[:, 1] ** 2)
            dists_initial = dists_initial.view(-1)

            condition_initial = dists_initial < D
            num_inliers_initial = (condition_initial).sum()
            epipolar_dist_initial = torch.mean(dists_initial[condition_initial])

            if num_inliers_initial > I:
                # Evaluate 2D projections and 3D reprojections (triangulation).
                kpts_2d_projs = evaluate_projection(all_3d_gt, Ks[0], Rs[0], ts[0], R_initial1[0] if m_idx == 0 else R_initial2[0])
                error_3d = evaluate_reconstruction(all_3d_gt, kpts_2d_projs, Ks[0], Rs[0], ts[0], R_initial1[0] if m_idx == 0 else R_initial2[0])

                quaternion = kornia.rotation_matrix_to_quaternion(R_initial1 if m_idx == 0 else R_initial2)[0]
                quat_error_pair = torch.unsqueeze(torch.cat((quaternion, torch.unsqueeze(error_3d, dim=0)), dim=0), dim=0)
                quat_error_pairs = torch.cat((quat_error_pairs, quat_error_pair), dim=0)
                print(f'{counter}. {quaternion} -> {error_3d:.2f}')

                '''
                ############# REFINEMENT ##############
                inliers_initial = point_corresponds[condition_initial]
                # TODO: Rename the following variables from _initial to _refine.
                dist_criterion = D
                for _ in range(10):
                    dist_criterion *= 1.04
                    # Estimate camera parameters using all inliers.
                    inliers_initial = point_corresponds[condition_initial]
                    
                    R_initial1, R_initial2, F_initial = find_rotation_matrices(inliers_initial, None, Ks)
                    R_sim, m_idx = compare_rotations(Rs, (R_initial1, R_initial2))

                    # Calculate distances and evaluations based on fundamental matrix estimated from inliers.
                    lines_initial = kornia.geometry.compute_correspond_epilines(torch.unsqueeze(point_corresponds[:, 0], dim=0), F_initial)[0]
                    dists_initial = torch.abs(lines_initial[:, 0] * point_corresponds[:, 1, 0] + \
                        lines_initial[:, 1] * point_corresponds[:, 1, 1] + lines_initial[:, 2]) \
                        / torch.sqrt(lines_initial[:, 0] ** 2 + lines_initial[:, 1] ** 2)

                    # Evaluate condition to get the number of inliers when initial inliers are used for estimation.
                    condition_initial = dists_initial < dist_criterion
                    num_inliers_initial = (condition_initial).sum()
                    epipolar_dist_initial = torch.mean(dists_initial[condition_initial])

                    #kpts_2d_projs = evaluate_projection(all_3d_gt, Ks[0], Rs[0], ts[0], R_initial1[0] if m_idx == 0 else R_initial2[0])
                    #error_3d = evaluate_reconstruction(all_3d_gt, kpts_2d_projs, Ks[0], Rs[0], ts[0], R_initial1[0] if m_idx == 0 else R_initial2[0])
                    #print(f'Num inliers: {num_inliers_initial}')
                    #print(f'Distance criterion: {dist_criterion:.4f}')

                #input('')

                dist_criterion = D
                for _ in range(10):
                    dist_criterion /= 1.05
                    # Estimate camera parameters using all inliers.
                    inliers_initial = point_corresponds[condition_initial]
                    
                    R_initial1, R_initial2, F_initial = find_rotation_matrices(inliers_initial, None, Ks)
                    R_sim, m_idx = compare_rotations(Rs, (R_initial1, R_initial2))

                    # Calculate distances and evaluations based on fundamental matrix estimated from inliers.
                    lines_initial = kornia.geometry.compute_correspond_epilines(torch.unsqueeze(point_corresponds[:, 0], dim=0), F_initial)[0]
                    dists_initial = torch.abs(lines_initial[:, 0] * point_corresponds[:, 1, 0] + \
                        lines_initial[:, 1] * point_corresponds[:, 1, 1] + lines_initial[:, 2]) \
                        / torch.sqrt(lines_initial[:, 0] ** 2 + lines_initial[:, 1] ** 2)

                    # Evaluate condition to get the number of inliers when initial inliers are used for estimation.
                    condition_initial = dists_initial < dist_criterion
                    num_inliers_initial = (condition_initial).sum()
                    epipolar_dist_initial = torch.mean(dists_initial[condition_initial])

                    #kpts_2d_projs = evaluate_projection(all_3d_gt, Ks[0], Rs[0], ts[0], R_initial1[0] if m_idx == 0 else R_initial2[0])
                    #error_3d = evaluate_reconstruction(all_3d_gt, kpts_2d_projs, Ks[0], Rs[0], ts[0], R_initial1[0] if m_idx == 0 else R_initial2[0])
                    #print(f'Num inliers: {num_inliers_initial}')
                    #print(f'Distance criterion: {dist_criterion:.4f}')

                #input('')
                #########################################
                '''
                counter += 1


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

        for idx1, idx2 in combinations2d:
            plt.clf()
            plt.scatter(quat_error_pairs_np[:, idx1], quat_error_pairs_np[:, idx2], c='blue')
            plt.savefig(f'quat_distro_{idx1}{idx2}_blue.png')

        plt.clf()
        for idx1, idx2 in combinations2d:
            plt.scatter(quat_error_pairs_np[~camera_inliers, idx1], quat_error_pairs_np[~camera_inliers, idx2], c='blue')
            plt.scatter(quat_error_pairs_np[camera_inliers, idx1], quat_error_pairs_np[camera_inliers, idx2], c='red')
        plt.savefig(f'quat_distro_2d.png')

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

        for idx1, idx2, idx3 in combinations3d:
            ax.cla()
            ax.scatter(quat_error_pairs_np[:, idx1], quat_error_pairs_np[:, idx2], \
                quat_error_pairs_np[:, idx3], c='blue')
            plt.savefig(f'quat_distro_{idx1}{idx2}{idx3}_blue.png')

        ax.cla()
        for idx1, idx2, idx3 in combinations3d:
            ax.scatter(quat_error_pairs_np[~camera_inliers, idx1], quat_error_pairs_np[~camera_inliers, idx2], \
                quat_error_pairs_np[~camera_inliers, idx3], c='blue')
            ax.scatter(quat_error_pairs_np[camera_inliers, idx1], quat_error_pairs_np[camera_inliers, idx2], \
                quat_error_pairs_np[camera_inliers, idx3], c='red')
        plt.savefig(f'quat_distro_3d.png')

        print(f'Number of camera inliers (total): {np.sum(camera_inliers)}')

