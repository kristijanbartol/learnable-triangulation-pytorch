import numpy as np
import torch
import kornia
import cv2
import time

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

M = 50             # number of frames
J = 17              # number of joints
P = M * J           # total number of point correspondences    
N = 20000           # trials
eps = 0.75           # outlier probability
S = 25              # sample size
I = (1 - eps) * P  # number of inliers condition
D = .5              # distance criterion


if __name__ == '__main__':
    print(f'Number of inliers condition: {I}')

    with torch.no_grad():
        all_images = np.load(IMGS_PATH)[:M]
        all_2d_preds = np.load(PRED_PATH)[:M, IDXS]
        all_3d_gt = torch.tensor(np.load(GT_PATH)[:M], device='cuda', dtype=torch.float32)
        Ks_bboxed = torch.tensor(np.load(KS_BBOX_PATH)[:M], device='cuda', dtype=torch.float32)
        Ks = np.load(K_PATH)
        Rs = np.load(R_PATH)
        ts = np.load(T_PATH)
        bboxes = np.load(BBOX_PATH)[:M, IDXS]

        all_right_imgs = image_batch_to_numpy(all_images[:, 1])
        img = denormalize_image(all_right_imgs[0]).astype(np.uint8)
        img = img[..., ::-1]  # bgr -> rgb

        all_2d_preds_bboxed = torch.tensor(all_2d_preds, device='cuda', dtype=torch.float32)

        # Unbbox keypoints.
        # TODO: Update magic number 384.
        # TODO: Write in Torch.
        bbox_height = np.abs(bboxes[:, :, 0, 0] - bboxes[:, :, 1, 0])
        all_2d_preds *= np.expand_dims(np.expand_dims(bbox_height / 384., axis=-1), axis=-1)
        all_2d_preds += np.expand_dims(bboxes[:, :, 0, :], axis=2)

        # All points stacked along a single dimension.
        point_corresponds = torch.tensor(np.concatenate(np.split(all_2d_preds, all_2d_preds.shape[0], axis=0), axis=2)[0], 
            device='cuda', dtype=torch.float32).transpose(0, 1)

        Ks = torch.unsqueeze(torch.tensor(Ks, device='cuda', dtype=torch.float32), dim=0)
        Rs = torch.unsqueeze(torch.tensor(Rs, device='cuda', dtype=torch.float32), dim=0)
        ts = torch.unsqueeze(torch.tensor(ts, device='cuda', dtype=torch.float32), dim=0)

        all_2d_preds = torch.tensor(all_2d_preds, device='cuda', dtype=torch.float32)

        '''
        ###################### GT #############################
        # Using bboxed Ks to visualize created fundamental matrix for the first frame and calculate all distances.
        F_bboxed = create_fundamental_matrix(Ks_bboxed, Rs.expand((M, 4, 3, 3)), ts.expand((M, 4, 3, 1)))
        lines_bboxed = kornia.geometry.compute_correspond_epilines(all_2d_preds_bboxed[:, 0], F_bboxed)
        dists_bboxed = torch.abs(lines_bboxed[:, :, 0] * all_2d_preds_bboxed[:, 1, :, 0] + \
            lines_bboxed[:, :, 1] * all_2d_preds_bboxed[:, 1, :, 1] + lines_bboxed[:, :, 2]) \
            / torch.sqrt(lines_bboxed[:, :, 0] ** 2 + lines_bboxed[:, :, 1] ** 2)
        dists_bboxed = dists_bboxed.view(-1)

        condition = dists_bboxed < .8
        num_inliers = (condition).sum()
        print(f'Number of inliers (bboxed GT): {num_inliers} ({P})')

        # Using unbboxed Ks to calculate all distances.
        F_gt = create_fundamental_matrix(Ks, Rs, ts)
        lines_gt = kornia.geometry.compute_correspond_epilines(torch.unsqueeze(point_corresponds[:, 0], dim=0), F_gt)[0]
        dists_gt = torch.abs(lines_gt[:, 0] * point_corresponds[:, 1, 0] + \
            lines_gt[:, 1] * point_corresponds[:, 1, 1] + lines_gt[:, 2]) \
            / torch.sqrt(lines_gt[:, 0] ** 2 + lines_gt[:, 1] ** 2)
        dists_gt = dists_gt.view(-1)

        condition = dists_gt < .3
        num_inliers = (condition).sum()
        print(f'Number of inliers (pre-loop GT): {num_inliers} ({P})')

        # Evaluate fundamental matrix.
        #inliers = point_corresponds[condition]
        inliers = point_corresponds[condition][:S]
        R_inliers1, R_inliers2, F_inliers = find_rotation_matrices(inliers, None, Ks)
        R_sim, m_idx = compare_rotations(Rs, (R_inliers1, R_inliers2))
        print(f'Quaternion similarity between the rotations: {R_sim:.5f}')
        
        kpts_2d_projs = evaluate_projection(all_3d_gt[0], Ks[0], Rs[0], ts[0], R_inliers1[0] if m_idx == 0 else R_inliers2[0])

        epipol_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        epipol_gt_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        kpts_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        epipol_gt_img = draw_2d_pose_cv2(all_2d_preds_bboxed[0, 1].cpu().numpy(), np.copy(epipol_gt_img))
        _, epipol_gt_img = draw_epipolar_lines(torch.unsqueeze(all_2d_preds_bboxed[0, 0], dim=0), F_bboxed, 
            epipol_gt_img, all_2d_preds_bboxed[0, 1], dists_bboxed)

        draw_images([epipol_gt_img], 'ransac')
        #####################################################################
        '''

        counter = 0
        best_error = torch.tensor([1000.], device='cuda', dtype=torch.float32)
        min_sim = torch.tensor([1000.], device='cuda', dtype=torch.float32)
        sim_error = None

        max_inliers = -1
        inlier_error = None

        min_epipolar_dist = 1000.
        epipolar_dist_error = None
        for i in range(N):
            selected_idxs = torch.tensor(np.random.choice(np.arange(point_corresponds.shape[0]), size=S), device='cuda')
            selected = point_corresponds[selected_idxs]

            R_est1, R_est2, F = find_rotation_matrices(selected, None, Ks)

            lines = kornia.geometry.compute_correspond_epilines(torch.unsqueeze(point_corresponds[:, 0], dim=0), F)[0]
            dists = torch.abs(lines[:, 0] * point_corresponds[:, 1, 0] + lines[:, 1] * point_corresponds[:, 1, 1] + lines[:, 2]) \
                / torch.sqrt(lines[:, 0] ** 2 + lines[:, 1] ** 2)
            dists = dists.view(-1)

            condition = dists < D
            num_inliers = (condition).sum()

            if num_inliers > I:
                counter += 1

                print(f'Number of inliers (estimated): {num_inliers} ({P})')

                # Estimate camera parameters using all inliers.
                inliers = point_corresponds[condition]
                #inliers = torch.tensor(point_corresponds, device='cuda', dtype=torch.float32)[condition_gt]
                R_inliers1, R_inliers2, F_inliers = find_rotation_matrices(inliers, None, Ks)
                R_sim, m_idx = compare_rotations(Rs, (R_inliers1, R_inliers2))
                print(f'Quaternion similarity between the rotations: {R_sim:.5f}')

                # Calculate distances and evaluations based on fundamental matrix estimated from inliers.
                lines_inliers = kornia.geometry.compute_correspond_epilines(torch.unsqueeze(point_corresponds[:, 0], dim=0), F_inliers)[0]
                dists_inliers = torch.abs(lines_inliers[:, 0] * point_corresponds[:, 1, 0] + \
                    lines_inliers[:, 1] * point_corresponds[:, 1, 1] + lines_inliers[:, 2]) \
                    / torch.sqrt(lines_inliers[:, 0] ** 2 + lines_inliers[:, 1] ** 2)

                # Evaluate condition to get the number of inliers when initial inliers are used for estimation.
                condition = dists_inliers < D
                num_inliers = (condition).sum()
                print(f'Number of inliers (inliers): {num_inliers} ({P})')

                # Mean distance from keypoints to the epipolar lines.
                epipolar_dist = torch.mean(dists_inliers)

                # Evaluate 2D projections and 3D reprojections (triangulation).
                kpts_2d_projs = evaluate_projection(all_3d_gt, Ks[0], Rs[0], ts[0], R_inliers1[0] if m_idx == 0 else R_inliers2[0])
                error_3d = evaluate_reconstruction(all_3d_gt, kpts_2d_projs, Ks[0], Rs[0], ts[0], R_est1[0] if m_idx == 0 else R_est2[0])

                if error_3d < best_error:
                    best_error = error_3d

                if R_sim < min_sim:
                    min_sim = R_sim
                    sim_error = error_3d

                if epipolar_dist < min_epipolar_dist:
                    min_epipolar_dist = epipolar_dist
                    epipolar_dist_error = error_3d

                if num_inliers > max_inliers:
                    max_inliers = num_inliers
                    inlier_error = error_3d

        print(f'Best error: {best_error}')
        print(f'Minimal quaternion similarity: {min_sim}')
        print(f'3D error based on the minimal quaternion similarity: {sim_error}')
        print(f'Maximal inlier set size: {max_inliers}')
        print(f'3D error based on the maximal inlier set size: {inlier_error}')
        print(f'Minimal mean distance between the keypoints and the epipolar lines: {min_epipolar_dist}')
        print(f'3D error based on the minimal epipolar distance: {epipolar_dist_error}')
