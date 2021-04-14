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

M = 15             # number of frames
J = 17              # number of joints
P = M * J           # total number of point correspondences    
N = 20000           # trials
eps = 0.71           # outlier probability
S = 50              # sample size
I = (1 - eps) * P  # number of inliers condition
D = .5              # distance criterion
H = 15              # top H hypotheses to use


if __name__ == '__main__':
    print(f'Number of inliers condition: {I} ({P})')

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

        
        ###################### GT #############################
        '''
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
        '''

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
        #inliers = point_corresponds[condition]
        inliers_gt_refine = point_corresponds[condition_gt_initial][:S]
        R_gt_refine1, R_gt_refine2, F_gt_refine = find_rotation_matrices(inliers_gt_refine, None, Ks)
        R_sim, m_idx = compare_rotations(Rs, (R_gt_refine1, R_gt_refine2))
        #print(f'Quaternion similarity between the rotations: {R_sim:.5f}')
        
        lines_gt_refine = kornia.geometry.compute_correspond_epilines(torch.unsqueeze(point_corresponds[:, 0], dim=0), F_gt_refine)[0]
        dists_gt_refine = torch.abs(lines_gt_refine[:, 0] * point_corresponds[:, 1, 0] + \
            lines_gt_refine[:, 1] * point_corresponds[:, 1, 1] + lines_gt_refine[:, 2]) \
            / torch.sqrt(lines_gt_refine[:, 0] ** 2 + lines_gt_refine[:, 1] ** 2)
        dists_gt_refine = dists_gt_refine.view(-1)

        condition_gt_refine = dists_gt_refine < D
        num_inliers_gt_refine = (condition_gt_refine).sum()
        print(f'Number of inliers (refine GT): {num_inliers_gt_refine} ({P})')

        kpts_2d_projs = evaluate_projection(all_3d_gt, Ks[0], Rs[0], ts[0], R_gt_refine1[0] if m_idx == 0 else R_gt_refine2[0])
        _ = evaluate_reconstruction(all_3d_gt, kpts_2d_projs, Ks[0], Rs[0], ts[0], R_gt_refine1[0] if m_idx == 0 else R_gt_refine2[0])

        '''
        epipol_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        epipol_gt_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        kpts_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        epipol_gt_img = draw_2d_pose_cv2(all_2d_preds_bboxed[0, 1].cpu().numpy(), np.copy(epipol_gt_img))
        _, epipol_gt_img = draw_epipolar_lines(torch.unsqueeze(all_2d_preds_bboxed[0, 0], dim=0), F_bboxed, 
            epipol_gt_img, all_2d_preds_bboxed[0, 1], dists_bboxed)

        draw_images([epipol_gt_img], 'ransac')
        '''
        #####################################################################
        

        counter = 0
        #best_error = torch.tensor([1000.], device='cuda', dtype=torch.float32)

        max_inliers_initial = -1
        inlier_error_initial = None
        best_inlier_error_initial = 1000.

        max_inliers_refine = -1
        inlier_error_refine = None
        best_inlier_error_refine = 1000.

        min_epipolar_dist_initial = 1000.
        epipolar_dist_error_initial = None
        best_epipolar_dist_error_initial = 1000.

        min_epipolar_dist_refine = 1000.
        epipolar_dist_error_refine = None
        best_epipolar_dist_error_refine = 1000.

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
                counter += 1

                inliers_initial = point_corresponds[condition_initial]

                if num_inliers_initial > max_inliers_initial:
                    print(f'num_inliers initial: {num_inliers_initial} ({epipolar_dist_initial:.4f})')
                    max_inliers_initial = num_inliers_initial

                    # Evaluate 2D projections and 3D reprojections (triangulation).
                    kpts_2d_projs = evaluate_projection(all_3d_gt, Ks[0], Rs[0], ts[0], R_initial1[0] if m_idx == 0 else R_initial2[0])
                    error_3d = evaluate_reconstruction(all_3d_gt, kpts_2d_projs, Ks[0], Rs[0], ts[0], R_initial1[0] if m_idx == 0 else R_initial2[0])

                    inlier_error_initial = error_3d
                    if error_3d < best_inlier_error_initial:
                        best_inlier_error_initial = error_3d

                    dist_criterion = D
                    for _ in range(20):
                        dist_criterion *= 1.03
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

                        kpts_2d_projs = evaluate_projection(all_3d_gt, Ks[0], Rs[0], ts[0], R_initial1[0] if m_idx == 0 else R_initial2[0])
                        error_3d = evaluate_reconstruction(all_3d_gt, kpts_2d_projs, Ks[0], Rs[0], ts[0], R_initial1[0] if m_idx == 0 else R_initial2[0])
                        print(f'Num inliers: {num_inliers_initial}')
                        print(f'Distance criterion: {dist_criterion:.4f}')

                    input('')

                    dist_criterion = D
                    for _ in range(20):
                        dist_criterion /= 1.04
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

                        kpts_2d_projs = evaluate_projection(all_3d_gt, Ks[0], Rs[0], ts[0], R_initial1[0] if m_idx == 0 else R_initial2[0])
                        error_3d = evaluate_reconstruction(all_3d_gt, kpts_2d_projs, Ks[0], Rs[0], ts[0], R_initial1[0] if m_idx == 0 else R_initial2[0])
                        print(f'Num inliers: {num_inliers_initial}')
                        print(f'Distance criterion: {dist_criterion:.4f}')

                    input('')

                '''
                if epipolar_dist_initial < min_epipolar_dist_initial:
                    print(f'epipolar_dist initial: {epipolar_dist_initial:.4f} ({num_inliers_initial})')
                    min_epipolar_dist_initial = epipolar_dist_initial

                    # Evaluate 2D projections and 3D reprojections (triangulation).
                    kpts_2d_projs = evaluate_projection(all_3d_gt, Ks[0], Rs[0], ts[0], R_initial1[0] if m_idx == 0 else R_initial2[0])
                    error_3d = evaluate_reconstruction(all_3d_gt, kpts_2d_projs, Ks[0], Rs[0], ts[0], R_initial1[0] if m_idx == 0 else R_initial2[0])

                    epipolar_dist_error_initial = error_3d
                    if error_3d < best_epipolar_dist_error_initial:
                        best_epipolar_dist_error_initial = error_3d
                '''


                '''
                # Estimate camera parameters using all inliers.
                #inliers = point_corresponds[condition]
                #inliers = torch.tensor(point_corresponds, device='cuda', dtype=torch.float32)[condition_gt]
                R_refine1, R_refine2, F_refine = find_rotation_matrices(inliers_initial, None, Ks)
                R_sim, m_idx = compare_rotations(Rs, (R_refine1, R_refine2))

                # Calculate distances and evaluations based on fundamental matrix estimated from inliers.
                lines_refine = kornia.geometry.compute_correspond_epilines(torch.unsqueeze(point_corresponds[:, 0], dim=0), F_refine)[0]
                dists_refine = torch.abs(lines_refine[:, 0] * point_corresponds[:, 1, 0] + \
                    lines_refine[:, 1] * point_corresponds[:, 1, 1] + lines_refine[:, 2]) \
                    / torch.sqrt(lines_refine[:, 0] ** 2 + lines_refine[:, 1] ** 2)

                # Evaluate condition to get the number of inliers when initial inliers are used for estimation.
                condition_refine = dists_refine < D
                num_inliers_refine = (condition_refine).sum()
                epipolar_dist_refine = torch.mean(dists_refine[condition_refine])


                if num_inliers_refine > max_inliers_refine:
                    print(f'num_inliers refine: {num_inliers_refine} ({epipolar_dist_refine:.4f})')
                    max_inliers_refine = num_inliers_refine

                    # Evaluate 2D projections and 3D reprojections (triangulation).
                    kpts_2d_projs = evaluate_projection(all_3d_gt, Ks[0], Rs[0], ts[0], R_refine1[0] if m_idx == 0 else R_refine2[0])
                    error_3d = evaluate_reconstruction(all_3d_gt, kpts_2d_projs, Ks[0], Rs[0], ts[0], R_refine1[0] if m_idx == 0 else R_refine2[0])

                    inlier_error_refine = error_3d
                    if error_3d < best_inlier_error_refine:
                        best_inlier_error_refine = error_3d

                    dist_criterion = D
                    for _ in range(15):
                        dist_criterion *= 1.03
                        # Estimate camera parameters using all inliers.
                        inliers_refine = point_corresponds[condition_refine]
                        
                        R_refine1, R_refine2, F_refine = find_rotation_matrices(inliers_refine, None, Ks)
                        R_sim, m_idx = compare_rotations(Rs, (R_refine1, R_refine2))

                        # Calculate distances and evaluations based on fundamental matrix estimated from inliers.
                        lines_refine = kornia.geometry.compute_correspond_epilines(torch.unsqueeze(point_corresponds[:, 0], dim=0), F_refine)[0]
                        dists_refine = torch.abs(lines_refine[:, 0] * point_corresponds[:, 1, 0] + \
                            lines_refine[:, 1] * point_corresponds[:, 1, 1] + lines_refine[:, 2]) \
                            / torch.sqrt(lines_refine[:, 0] ** 2 + lines_refine[:, 1] ** 2)

                        # Evaluate condition to get the number of inliers when initial inliers are used for estimation.
                        condition_refine = dists_refine < dist_criterion
                        num_inliers_refine = (condition_refine).sum()
                        epipolar_dist_refine = torch.mean(dists_refine[condition_refine])

                        kpts_2d_projs = evaluate_projection(all_3d_gt, Ks[0], Rs[0], ts[0], R_refine1[0] if m_idx == 0 else R_refine2[0])
                        error_3d = evaluate_reconstruction(all_3d_gt, kpts_2d_projs, Ks[0], Rs[0], ts[0], R_refine1[0] if m_idx == 0 else R_refine2[0])
                        print(f'Num inliers: {num_inliers_refine}')
                        print(f'Distance criterion: {dist_criterion:.4f}')

                    input('')

                    dist_criterion = D
                    for _ in range(15):
                        dist_criterion /= 1.04
                        # Estimate camera parameters using all inliers.
                        inliers_refine = point_corresponds[condition_refine]
                        
                        R_refine1, R_refine2, F_refine = find_rotation_matrices(inliers_refine, None, Ks)
                        R_sim, m_idx = compare_rotations(Rs, (R_refine1, R_refine2))

                        # Calculate distances and evaluations based on fundamental matrix estimated from inliers.
                        lines_refine = kornia.geometry.compute_correspond_epilines(torch.unsqueeze(point_corresponds[:, 0], dim=0), F_refine)[0]
                        dists_refine = torch.abs(lines_refine[:, 0] * point_corresponds[:, 1, 0] + \
                            lines_refine[:, 1] * point_corresponds[:, 1, 1] + lines_refine[:, 2]) \
                            / torch.sqrt(lines_refine[:, 0] ** 2 + lines_refine[:, 1] ** 2)

                        # Evaluate condition to get the number of inliers when initial inliers are used for estimation.
                        condition_refine = dists_refine < dist_criterion
                        num_inliers_refine = (condition_refine).sum()
                        epipolar_dist_refine = torch.mean(dists_refine[condition_refine])

                        kpts_2d_projs = evaluate_projection(all_3d_gt, Ks[0], Rs[0], ts[0], R_refine1[0] if m_idx == 0 else R_refine2[0])
                        error_3d = evaluate_reconstruction(all_3d_gt, kpts_2d_projs, Ks[0], Rs[0], ts[0], R_refine1[0] if m_idx == 0 else R_refine2[0])
                        print(f'Num inliers: {num_inliers_refine}')
                        print(f'Distance criterion: {dist_criterion:.4f}')

                    input('')
                    '''

                '''
                if epipolar_dist_refine < min_epipolar_dist_refine:
                    print(f'epipolar_dist refine: {epipolar_dist_refine:.4f} ({num_inliers_refine})')
                    min_epipolar_dist_refine = epipolar_dist_refine

                    # Evaluate 2D projections and 3D reprojections (triangulation).
                    kpts_2d_projs = evaluate_projection(all_3d_gt, Ks[0], Rs[0], ts[0], R_refine1[0] if m_idx == 0 else R_refine2[0])
                    error_3d = evaluate_reconstruction(all_3d_gt, kpts_2d_projs, Ks[0], Rs[0], ts[0], R_refine1[0] if m_idx == 0 else R_refine2[0])

                    epipolar_dist_error_refine = error_3d
                    if error_3d < best_epipolar_dist_error_refine:
                        best_epipolar_dist_error_refine = error_3d
                '''


        print(inlier_error_initial)
        print(inlier_error_refine)
        print(best_inlier_error_initial)
        print(best_inlier_error_refine)

        print(epipolar_dist_error_initial)
        print(epipolar_dist_error_refine)
        print(best_epipolar_dist_error_initial)
        print(best_epipolar_dist_error_refine)
