import numpy as np
import torch
import itertools
import time


class Camera:
    def __init__(self, R, t, K, dist=None, name=""):
        self.R = np.array(R).copy()
        assert self.R.shape == (3, 3)

        self.t = np.array(t).copy()
        assert self.t.size == 3
        self.t = self.t.reshape(3, 1)

        self.K = np.array(K).copy()
        assert self.K.shape == (3, 3)

        self.dist = dist
        if self.dist is not None:
            self.dist = np.array(self.dist).copy().flatten()

        self.name = name

    def update_after_crop(self, bbox):
        left, upper, right, lower = bbox

        cx, cy = self.K[0, 2], self.K[1, 2]

        new_cx = cx - left
        new_cy = cy - upper

        self.K[0, 2], self.K[1, 2] = new_cx, new_cy

    def update_after_resize(self, image_shape, new_image_shape):
        height, width = image_shape
        new_height, new_width = new_image_shape

        fx, fy, cx, cy = self.K[0, 0], self.K[1, 1], self.K[0, 2], self.K[1, 2]

        new_fx = fx * (new_width / width)
        new_fy = fy * (new_height / height)
        new_cx = cx * (new_width / width)
        new_cy = cy * (new_height / height)

        self.K[0, 0], self.K[1, 1], self.K[0, 2], self.K[1, 2] = new_fx, new_fy, new_cx, new_cy

    @property
    def projection(self):
        return self.K.dot(self.extrinsics)

    @property
    def extrinsics(self):
        return np.hstack([self.R, self.t])


def euclidean_to_homogeneous(points):
    """Converts euclidean points to homogeneous

    Args:
        points numpy array or torch tensor of shape (N, M): N euclidean points of dimension M

    Returns:
        numpy array or torch tensor of shape (N, M + 1): homogeneous points
    """
    if isinstance(points, np.ndarray):
        return np.hstack([points, np.ones((len(points), 1))])
    elif torch.is_tensor(points):
        return torch.cat([points, torch.ones((points.shape[0], 1), dtype=points.dtype, device=points.device)], dim=1)
    else:
        raise TypeError("Works only with numpy arrays and PyTorch tensors.")


def homogeneous_to_euclidean(points):
    """Converts homogeneous points to euclidean

    Args:
        points numpy array or torch tensor of shape (N, M + 1): N homogeneous points of dimension M

    Returns:
        numpy array or torch tensor of shape (N, M): euclidean points
    """
    if isinstance(points, np.ndarray):
        return (points.T[:-1] / points.T[-1]).T
    elif torch.is_tensor(points):
        return (points.transpose(1, 0)[:-1] / points.transpose(1, 0)[-1]).transpose(1, 0)
    else:
        raise TypeError("Works only with numpy arrays and PyTorch tensors.")


def project_3d_points_to_image_plane_without_distortion(proj_matrix, points_3d, convert_back_to_euclidean=True):
    """Project 3D points to image plane not taking into account distortion
    Args:
        proj_matrix numpy array or torch tensor of shape (3, 4): projection matrix
        points_3d numpy array or torch tensor of shape (N, 3): 3D points
        convert_back_to_euclidean bool: if True, then resulting points will be converted to euclidean coordinates
                                        NOTE: division by zero can be here if z = 0
    Returns:
        numpy array or torch tensor of shape (N, 2): 3D points projected to image plane
    """
    if isinstance(proj_matrix, np.ndarray) and isinstance(points_3d, np.ndarray):
        result = euclidean_to_homogeneous(points_3d) @ proj_matrix.T
        if convert_back_to_euclidean:
            result = homogeneous_to_euclidean(result)
        return result
    elif torch.is_tensor(proj_matrix) and torch.is_tensor(points_3d):
        result = euclidean_to_homogeneous(points_3d) @ proj_matrix.t()
        if convert_back_to_euclidean:
            result = homogeneous_to_euclidean(result)
        return result
    else:
        raise TypeError("Works only with numpy arrays and PyTorch tensors.")


def triangulate_point_from_multiple_views_linear(proj_matricies, points):
    """Triangulates one point from multiple (N) views using direct linear transformation (DLT).
    For more information look at "Multiple view geometry in computer vision",
    Richard Hartley and Andrew Zisserman, 12.2 (p. 312).

    Args:
        proj_matricies numpy array of shape (N, 3, 4): sequence of projection matricies (3x4)
        points numpy array of shape (N, 2): sequence of points' coordinates

    Returns:
        point_3d numpy array of shape (3,): triangulated point
    """
    assert len(proj_matricies) == len(points)

    n_views = len(proj_matricies)
    A = np.zeros((2 * n_views, 4))
    for j in range(len(proj_matricies)):
        A[j * 2 + 0] = points[j][0] * proj_matricies[j][2, :] - proj_matricies[j][0, :]
        A[j * 2 + 1] = points[j][1] * proj_matricies[j][2, :] - proj_matricies[j][1, :]

    u, s, vh =  np.linalg.svd(A, full_matrices=False)
    point_3d_homo = vh[3, :]

    point_3d = homogeneous_to_euclidean(point_3d_homo)

    return point_3d


def triangulate_point_from_multiple_views_linear_torch(proj_matricies, points, confidences=None):
    """Triangulate batch of points. 
    Squeeze B and J into the batch channel for parallel computation.
    Args:
        proj_matricies torch tensor of shape (B, N, 3, 4): sequence of projection matricies (3x4)
        points torch tensor of of shape (B, N, J, 2): sequence of points' coordinates
        confidences None or torch tensor of shape (B, N, J): confidences of points [0.0, 1.0].
                                                        If None, all confidences are supposed to be 1.0
    Returns:
        point_3d numpy torch tensor of shape (B, J, 3): triangulated point
    """
    '''
    assert proj_matricies.shape[1] == points.shape[1]

    n_batch = points.shape[0]
    n_views = points.shape[1]
    n_joints = points.shape[2]
    n_coord = points.shape[3]

    # Transpose to properly triangulate (take all views).
    points = points.transpose(1, 2).reshape(-1, n_views, n_coord, 1)
    confidences = confidences.transpose(1, 2).reshape(-1, n_views, 1, 1)

    A = proj_matricies[:, :, 2:3].expand(n_batch, n_views, 2, 4).repeat(n_joints, 1, 1, 1) * points
    A -= proj_matricies[:, :, :2].repeat(n_joints, 1, 1, 1)
    A *= confidences.view(-1, n_views, 1, 1)

    u, s, vh = torch.svd(A.reshape(-1, n_views * n_coord, 4))

    point_3d_homo = -vh[:, :, 3]
    point_3d = homogeneous_to_euclidean(point_3d_homo).reshape(n_batch, n_joints, n_coord + 1)

    return point_3d
    '''
    assert len(proj_matricies) == len(points)

    n_views = len(proj_matricies)

    if confidences is None:
        confidences = torch.ones(n_views, dtype=torch.float32, device=points.device)

    A = proj_matricies[:, 2:3].expand(n_views, 2, 4) * points.view(n_views, 2, 1)
    A -= proj_matricies[:, :2]
    A *= confidences.view(-1, 1, 1)

    u, s, vh = torch.svd(A.view(-1, 4))

    point_3d_homo = -vh[:, 3]
    point_3d = homogeneous_to_euclidean(point_3d_homo.unsqueeze(0))[0]

    return point_3d


def triangulate_batch_of_points(proj_matricies_batch, points_batch, confidences_batch=None, use_view_comb_triang=True):
    batch_size, n_views, n_joints = points_batch.shape[:3]

    if use_view_comb_triang:
        # Generate view [0, 1, ..., n_views] combinations for subsets of length [2, ..., n_views].
        view_combinations = [itertools.combinations(range(n_views), x) for x in range(2, n_views + 1)]
        # Merge sublists
        view_combinations = [j for i in view_combinations for j in i]
        n_view_comb = len(view_combinations)

        points_3d_batch = torch.zeros(batch_size, n_view_comb, n_joints, 3, dtype=torch.float32, device=points_batch.device)

        for comb_i, v_comb in enumerate(view_combinations):
            points = points_batch[:, v_comb, :, :]
            proj_matricies = proj_matricies_batch[:, v_comb]
            confidences = confidences_batch[:, v_comb, :] if confidences_batch is not None else None

            points_3d = triangulate_point_from_multiple_views_linear_torch(proj_matricies, points, confidences=confidences)
            points_3d_batch[:, comb_i] = points_3d

    else:
        '''
        confidences = confidences_batch if confidences_batch is not None else None
        points_3d_batch = triangulate_point_from_multiple_views_linear_torch(proj_matricies_batch, points_batch, confidences=confidences)
        '''

        batch_size, n_views, n_joints = points_batch.shape[:3]
        point_3d_batch = torch.zeros(batch_size, n_joints, 3, dtype=torch.float32, device=points_batch.device)

        for batch_i in range(batch_size):
            for joint_i in range(n_joints):
                points = points_batch[batch_i, :, joint_i, :]

                confidences = confidences_batch[batch_i, :, joint_i] if confidences_batch is not None else None
                point_3d = triangulate_point_from_multiple_views_linear_torch(proj_matricies_batch[batch_i], points, confidences=confidences)
                point_3d_batch[batch_i, joint_i] = point_3d

        return point_3d_batch

    return points_3d_batch


def calc_reprojection_error_matrix(keypoints_3d, keypoints_2d_list, proj_matricies):
    reprojection_error_matrix = []
    for keypoints_2d, proj_matrix in zip(keypoints_2d_list, proj_matricies):
        keypoints_2d_projected = project_3d_points_to_image_plane_without_distortion(proj_matrix, keypoints_3d)
        reprojection_error = 1 / 2 * np.sqrt(np.sum((keypoints_2d - keypoints_2d_projected) ** 2, axis=1))
        reprojection_error_matrix.append(reprojection_error)

    return np.vstack(reprojection_error_matrix).T
