import torch
import torchvision.transforms as transforms
import numpy as np


"""
Projective Geometry Helper Functions 
"""

PADDING = 200
FACE_CENTER = np.array([0, 0, -0.5])

pad = transforms.Pad(PADDING, padding_mode="reflect")


def normalize(pnts):
    """
    pnts - (k) or (n, k) array of k-dimensional Cartesian coordiantes
    Return (k+1) or (n, k+1) homogenous coordinates
    """
    if len(pnts.shape) < 2:
        return pnts / np.linalg.norm(pnts)
    else:
        return pnts / np.linalg.norm(pnts, axis=1)[:, None]


def get_ndc_corners():
    """
    Return four corners of an image in normalized device coordinates.
    """
    return np.array([[-1, -1, 1, 1], [1, -1, 1, 1], [1, 1, 1, 1], [-1, 1, 1, 1]])


def get_viewport_corners(size):
    """
    Return four corners of an image in pixel coordinates.
    """
    return np.array([[0, 0], [0, size[1]], [size[0], size[1]], [size[0], 0]])


def apply_mat(m, pnts):
    """
    pnts - (n, k) array of k-dimensional points
    Multiple every point in ps with matrix m
    """
    if len(pnts.shape) > 2:
        h, w, c = pnts.shape
        pnts_flat = pnts.reshape(h * w, c)
        pnts_transformed_float = np.tensordot(pnts_flat.T, m, axes=(0, 1))
        return pnts_transformed_float.reshape(h, w, c)
    else:
        return np.tensordot(pnts.T, m, axes=(0, 1))


def to_homogenous(pnts):
    """
    pnts - (k) or (n, k) array of k-dimensional Cartesian coordiantes
    Return (k+1) or (n, k+1) homogenous coordinates
    """
    shape_h = np.array(pnts.shape)
    last_dim = pnts.shape[-1]
    shape_h[-1] = last_dim + 1
    out = np.ones(shape_h)
    if len(shape_h) < 2:
        out[:last_dim] = pnts
    else:
        out[:, :last_dim] = pnts
    return out


def get_homogenized(pnts):
    """
    pnts - (k) or (n, k) array of k-dimensional homogenous coordinates
    Return (k-1) or (n, k-1) Cartesian coordiantes
    """
    h_idx = pnts.shape[-1] - 1
    if len(pnts.shape) < 2:
        return pnts[:h_idx] / pnts[h_idx]
    else:
        return pnts[:, :h_idx] / pnts[:, h_idx:]


class Camera:
    def __init__(self, mEx, mEx_inv, mIn, mIn_inv):
        self.mEx = mEx
        self.mIn = mIn
        self.mEx_inv = mEx_inv
        self.mIn_inv = mIn_inv

    def Create(pose, intrinsics):
        mEx = np.linalg.inv(pose)  # world to camera
        mEx_inv = pose
        proj = np.eye(4)
        proj[0, 0] = intrinsics[0, 0]
        proj[1, 1] = intrinsics[1, 1]
        mIn = proj  # camera to ndc
        mIn_inv = np.linalg.inv(proj)
        return Camera(mEx, mEx_inv, mIn, mIn_inv)

    def get_moved(self, new_mEx):
        return Camera(new_mEx, np.linalg.inv(new_mEx), self.mIn, self.mIn_inv)

    def world_to_viewport(self, pnts_world, size):
        """
        pnts - (n, 3) numpy array of points in world coordinates
        size - (h, w) size of the output image measured in number of pixels
        Transform an array of points in world coordinates to pixel coordinates of given size.
        """
        pnts_world = to_homogenous(pnts_world)
        pnts_local = apply_mat(self.mEx, pnts_world)
        pnts_ndc = apply_mat(self.mIn, pnts_local)
        pnts_ndc_xy = pnts_ndc[:, :2] / pnts_ndc[:, 2:3]
        pnts_ndx_yx = np.flip(pnts_ndc_xy, axis=1)
        pnts_viewport = (pnts_ndx_yx + 1.0) / 2.0 * size
        return pnts_viewport

    def get_image_plane(self, pnt_interest):
        """
        pnt_interest - point of interest in world coordinate
        Return the image plane of current camera pose containing some point of interest,
        represented by the four corners in (4, 3) numpy array.
        """
        # compute focus distance with point of interest
        point_of_interest_world = to_homogenous(pnt_interest)
        point_of_interest_local = self.mEx @ point_of_interest_world
        focus_distance = point_of_interest_local[2] / point_of_interest_local[3]

        # unproject the image bounding box onto the plane of focus
        b_box_ndc = get_ndc_corners()
        b_box_local = apply_mat(self.mIn_inv, b_box_ndc)
        # scale by focus distance
        b_box_local[:, :3] = (
            b_box_local[:, :3] / b_box_local[0, 2] * b_box_local[0, 3] * focus_distance
        )
        b_box_world = apply_mat(self.mEx_inv, b_box_local)
        return b_box_world

    def get_reprojected_image_plane(self, plane, pnt_interest):
        """
        plane - (4, 3) numpy array, four corners of a given image plane in world coordinates
        Return the image plane re-projected onto the given plane.
        """
        original_plane = self.get_image_plane(pnt_interest)
        original_plane = get_homogenized(original_plane)
        plane = get_homogenized(plane)
        camera_position = self.mEx_inv[:3, 3]

        # find the directs of the four rays surrounding the view frustum
        bounding_rays = normalize(original_plane - camera_position)
        origin_plane_vec = camera_position - plane[0]
        plane_v1 = normalize(plane[1] - plane[0])
        plane_v2 = normalize(plane[2] - plane[0])
        plane_normal = np.cross(plane_v1, plane_v2)
        dist = np.dot(origin_plane_vec, plane_normal)
        cos_thetas = np.dot(bounding_rays, -plane_normal)
        ray_lengths = dist / cos_thetas
        reprojected = bounding_rays * ray_lengths[:, None] + camera_position
        return reprojected


def get_reprojection_mapping(
    image, src_camera, dst_camera, point_of_interest=FACE_CENTER
):
    size = image.shape[1:]
    # Find the image plane for reprojection
    target_img_plane = src_camera.get_image_plane(point_of_interest)
    # Compute world coordinates of the reprojected image plane
    reproject_img_plane = src_camera.get_reprojected_image_plane(
        target_img_plane, point_of_interest
    )
    # Compute pixel coordinates of the reprojected image plane
    dst_img_corners = dst_camera.world_to_viewport(reproject_img_plane, size)
    src_img_corners = get_viewport_corners(size)
    return (
        np.flip(src_img_corners, axis=1).copy(),
        np.flip(dst_img_corners, axis=1).copy(),
    )


def apply_homography(image, src_pnts, dst_pnts, padding=True):
    h, w = image.shape[1:]
    return transforms.functional.perspective(
        pad(image), src_pnts + PADDING, dst_pnts + PADDING
    )[:, PADDING : PADDING + h, PADDING : PADDING + w]


def get_front_cam(camera):
    # pos = camera.mEx_inv[:3, 3] - FACE_CENTER
    # z = np.linalg.norm(pos)
    mEx = np.eye(4)
    mEx[0, 0] = 1
    mEx[1, 1] = -1
    mEx[2, 2] = -1
    # mEx[:3, 3] = FACE_CENTER
    # mEx[2, 3] += z
    mEx[:3, 3] = camera.mEx[:3, 3]
    return camera.get_moved(mEx)


def transform_front(image, camera, padding=True):
    rotated_cam = get_front_cam(camera)
    src_pnts, dst_pnts = get_reprojection_mapping(image, camera, rotated_cam)
    warped_img = apply_homography(image, src_pnts, dst_pnts, padding)
    return warped_img


def transform_camera(image, camera1, camera2, padding=True):
    rotated_cam = camera2
    src_pnts, dst_pnts = get_reprojection_mapping(image, camera1, rotated_cam)
    warped_img = apply_homography(image, src_pnts, dst_pnts, padding)
    return warped_img
