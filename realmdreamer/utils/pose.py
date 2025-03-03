import torch
import numpy as np
from pytorch3d.renderer import PerspectiveCameras
from pytorch3d.utils import (
    cameras_from_opencv_projection,
    opencv_from_cameras_projection,
)


def pytorch3d_to_opengl(pose):
    """
    Input: PerspectiveCameras object - pytorch3d

    Output: 4x4 matrix - opengl (c2w) - compatible w Nerfstudio
    """

    R, tvec, _ = opencv_from_cameras_projection(
        pose, torch.Tensor([512, 512]).to(pose.device).unsqueeze(0)
    )

    # Combine into a 4x4 matrix
    Rt_opencv = np.eye(4)
    Rt_opencv[:3, :3] = R.squeeze(0).cpu().numpy()
    Rt_opencv[:3, 3] = tvec.squeeze(0).cpu().numpy()

    # # Convert to opengl - this is just for exporting back to nerfstudio - not for rendering
    transform = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

    # Convert to openGL
    Rt = transform @ Rt_opencv

    # Convert from w2c to c2w
    Rt = np.linalg.inv(Rt)

    return Rt


def opencv2pytorch3d(pose, f, img_size):
    """
    Convert a pose from OpenCV format to pytorch3d FOV format
    Input:
        pose: 4x4 matrix
    Output:
        cameras: PerspectiveCameras object
    """

    # Convert to pytorch3d format
    R = torch.from_numpy(pose[:3, :3]).float()
    T = torch.from_numpy(pose[:3, 3]).float()
    T = T.reshape(1, 3)
    R = R.reshape(1, 3, 3)

    K = torch.eye(3)[None, ...].float()
    K[..., 0, 0] = K[..., 1, 1] = f
    K[..., 0, 2] = K[..., 1, 2] = f

    img_size = torch.ones(1, 2) * img_size

    # cameras = PerspectiveCameras(R=R, T=T, focal_length=[img_size], principal_point=[(img_size/2, img_size/2)], in_ndc=False, image_size=[(img_size, img_size)])
    cameras = cameras_from_opencv_projection(
        R=R, tvec=T, camera_matrix=K, image_size=img_size
    )

    return cameras


def nerfstudio_camera_to_opencv(pose):
    """
    Convert a pose from NeRFStudio Render Camera coordinate system to opencv camera convention
    """

    # idk why we do this honestly
    pose[:, 1:3] *= -1

    # # Convert to opencv format
    pose = pose_to_opencv(pose)

    # # Convert from c2w to w2c
    pose = np.linalg.inv(pose)

    # # Rotate the FRAME around Z by 180 degrees (again, idk why tbh)
    pose = rotatez180(pose)

    # Rotate teh FRAME around y by 180 degrees
    pose = rotatey180(pose)

    return pose


def pose_to_opencv(pose):
    transform = np.array(
        [
            [1, 0, 0, 0],
            [0, -1, 0, 0],
            [0, 0, -1, 0],
            [0, 0, 0, 1],
        ]
    )

    return transform @ pose


def rotatey180(pose):

    transform = np.array(
        [
            [-1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, -1, 0],
            [0, 0, 0, 1],
        ]
    )

    return pose @ transform


def rotatez180(pose):

    transform = np.array(
        [
            [-1, 0, 0, 0],
            [0, -1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ]
    )

    return pose @ transform


# camera_pos (B,3)
#
# Returns the full camera pose from the camera position in world coordindates
# The camera will be pointed at pointed_at
# Also assumes 0 camera roll, i.e., horizontal image plane axis is orthogonal to vertical direction.
def get_pose_from_camera_torch(
    camera_pos, vertical=np.array((0.0, 0.0, 1.0)), pointed_at=np.array((0, 0, 0.0))
):
    if camera_pos.shape[-2:] == (4, 4):
        camera_pos = camera_pos[..., :3, -1]

    if len(camera_pos.shape) == 1:
        camera_pos = camera_pos[None]
        single_dim = True
    else:
        single_dim = False

    if type(camera_pos) is np.ndarray:
        camera_pos = torch.from_numpy(camera_pos)
        out_numpy = True
    else:
        out_numpy = False

    if type(vertical) is np.ndarray:
        vertical = torch.from_numpy(vertical).to(
            dtype=camera_pos.dtype, device=camera_pos.device
        )
    if type(pointed_at) is np.ndarray:
        pointed_at = torch.from_numpy(pointed_at).to(
            dtype=camera_pos.dtype, device=camera_pos.device
        )

    if len(vertical.shape) == 0:
        vertical = torch.broadcast_to(
            vertical[None], (camera_pos.shape[0], vertical.shape[0])
        )
    if len(pointed_at.shape) == 0:
        pointed_at = torch.broadcast_to(
            pointed_at[None], (camera_pos.shape[0], vertical.shape[0])
        )

    c_dir = pointed_at - camera_pos
    c_dir = c_dir / torch.linalg.norm(c_dir, dim=-1, keepdim=True)

    # The horizontal axis of the camera sensor is horizontal (orthogonal to vertical, z=0) and orthogonal to the view axis
    img_plane_horizontal = torch.linalg.cross(c_dir, vertical)
    img_plane_horizontal = img_plane_horizontal / torch.linalg.norm(
        img_plane_horizontal, dim=-1, keepdim=True
    )
    # The vertical axis is orthogonal to both the view axis and the horizontal axis
    img_plane_vertical = torch.linalg.cross(c_dir, img_plane_horizontal)
    img_plane_vertical = img_plane_vertical / torch.linalg.norm(
        img_plane_vertical, dim=-1, keepdim=True
    )

    # Solve linear system of equations satisfying that each of the orthogonal unit vectors is rotated onto the correct part of the local camera system
    R = torch.stack([img_plane_horizontal, img_plane_vertical, c_dir], -1)

    pose = torch.eye(4, 4, dtype=camera_pos.dtype, device=camera_pos.device)
    pose = torch.broadcast_to(pose[None], (camera_pos.shape[0], 4, 4)).contiguous()

    pose[:, :3, :3] = R
    pose[:, :3, 3] = camera_pos

    if single_dim:
        pose = pose[0]

    if out_numpy:
        pose = pose.cpu().numpy()

    return pose


def find_rotation(x, y):
    """
    Returns a rotation which rotates x onto y.
    https://math.stackexchange.com/questions/180418/calculate-rotation-matrix-to-align-vector-a-to-vector-b-in-3d
    """
    x = x / np.linalg.norm(x)
    y = y / np.linalg.norm(y)
    z = x + y
    return 2 * np.outer(z, z) / np.dot(z, z) - np.eye(x.shape[0])


# find_rotation(np.array([0,1,0]), np.array([1,.1,1.5])) @ np.array([0,1,0])


def closest_ray_intersection(o1, d1, o2, d2):
    ### Equation 11 from SCNeRF. Returns t2 such that o2 + t2 * d2 is closest to the ray o1 + t * d1 in L2 sense.
    t2 = np.dot(np.cross((o1 - o2), d1), np.cross(d1, d2)) / np.dot(
        np.cross(d1, d2), np.cross(d1, d2)
    )
    ### idk why we need to return the -1 * this................
    return -1 * t2


def get_lookat(pose1, pose2):
    """
    Assuming both poses are looking at the same point, finds this point. Finds intersection of the two optical axes.
    """
    cam1 = pose1[:3, -1]
    dir1 = pose1[:3, :3] @ np.array([0, 0, 1])

    cam2 = pose2[:3, -1]
    dir2 = pose2[:3, :3] @ np.array([0, 0, 1])

    t2 = closest_ray_intersection(cam1, dir1, cam2, dir2)

    along2 = cam2 + dir2 * t2

    t1 = closest_ray_intersection(cam2, dir2, cam1, dir1)

    along1 = cam1 + dir1 * t1

    return (along1 + along2) / 2


### Assumes relative_poses[0] is identity and relative_poses has shape (3,4,4)
### Returns circle arc which is interpolating these cameras and looking at the same point
def make_relative_clevr_trajectory(relative_poses, num_frames=60):

    relative_origin = get_lookat(relative_poses[0], relative_poses[1])

    relative_cams = relative_poses[1:, :3, -1]

    centroid = np.mean(relative_poses[:, :3, -1], 0)[None]
    r = np.mean(np.linalg.norm(relative_poses[:, :3, -1] - centroid, axis=-1))

    ### define the circle in plane with z=0
    ts = np.linspace(0, 2 * np.pi, num_frames)
    zs = np.linspace(0, 0, num_frames)
    circle = np.stack([r * np.cos(ts), r * np.sin(ts), zs], -1)

    ### rotate the circle onto the normal
    normal = np.cross(relative_cams[0], relative_cams[1])
    normal = normal / np.linalg.norm(normal)

    R = find_rotation([0, 0, 1], normal)

    circle = (R @ circle.T).T
    ### translate the circle such that its center is between the 3 cameras
    trans_direction = np.cross(normal, relative_cams[0] - relative_cams[1])
    trans_direction = trans_direction / np.linalg.norm(trans_direction)

    circle = circle + r * trans_direction

    return get_pose_from_camera_torch(circle, pointed_at=relative_origin)


def compare_poses(pose1, pose2):
    """
    Compares two poses given as 4x4 matrices and determines the relative position and gaze direction.

    Args:
    pose1: The first pose matrix.
    pose2: The second pose matrix.

    Returns:
    A dictionary containing the following keys:
        position: A string indicating the relative position ("left", "right", "up", "down", or "none").
        gaze: A string indicating the relative gaze direction ("left", "right", "up", "down", or "none").
    """

    # Extract translation vectors.
    translation1 = pose1[:3, 3]
    translation2 = pose2[:3, 3]

    # Extract rotation matrices.
    rotation1 = pose1[:3, :3]
    rotation2 = pose2[:3, :3]

    # Calculate relative translation.
    relative_translation = translation2 - translation1

    # Calculate relative gaze direction - as rotation around Y axis.
    gaze1 = rotation1 @ np.array([0, 0, 1])
    gaze2 = rotation2 @ np.array([0, 0, 1])

    angle = np.arccos(
        np.dot(gaze1, gaze2) / (np.linalg.norm(gaze1) * np.linalg.norm(gaze2))
    )

    print(angle * 180 / np.pi, "is the angle")

    # Determine relative position.
    position = "none"
    if abs(relative_translation[0]) > 0.1:
        position = "left" if relative_translation[0] < 0 else "right"
    elif abs(relative_translation[1]) > 0.1:
        position = "down" if relative_translation[1] < 0 else "up"
    elif abs(relative_translation[2]) > 0.1:
        position = "back" if relative_translation[2] < 0 else "forward"

    # Determine relative gaze direction.
    gaze = "none"
    if angle > 0:
        gaze = "left"
    elif angle < 0:
        gaze = "right"

    return {"position": position, "gaze": gaze}


if __name__ == "__main__":

    poses = np.load("data/poses/look_left_right_back_left_right.npy")

    # Example usage:
    pose1 = poses[0]
    pose2 = poses[2]

    result = compare_poses(pose1, pose2)
    print("Relative position:", result["position"])
    print("Relative gaze direction:", result["gaze"])
