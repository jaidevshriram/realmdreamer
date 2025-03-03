import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import KDTree
from scipy.spatial import ConvexHull
import json
import argparse
from pathlib import Path

def compute_adaptive_ellipsoid_params(point_cloud, safety_factor=1.5):
    centered = point_cloud - np.mean(point_cloud, axis=0)
    pca = PCA(n_components=3)
    pca.fit(centered)
    
    transformed = pca.transform(centered)
    extents = np.max(transformed, axis=0) - np.min(transformed, axis=0)
    
    axes_lengths = safety_factor * extents
    min_axis_length = 2.0
    axes_lengths = np.maximum(axes_lengths, min_axis_length)
    
    return {
        'center': np.mean(point_cloud, axis=0),
        'axes_lengths': axes_lengths,
        'orientation': pca.components_
    }

def calculate_num_poses(point_cloud, min_poses=12, max_poses=48, base_volume=27):
    hull = ConvexHull(point_cloud)
    scene_volume = hull.volume
    
    volume_ratio = (scene_volume / base_volume) ** (1/3)
    num_poses = int(min_poses + (max_poses - min_poses) * min(1.0, volume_ratio))
    
    return num_poses

def check_pose_validity(pose, point_cloud, min_distance=0.5):
    tree = KDTree(point_cloud)
    
    # Check immediate surroundings
    neighbors = tree.query_radius([pose['position']], r=min_distance)
    if len(neighbors[0]) > 0:
        return False
    
    # Check viewing direction occlusions
    forward = -pose['rotation'][2]
    max_view_distance = 20.0
    
    cone_points = tree.query_radius([pose['position']], r=max_view_distance)[0]
    
    if len(cone_points) > 0:
        points = point_cloud[cone_points]
        to_points = points - pose['position']
        distances = np.linalg.norm(to_points, axis=1)
        
        dots = np.dot(to_points, forward)
        angles = np.arccos(dots / distances)
        
        view_mask = angles < np.pi/4
        if view_mask.any():
            close_points = distances[view_mask] < min_distance
            if close_points.any():
                return False
    
    return True

def generate_ellipsoid_poses(point_cloud):
    ellipsoid = compute_adaptive_ellipsoid_params(point_cloud)
    num_poses = calculate_num_poses(point_cloud)
    
    golden_angle = np.pi * (3 - np.sqrt(5))
    theta = golden_angle * np.arange(num_poses)
    z = np.linspace(1 - 1/num_poses, 1/num_poses - 1, num_poses)
    radius = np.sqrt(1 - z*z)
    
    poses = []
    for i in range(num_poses):
        point = np.array([
            radius[i] * np.cos(theta[i]),
            radius[i] * np.sin(theta[i]),
            z[i]
        ])
        
        scaled_point = point * ellipsoid['axes_lengths']
        rotated_point = np.dot(ellipsoid['orientation'], scaled_point)
        position = rotated_point + ellipsoid['center']
        
        forward = ellipsoid['center'] - position
        forward = forward / np.linalg.norm(forward)
        
        up = np.array([0, 0, 1])
        right = np.cross(forward, up)
        if np.allclose(right, 0):
            up = np.array([0, 1, 0])
            right = np.cross(forward, up)
        right = right / np.linalg.norm(right)
        up = np.cross(right, forward)
        
        rotation = np.stack([right, up, -forward])
        
        pose = {
            'position': position,
            'rotation': rotation
        }
        
        if check_pose_validity(pose, point_cloud):
            poses.append(pose)
    
    return poses

def create_transform_matrix(position, rotation):
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = position
    return transform

def main():
    parser = argparse.ArgumentParser(description='Generate viewing poses for a point cloud')
    parser.add_argument('point_cloud_path', type=str, help='Path to point cloud file (.npy)')
    parser.add_argument('config_path', type=str, help='Path to camera configuration JSON')
    parser.add_argument('output_path', type=str, help='Path to save output JSON')
    
    args = parser.parse_args()
    
    # Load point cloud
    point_cloud = np.load(args.point_cloud_path)
    
    # Load existing configuration
    with open(args.config_path, 'r') as f:
        config = json.load(f)
    
    # Generate new poses
    poses = generate_ellipsoid_poses(point_cloud)
    
    # Create new frames
    new_frames = []
    for i, pose in enumerate(poses, start=len(config['frames'])):
        transform_matrix = create_transform_matrix(pose['position'], pose['rotation'])
        
        frame = {
            'file_path': f'rgb/{i}.png',
            'depth_file_path': f'depth/{i}.npy',
            'mask_path': f'mask/{i}.png',
            'mask_inpainting_file_path': f'mask/{i}.png',
            'transform_matrix': transform_matrix.tolist()
        }
        new_frames.append(frame)
    
    # Update configuration
    config['frames'].extend(new_frames)
    
    # Save updated configuration
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(config, f, indent=4)

if __name__ == '__main__':
    main()