import numpy as np

def get_rays(height, width, intrinsic, extrinsic):
    
    origin = extrinsic[:3, 3]
    
    u, v = np.meshgrid(np.arange(width), np.arange(height))
    pixels = np.stack([u, v, np.ones_like(u)], axis=-1).reshape(-1, 3)
    
    rays_dir_camera = np.dot(np.linalg.inv(intrinsic), pixels.T)
    rays_dir_world = extrinsic[:3, :3] @ rays_dir_camera
    
    return origin, rays_dir_world.T

def projection_matrix_to_intrinsic(projection_matrix, width, height):

    fx = projection_matrix[0, 0] * (width / 2)
    fy = projection_matrix[1, 1] * (height / 2)
    cx = (1 - projection_matrix[0, 2]) * (width / 2)
    cy = (1 - projection_matrix[1, 2]) * (height / 2)

    return np.float32([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])