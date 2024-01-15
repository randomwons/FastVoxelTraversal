import numpy as np
import open3d as o3d

class RayTraversalVoxel:
    def __init__(self, voxel_bnds, voxel_size):
        
        self.voxel_bnds = np.float32(voxel_bnds)
        self.voxel_size = voxel_size
        
        self.voxel_dims = np.ceil((self.voxel_bnds[1] - self.voxel_bnds[0]) / self.voxel_size).astype(int)
        self.voxel_bnds[1] = self.voxel_bnds[0] + self.voxel_size * self.voxel_dims
        self.voxel_origin = self.voxel_bnds[0]
    
        ### Intialize Volumes
        self.volume = np.zeros(self.voxel_dims).astype(np.float32)
        
        ### Get voxel coords and world points
        ## shape (prod(voxel_dims), 3)
        self.voxel_coords = get_grid_coords(self.voxel_dims)
        self.world_pts = self.voxel_origin + voxel_size * self.voxel_coords
    
    def ray_traversal(self, ray_origin, ray_dir, ray_max=5):
                
        zero_mask = (ray_dir == 0)

        start_point = ray_origin
        end_point = ray_origin + ray_dir * ray_max
        
        start_voxel_index = np.floor((start_point - self.voxel_origin) / self.voxel_size)
        end_voxel_index = np.floor((end_point - self.voxel_origin) / self.voxel_size)

        step = np.sign(ray_dir).astype(np.int32)
        
        next_voxel_boundary = (start_voxel_index + (step + 1) / 2) * self.voxel_size + self.voxel_origin
        
        ray_dir[zero_mask] = 1 # zero divide
        tMax = (next_voxel_boundary - ray_origin) / ray_dir
        tMax[zero_mask] = np.inf

        tDelta = self.voxel_size / ray_dir * step
        
        travsered_index = [start_voxel_index]
        
        current_voxel_index = start_voxel_index.copy()
        while (end_voxel_index != current_voxel_index).any():

            min_index = np.argsort(tMax)[0]
            current_voxel_index[min_index] += step[min_index]
            tMax[min_index] += tDelta[min_index]
            travsered_index.append(current_voxel_index.copy())

        travsered_index = np.int32(travsered_index)
        x, y, z = travsered_index.T
        valid_indices = (0 <= x) & (x < self.voxel_dims[0]) & (0 <= y) & (y < self.voxel_dims[1]) & (0 <= z) & (z < self.voxel_dims[2])
        self.volume[x[valid_indices], y[valid_indices], z[valid_indices]] = 1.5

    def get_bounding_box(self):
        
        bb = o3d.geometry.AxisAlignedBoundingBox(self.voxel_bnds[0].reshape(-1, 1),
                                                 self.voxel_bnds[1].reshape(-1, 1))
        bb.color = [0, 0, 0]
        return bb
    
    def get_traversed_voxel(self):
        
        pcd = o3d.geometry.PointCloud()
        
        traversed = (self.volume.ravel() >= 1.0)
        pcd.points = o3d.utility.Vector3dVector(self.world_pts[traversed])
        pcd.paint_uniform_color([0, 0, 1])

        voxel = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, self.voxel_size)

        return voxel 
    
                    
def get_grid_coords(dims):
    
    xv, yv, zv = np.meshgrid(
        range(dims[0]),
        range(dims[1]),
        range(dims[2]),
        indexing='ij'
    )
    
    grid_coords = np.concatenate([
        xv.reshape(1, -1),
        yv.reshape(1, -1),
        zv.reshape(1, -1)
    ], axis=0).astype(int).T
    
    return grid_coords

if __name__ == "__main__":
    
    ray_origin = np.float32([0.9472733, 0.6857862, 1.1274253])
    ray_dir = np.float32([-0.57133539, -0.36322167, -0.73596584])

    VOXEL_BNDS = [[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]]
    VOXEL_SIZE = 0.01

    volume = RayTraversalVoxel(VOXEL_BNDS, VOXEL_SIZE)
    volume.ray_traversal(ray_origin, ray_dir)