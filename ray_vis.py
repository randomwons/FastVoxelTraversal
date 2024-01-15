import threading

import numpy as np
from open3d.visualization import gui
from open3d.visualization import rendering

from ray_voxel_traversal import RayTraversalVoxel
from utils import projection_matrix_to_intrinsic, get_rays

WINDOW_WIDTH = 1280 
WINDOW_HEIGHT = 720
FOV = 60
VOXEL_BNDS = [[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]]
VOXEL_SIZE = 0.01

class RayVisApp:
    def __init__(self):
        self.flag_loop = True
        self.flag_ray_shot = False
    
    def run(self):
        
        app = gui.Application.instance
        app.initialize()
        self.__create_window()
        threading.Thread(target=self.context, daemon=True).start()
        app.run()
        
    def __create_window(self):
        
        self.window = gui.Application.instance.create_window(
            "Fast Ray Voxel Traversal", WINDOW_WIDTH, WINDOW_HEIGHT
        )
        self.__gui_layout()
        self.window.set_on_layout(self.__layout)
        self.window.set_on_key(self.__set_key)
        self.window.set_on_close(self.__on_close)

    def __on_close(self):
        self.flag_loop = False
        return True

    def __gui_layout(self):
        self.widget3d = gui.SceneWidget()
        self.widget3d.scene = rendering.Open3DScene(self.window.renderer)
        self.widget3d.scene.set_background([1, 1, 1, 1])
        self.widget3d.scene.show_axes(True)
        self.widget3d.setup_camera(FOV, self.widget3d.scene.bounding_box, [0, 0, 0])
        self.widget3d.scene.camera.look_at([0, 0, 0], [0, 0, 2], [0, 1, 0])
        
        self.window.add_child(self.widget3d) 
    
    def __layout(self, ctx):
        
        rect = self.window.content_rect
        self.widget3d.frame = gui.Rect(rect.x, rect.y, rect.width, rect.height)
        self.intrinsic = projection_matrix_to_intrinsic(self.widget3d.scene.camera.get_projection_matrix(), 
                                                        self.widget3d.frame.width, 
                                                        self.widget3d.frame.height)

    
    def context(self):
        
        volume = RayTraversalVoxel(VOXEL_BNDS, VOXEL_SIZE)
        bb = volume.get_bounding_box()
        gui.Application.instance.post_to_main_thread(
            self.window, lambda : self.update_geometry("bb", bb)
        )
        
        while self.flag_loop:
            if self.flag_ray_shot:
                
                pose = self.get_pose_from_widget()
                
                ray_origin, rays_dir = get_rays(self.widget3d.frame.height,
                                                self.widget3d.frame.width,
                                                self.intrinsic,
                                                np.linalg.inv(pose))
                
                
                index = self.widget3d.frame.width * (self.widget3d.frame.height // 2) + (self.widget3d.frame.width // 2)
                volume.ray_traversal(ray_origin, rays_dir[index])
                
                voxel = volume.get_traversed_voxel()
                gui.Application.instance.post_to_main_thread(
                    self.window, lambda : self.update_geometry("voxel", voxel)
                )
                
                
                self.flag_ray_shot = False
        
    def get_pose_from_widget(self):
        
        view_matrix = self.widget3d.scene.camera.get_view_matrix()
        pose = np.float32([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]) @ view_matrix
        return pose
     
    
    def update_geometry(self, name, geo):
        self.remove_geometry(name)
        mat = rendering.MaterialRecord()        
        self.widget3d.scene.add_geometry(name, geo, mat)
        
    def remove_geometry(self, name):
        if self.widget3d.scene.has_geometry(name):
            self.widget3d.scene.remove_geometry(name)
    
    def __set_key(self, keyevent):
        if keyevent.type == keyevent.DOWN:
            if keyevent.key == ord('r'):
                self.flag_ray_shot = True
    
        
    
    
if __name__ == "__main__":
    app = RayVisApp()
    app.run()