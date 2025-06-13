import os
import numpy as np
import time


import mujoco
from mujoco import viewer
from utils.quat_math import rotation_quaternion, quaternion_multiply, quaternion_distance

target_positions_1 = [
            [-0.3, 0.3, 0.8],
            [-0.2, -0.4, 1.0],
            [-0.3, -0.1, 0.8],
        ]

target_rotations_1 = [
    rotation_quaternion(-135, np.array([1,0,0])),
    quaternion_multiply(rotation_quaternion(90, np.array([0,0,1])),rotation_quaternion(135, np.array([1,0,0]))),
    quaternion_multiply(rotation_quaternion(180, np.array([0,0,1])),rotation_quaternion(-90, np.array([0,1,0]))),
]

target_positions_2 = [
    [-0.3, -0.3, 0.8],
    [-0.2, 0.4, 1.0],
    [-0.3, 0.1, 0.8],
]

target_rotations_2 = [
    quaternion_multiply(rotation_quaternion(180, np.array([0,0,1])), rotation_quaternion(135, np.array([1,0,0]))),
    quaternion_multiply(rotation_quaternion(-90, np.array([0,0,1])),rotation_quaternion(-135, np.array([1,0,0]))),
    quaternion_multiply(rotation_quaternion(0, np.array([0,0,1])),rotation_quaternion(90, np.array([0,1,0]))),
]


class Visualizer():
    def __init__(self, ctrl: bool=False, traj: bool=False):
        self.init_joint_state = np.array([1.5, -1.8, 1.75, -1.25, -1.6, 0, -1.5, -1.8, 1.75, -1.25, -1.6, 0])

        if ctrl:
            model_path = f"{os.path.dirname(__file__)}/ur5e_hande_mjx/scene_control.xml" 
        else:
            model_path = f"{os.path.dirname(__file__)}/ur5e_hande_mjx/scene.xml" 

        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.model.opt.timestep = 0.002
        self.data = mujoco.MjData(self.model)
        self.data.qpos[:12] = self.init_joint_state
        target_idx = 2
        self.model.body(name="target_0").pos = target_positions_1[target_idx]
        self.model.body(name="target_0").quat = target_rotations_1[target_idx]
        self.model.body(name="target_1").pos = target_positions_2[target_idx]
        self.model.body(name="target_1").quat = target_rotations_2[target_idx]

        if traj:
            file_path = f"{os.path.dirname(__file__)}/data/thetadot.csv" 
            self.thetadot = np.genfromtxt(file_path, delimiter=',')


        


    def view_model(self):
        viewer.launch(self.model, self.data)

    def view_traj_mujoco(self):
        with viewer.launch_passive(self.model, self.data) as viewer_:
            viewer_.cam.distance = 4
            viewer_.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
            viewer_.opt.sitegroup[:] = False  
            viewer_.opt.sitegroup[1] = True 

            i = 0
            while viewer_.is_running():
                step_start = time.time()
                self.data.qvel[:6] = self.thetadot[i]

                mujoco.mj_step(self.model, self.data)
                viewer_.sync()

                time_until_next_step = self.model.opt.timestep - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)    

                if i < self.thetadot.shape[0]-1:
                    i+=1
                else:
                    self.data.qvel[:6] = np.zeros(6)
                    self.data.qpos[:6] = self.init_joint_state
                    i=0




def main():
    viz = Visualizer(ctrl=False, traj=False)
    viz.view_model()
    # viz.view_traj_mujoco()


if __name__=="__main__":
    main()