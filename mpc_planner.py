import numpy as np
from cem_optimization import cem_optimization
import mujoco
import time
import os
from mujoco import viewer
import matplotlib.pyplot as plt
from utils.quat_math import rotation_quaternion, quaternion_multiply, quaternion_distance


class MPC_Planner():
    def __init__(self):

        self.cem = None
        self.model = None
        self.data = None
        self.xi_mean = None
        self.target_pos = None
        self.target_rot = None
        self.init_position = None
        self.init_rotation = None
        self.init_joint_position = np.array([1.5, -1.8, 1.75, -1.25, -1.6, 0])
        self.info = dict(
            cost_g_list = list(),
            cost_list = list(),
            cost_r_list = list(),
            cost_c_list = list(),
            thetadot_list = list(),
            theta_list = list()
        )


    def init_cem(self):
        start_time = time.time()
        self.cem =  cem_optimization(num_dof=6, num_batch=1000, num_steps=8, maxiter_cem=1,
                           w_pos=5, w_rot=1.5, w_col=5, num_elite=0.05, timestep=0.05)
        print(f"Initialized CEM Planner: {round(time.time()-start_time, 2)}s")

        self.model = self.cem.model
        self.data = self.cem.data
        self.data.qpos[:6] = np.array(self.init_joint_position)
        mujoco.mj_forward(self.model, self.data)

        self.xi_mean = np.zeros(self.cem.nvar)
        self.target_pos = self.model.body(name="target_0").pos
        self.target_rot = self.model.body(name="target_0").quat

        self.init_position = self.data.site_xpos[self.model.site(name="tcp").id].copy()
        self.init_rotation = self.data.xquat[self.model.body(name="hande").id].copy()


    def run_mpc(self):
        thetadot = np.zeros(6)

        current_pos = self.init_joint_position
        current_vel = thetadot

        start_time = time.time()
        _ = self.cem.compute_cem(xi_mean=self.xi_mean, 
                                      init_pos=current_pos, init_vel=current_vel, 
                                      target_pos=self.target_pos, target_rot=self.target_rot)
        print(f"Compute CEM: {round(time.time()-start_time, 2)}s")

        init_position = self.data.site_xpos[self.model.site(name="tcp").id].copy()
        init_rotation = self.data.xquat[self.model.body(name="hande").id].copy()

        target_positions = [
            [-0.3, 0.3, 0.8],
            [-0.2, -0.4, 1.0],
            [-0.3, -0.1, 0.8],
            init_position
        ]

        target_rotations = [
            rotation_quaternion(-135, np.array([1,0,0])),
            quaternion_multiply(rotation_quaternion(90, np.array([0,0,1])),rotation_quaternion(135, np.array([1,0,0]))),
            quaternion_multiply(rotation_quaternion(180, np.array([0,0,1])),rotation_quaternion(-90, np.array([0,1,0]))),
            init_rotation
        ]

        target_idx = 0        
        with viewer.launch_passive(self.model, self.data) as viewer_:
            viewer_.cam.distance = 4
            viewer_.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
            viewer_.opt.sitegroup[:] = False  
            viewer_.opt.sitegroup[1] = True 

            while viewer_.is_running():
                start_time = time.time()

                current_pos = self.data.qpos[:6]
                current_vel = self.data.qvel[:6]


                cost, cost_g, cost_r, cost_c, thetadot, theta, self.xi_mean = self.cem.compute_cem(xi_mean=self.xi_mean, 
                                      init_pos=current_pos, init_vel=current_vel, 
                                      target_pos=self.target_pos, target_rot=self.target_rot)
                                
                thetadot = np.mean(thetadot[1:5], axis=0)

                self.data.qvel[:6] = thetadot
                mujoco.mj_step(self.model, self.data)

                cost_g = np.linalg.norm(self.data.site_xpos[self.cem.tcp_id] - self.target_pos)   
                cost_r = quaternion_distance(self.data.xquat[self.cem.hande_id], self.target_rot)  
                cost = np.round(cost, 2)
                viewer_.sync()


                self.info['cost_g_list'].append(cost_g)
                self.info['cost_r_list'].append(cost_r)
                self.info['cost_c_list'].append(cost_c)
                self.info['thetadot_list'].append(thetadot)
                self.info['theta_list'].append(self.data.qpos[:6].copy())
                self.info['cost_list'].append(cost[-1])

                time_until_next_step = self.model.opt.timestep - (time.time() - start_time)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step) 

                if cost_g<0.01 and cost_r<0.3:
                    self.model.body(name="target_0").pos = target_positions[target_idx]
                    self.model.body(name="target_0").quat = target_rotations[target_idx]
                    if target_idx<len(target_positions)-1:
                        target_idx += 1
                
                print(f'Step Time: {"%.0f"%((time.time() - start_time)*1000)}ms | Cost g: {"%.2f"%(float(cost_g))} | Cost r: {"%.2f"%(float(cost_r))} | Cost c: {"%.2f"%(float(cost_c))} | Cost: {cost}')


    def save_info(self):
        np.savetxt('data/costs.csv',self.info['cost_list'], delimiter=",")
        np.savetxt('data/thetadot.csv',self.info['thetadot_list'], delimiter=",")
        np.savetxt('data/theta.csv',self.info['theta_list'], delimiter=",")
        np.savetxt('data/cost_g.csv',self.info['cost_g_list'], delimiter=",")
        np.savetxt('data/cost_r.csv',self.info['cost_r_list'], delimiter=",")
        np.savetxt('data/cost_c.csv',self.info['cost_c_list'], delimiter=",")


def main():
    mpc = MPC_Planner()
    mpc.init_cem()
    mpc.run_mpc()
    mpc.save_info()

if __name__=="__main__":
    main()