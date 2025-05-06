import numpy as np
from cem_optimization import cem_optimization
import mujoco.mjx as mjx 
import mujoco
import time
import jax.numpy as jnp
import jax
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
        self.cem =  cem_optimization(num_dof=6, num_batch=500, num_steps=8, maxiter_cem=1, num_elite=0.05, timestep=0.05,
                                     eef_to_obj=5, obj_to_goal=7, push_align=0.2, obj_rot=0, eef_rot=2, collision=10)
        print(f"Initialized CEM Planner: {round(time.time()-start_time, 2)}s")

        self.model = self.cem.model
        self.data = self.cem.data
        self.data.qpos[:6] = jnp.array(self.init_joint_position)
        mujoco.mj_forward(self.model, self.data)

        self.xi_mean = jnp.zeros(self.cem.nvar)
        self.target_pos = self.model.body(name="target_1").pos
        self.target_rot = self.model.body(name="target_1").quat

        self.init_position = self.data.xpos[self.model.body(name="target_0").id].copy()
        self.init_rotation = self.data.xquat[self.model.body(name="target_0").id].copy()



    def run_mpc(self):
        thetadot = np.zeros(6)

        current_pos = self.init_joint_position
        current_vel = thetadot

        start_time = time.time()
        _ = self.cem.compute_cem(xi_mean=self.xi_mean, 
                                      init_pos=current_pos, init_vel=current_vel, 
                                      target_pos=self.target_pos, target_rot=self.target_rot, 
                                      init_target_pos=self.init_position, init_target_rot=self.init_rotation)
        print(f"Compute CEM: {round(time.time()-start_time, 2)}s")

        target_positions = [
            [-0.2, -0.3, 0.4],
            [-0.2, 0.3, 0.4],
            [-0.2, 0.0, 0.4],
        ]

        target_rotations = [
            quaternion_multiply(self.init_rotation,rotation_quaternion(-30, np.array([0,0,1]))),
            quaternion_multiply(self.init_rotation,rotation_quaternion(45, np.array([0,0,1]))),
            quaternion_multiply(self.init_rotation,rotation_quaternion(-20, np.array([0,0,1]))),
        ]
        target_idx = 0
        
        with viewer.launch_passive(self.model, self.data) as viewer_:
            viewer_.cam.distance = 4
            # viewer_.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True

            while viewer_.is_running():
                start_time = time.time()

                current_pos = self.data.qpos[:6]
                current_vel = self.data.qvel[:6]

                self.target_pos = self.model.body(name="target_1").pos
                self.target_rot = self.model.body(name="target_1").quat


                cost, cost_c, thetadot, theta, self.xi_mean = self.cem.compute_cem(xi_mean=self.xi_mean, 
                                      init_pos=current_pos, init_vel=current_vel, 
                                      target_pos=self.target_pos, target_rot=self.target_rot,
                                      init_target_pos=self.init_position, init_target_rot=self.init_rotation)
                
                                
                thetadot = np.mean(thetadot[1:5], axis=0)

                self.data.qvel[:6] = thetadot
                mujoco.mj_step(self.model, self.data)

                self.init_position = self.data.xpos[self.model.body(name="target_0").id].copy()
                self.init_rotation = self.data.xquat[self.model.body(name="target_0").id].copy()

                cost_g = np.linalg.norm(self.init_position[:-1] - self.target_pos[:-1])   
                cost_r = quaternion_distance(self.init_rotation, self.target_rot)  
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

                if cost_g<0.03:
                    self.model.body(name="target_1").pos = target_positions[target_idx]
                    self.model.body(name="target_1").quat = target_rotations[target_idx]
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