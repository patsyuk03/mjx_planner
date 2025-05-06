import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from rclpy.qos import QoSProfile, QoSReliabilityPolicy
from ament_index_python.packages import get_package_share_directory
import os

import urx
import time

import mujoco
from mujoco import viewer
import numpy as np

from mj_planner.mjx_planner import cem_planner





class MocapListener(Node):
    def __init__(self):
        super().__init__('mocap_listener')
        qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            depth=10
        )

        self.cem = None
        self.model = None
        self.viewer = None
        self.data = None
        self.xi_mean = None
        self.target_pos = None
        self.target_rot = None
        self.init_position = None
        self.init_rotation = None
        self.init_joint_position = [-1.52, -1.26, -1.75, -1.69,  1.64, 0.2]

        connected = False
        try:
            self.rob = urx.Robot("192.168.0.120")
            connected = True
            print("Connection with UR5e established.")
        except:
            print("Could not connect...")

        if connected:
            self.move_to_start()
            self.init_cem()
            self.subscription_object1 = self.create_subscription(
                PoseStamped,
                '/vrpn_mocap/object1/pose',
                self.object1_callback,
                qos_profile 
            )
            # self.run_mpc()
            self.timer = self.create_timer(0.1, self.run_mpc)
            # self.close_connection()

    def move_to_start(self):
        self.rob.movej(self.init_joint_position, acc=0.5, vel=0.5, wait=True)
        print("Moved to initial pose.")

    def close_connection(self):
        self.rob.stopj()
        self.rob.close()
        print("Disconnected from UR5 Robot")

    def init_cem(self):
        start_time = time.time()
        self.cem =  cem_planner(num_dof=6, num_batch=800, num_steps=8, maxiter_cem=1,
                           w_pos=5, w_rot=1.5, w_col=10, num_elite=0.01, timestep=0.05)
        print(f"Initialized CEM Planner: {round(time.time()-start_time, 2)}s")

        self.model = self.cem.model
        self.data = self.cem.data
        self.data.qpos[:6] = np.array(self.init_joint_position)
        mujoco.mj_forward(self.model, self.data)

        self.xi_mean = np.zeros(self.cem.nvar)
        self.target_pos = self.model.body(name="target").pos
        self.target_rot = self.model.body(name="target").quat

        self.init_position = self.data.site_xpos[self.model.site(name="tcp").id].copy()
        self.init_rotation = self.data.xquat[self.model.body(name="hande").id].copy()

        self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        self.viewer.cam.lookat[:] = [0.0, 0.0, 0.8]  
        self.viewer.cam.distance = 5.0 
        self.viewer.cam.azimuth = 90.0 
        self.viewer.cam.elevation = -30.0 

        start_time = time.time()
        _ = self.cem.compute_cem(xi_mean=self.xi_mean, 
                                      init_pos=np.array(self.rob.getj()), init_vel=np.zeros(6), 
                                      target_pos=self.target_pos, target_rot=self.target_rot)
        print(f"Compute CEM: {round(time.time()-start_time, 2)}s")

    def run_mpc(self):
        thetadot = np.zeros(6)

        start_time = time.time()

        current_pos = np.array(self.rob.getj())
        current_vel = thetadot


        self.target_pos = self.model.body(name="target").pos
        self.target_rot = self.model.body(name="target").quat


        cost, best_cost_g, best_cost_c, best_vels, best_traj, self.xi_mean = self.cem.compute_cem(xi_mean=self.xi_mean, 
                                init_pos=current_pos, init_vel=current_vel, 
                                target_pos=self.target_pos, target_rot=self.target_rot)
        
        thetadot = np.mean(best_vels[1:5], axis=0)

        # print(thetadot)

        # s_time = time.time()
        self.rob.speedj(thetadot, acc=1, min_time=0.2)
        # print(f'Time: {"%.0f"%((time.time() - s_time)*1000)}ms')

        self.data.qpos[:6] = self.rob.getj()
        mujoco.mj_step(self.model, self.data)

        cost_g = np.linalg.norm(self.data.site_xpos[self.cem.tcp_id] - self.target_pos)   
        self.viewer.sync()
        
        print(f'Step Time: {"%.0f"%((time.time() - start_time)*1000)}ms | Cost g: {"%.2f"%(float(cost_g))}')




    # def update_viewer(self, body_name, pose):
    #     self.model.body(name=body_name).pos = [-pose.position.x, -pose.position.y, pose.position.z]
    #     mujoco.mj_step(self.model, self.data)
    #     self.viewer.sync()



    def object1_callback(self, msg):
        self.get_logger().info(
            f"Received Pose:\n"
            f"  Position -> x: {msg.pose.position.x}, y: {msg.pose.position.y}, z: {msg.pose.position.z}\n"
        )
        pose = msg.pose
        self.model.body(name='target').pos = [-pose.position.x, -pose.position.y, pose.position.z]


def main(args=None):
    rclpy.init(args=args)
    mocap_listener = MocapListener()
    print("Initialized node.")
    rclpy.spin(mocap_listener)

    mocap_listener.close_connection()

    mocap_listener.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
