import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
import message_filters
import numpy as np
import cv2, os, sys
import time
import zmq
from src.mujoco_helper.mujoco_parser import MuJoCoParserClass
JOINT_NAMES = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6","rh_r1_joint"]


ZERO_POSE = np.array(
    [-0.027611654150390626,  # joint1
    -1.5447186516357423,    # joint2
    2.6612221080117244,     # joint3
    -1.136679762524414,     # joint4
    1.5884457219101125,     # joint5
    -0.0122718462890625,     # joint6,
    0.4086213869140625], dtype=np.float32
)

class Leader(Node):
    def __init__(self):
        super().__init__('synchronized_subscriber')
        self.sub = self.create_subscription(JointState, '/leader/joint_states',self.callback, 10)

        self.env = MuJoCoParserClass('leader','asset/example_scene_y.xml')
        self.env.init_viewer()

        self.context_ = zmq.Context()
        self.socket = self.context_.socket(zmq.PUB)
        self.socket.connect("tcp://127.0.0.1:5555")  # connect

        self.timer = self.create_timer(1/100, self.timer_callback)

    def callback(self, msg):
        joint_names = msg.name
        joint_q = msg.position
        unsorted_mapping = dict(zip(joint_names, joint_q))
            
        # Using the desired order, reassemble the joint positions.
        sorted_positions = [unsorted_mapping[j] for j in JOINT_NAMES if j in unsorted_mapping]
        sorted_positions =  np.array(sorted_positions,dtype=np.float32)

        self.env.forward(sorted_positions)
        p, R = self.env.get_pR_body('eef_pose')
        self.q = sorted_positions
        self.env.plot_T(p,R)
        self.env.render()
        self.p = p
        self.R = R
        if sorted_positions[-1] < -0.4:
            self.gripper = 1
        else:
            self.gripper = 0 
    def timer_callback(self):
        try: self.p
        except: return
        data = {'p':self.p,'R':self.R, 'gripper': self.gripper, 'qpos': self.q}
        
        self.socket.send_pyobj(data)  # send Pythong objects
        print("SEND")


def main(args=None):
    rclpy.init(args=args)
    node = Leader()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
