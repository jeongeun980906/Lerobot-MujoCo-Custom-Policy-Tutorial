import pygame
import numpy as np
class controller:
    def __init__(self, pose_gain, rot_gain):
        self.pose_gain = pose_gain
        self.rot_gain = rot_gain
        pygame.init()
        pygame.joystick.init()
        self.conntect_joystick()
        self.gripper_state = False
        self.past_button = False
        self.past_fail = False

    def reset(self, env):
        self.past_button = False
        self.past_fail = False
        self.gripper_state = False

    def conntect_joystick(self):
        for event in pygame.event.get():
            if event.type == pygame.JOYDEVICEADDED:
                self.joystick = pygame.joystick.Joystick(event.device_index)

    def get_action(self):
        self.conntect_joystick()
        x = self.joystick.get_axis(1)
        y = self.joystick.get_axis(0)
        z_plus =  self.joystick.get_button(4) 
        z_minus =(self.joystick.get_axis(2)+1)/2 
        z = z_plus - z_minus
        roll_plus = self.joystick.get_button(3) #(self.joystick.get_axis(5)+1)/2
        roll_minus = self.joystick.get_button(5)
        roll = roll_plus - roll_minus
        pitch = self.joystick.get_axis(4)
        yaw = self.joystick.get_axis(3)

        dpos = np.array([x, y, z]) * self.pose_gain
        drot = np.array([-roll, pitch, yaw]) * self.rot_gain

        button = self.joystick.get_button(2)
        if button and button != self.past_button:
            self.gripper_state = not self.gripper_state

        self.past_button = button
        reset = self.joystick.get_button(1)
        action = np.concatenate([dpos, drot, [self.gripper_state]], dtype=np.float32)
        fail = self.joystick.get_button(0)
        if fail and fail != self.past_fail:
            fail_ret = True
        else:
            fail_ret = False
        self.past_fail = fail
        return action, reset, fail_ret
    
    def close(self):
        pygame.quit()