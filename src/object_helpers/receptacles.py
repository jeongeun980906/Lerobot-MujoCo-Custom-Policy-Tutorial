import numpy as np
'''
Code for handling receptacle objects like cabinets and microwaves.
Each receptacle class has methods to check if it's in a certain state (open/close),
reset its state, and get/set its joint states.
'''
class Microwave():
    def __init__(self, env):
        self.env = env
        self.obj_name = 'body_microwave'
        self.door_state = False
        
    def check_range(self, condition="open"):
        joint = self.env.get_qpos_joint('microjoint')[0]
        if condition == 'open' and joint < -1.2: return True
        if condition == 'close' and joint > -0.05: return True
        else: return False
    
    def reset(self, condition='close'):
        self.door_state = False
        joint_idx = self.env.get_idxs_fwd(joint_names=["microjoint"])[0]
        if condition == 'open':
            self.env.data.qpos[joint_idx] = -1.8
        elif condition == 'close':
            self.env.data.qpos[joint_idx] = 0.0
        else:
            raise NotImplementedError
        
    def get_q_state(self):
        joint_state = self.env.get_qpos_joint('microjoint')[0]
        return {'microjoint': joint_state}
    def set_q_state(self, joint_states):
        joint_idx = self.env.get_idxs_fwd(joint_names=["microjoint"])[0]
        self.env.data.qpos[joint_idx] = joint_states['microjoint']

        

class WhiteSlideCabinet():
    def __init__(self, env):
        self.env = env
        self.obj_name = 'body_white_slide_cabinet'
        self.door_state = False
        
    def check_range(self, condition="open"):
        joint = self.env.get_qpos_joint('slidedoor_joint')[0]
        if condition == 'open' and joint > 0.18: return True
        if condition == 'close' and joint < 0.05: return True
        else: return False
    
    def reset(self, condition='close'):
        self.door_state = False
        joint_idx = self.env.get_idxs_fwd(joint_names=["slidedoor_joint"])[0]
        if condition == 'open':
            self.env.data.qpos[joint_idx] = 0.2
        elif condition == 'close':
            self.env.data.qpos[joint_idx] = 0.0
        else:
            raise NotImplementedError
        
    def get_q_state(self):
        joint_state = self.env.get_qpos_joint('slidedoor_joint')[0]
        return {'slidedoor_joint': joint_state}
    
    def set_q_state(self, joint_states):
        joint_idx = self.env.get_idxs_fwd(joint_names=["slidedoor_joint"])[0]
        self.env.data.qpos[joint_idx] = joint_states['slidedoor_joint']
    
        

class WoodenSlideCabinet():
    def __init__(self, env):
        self.env = env
        self.obj_name = 'body_wooden_slide_cabinet'
        self.door_state = False
        
    def check_range(self, condition="open"):
        joint = self.env.get_qpos_joint('slidedoor_joint')[0]
        if condition == 'open' and joint > 0.18: return True
        if condition == 'close' and joint < 0.05: return True
        else: return False

    def reset(self, condition='close'):
        self.door_state = False
        joint_idx = self.env.get_idxs_fwd(joint_names=["slidedoor_joint"])[0]
        if condition == 'open':
            self.env.data.qpos[joint_idx] = 0.2
        elif condition == 'close':
            self.env.data.qpos[joint_idx] = 0.0
        else:
            raise NotImplementedError
    def get_q_state(self):
        joint_state = self.env.get_qpos_joint('slidedoor_joint')[0]
        return {'slidedoor_joint': joint_state}
    
    def set_q_state(self, joint_states):
        joint_idx = self.env.get_idxs_fwd(joint_names=["slidedoor_joint"])[0]
        self.env.data.qpos[joint_idx] = joint_states['slidedoor_joint']

class WoodenCabinet():
    def __init__(self, env):
        self.env = env
        self.obj_name = 'body_wooden_cabinet'
    
    def check_range(self, region="top", condition="open"):
        joint = self.env.get_qpos_joint(f'wooden_cabinet_{region}_level')[0]
        if condition == 'open' and joint < -0.15: return True
        if condition == 'close' and joint > -0.05: return True
        else: return False
        
    def reset(self, condition=['top','close']):
        joint_idx = self.env.get_idxs_fwd(joint_names=[f"wooden_cabinet_{condition[0]}_level"])[0]
        if condition[1] == 'open':
            self.env.data.qpos[joint_idx] = -0.15
        elif condition[1] == 'close':
            self.env.data.qpos[joint_idx] = 0.0
        else:
            raise NotImplementedError
    def get_q_state(self, regions=["top", "middle", "bottom"]):
        joint_states = {}
        for region in regions:
            joint_states[f'wooden_cabinet_{region}_level'] = self.env.get_qpos_joint(f'wooden_cabinet_{region}_level')[0]
        return joint_states
    
    def set_q_state(self, joint_states):
        for joint_name, state in joint_states.items():
            joint_idx = self.env.get_idxs_fwd(joint_names=[joint_name])[0]
            self.env.data.qpos[joint_idx] = state
        

class WhiteCabinet():
    def __init__(self, env):
        self.env = env
        self.obj_name = 'body_white_cabinet'
    
    def check_range(self, region="top", condition="open"):
        joint = self.env.get_qpos_joint(f'white_cabinet_{region}_level')[0]
        if condition == 'open' and joint < -0.15: return True
        if condition == 'close' and joint > -0.02: return True
        else: return False
        
    def reset(self, condition=['top','close']):
        joint_idx = self.env.get_idxs_fwd(joint_names=[f"white_cabinet_{condition[0]}_level"])[0]
        if condition[1] == 'open':
            self.env.data.qpos[joint_idx] = -0.15
        elif condition[1] == 'close':
            self.env.data.qpos[joint_idx] = 0.0
        else:
            raise NotImplementedError
    def get_q_state(self, regions=["top", "middle", "bottom"]):
        joint_states = {}
        for region in regions:
            joint_states[f'white_cabinet_{region}_level'] = self.env.get_qpos_joint(f'white_cabinet_{region}_level')[0]
        return joint_states
    
    def set_q_state(self, joint_states):
        for joint_name, state in joint_states.items():
            joint_idx = self.env.get_idxs_fwd(joint_names=[joint_name])[0]
            self.env.data.qpos[joint_idx] = state