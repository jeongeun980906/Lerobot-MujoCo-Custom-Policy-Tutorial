
def load_controller(controller_type, env_conf):
    if controller_type == 'joystick':
        from src.controllers.joystick_controller import controller
        controller_instance = controller(env_conf['control_mode'])
    elif controller_type == 'leader':
        from src.controllers.leader_conrtoller import controller
        controller_instance = controller(env_conf['control_mode'])
    elif controller_type == 'keyboard':
        from src.controllers.keyboard_controller import controller
        controller_instance = controller(pose_gain=0.01, rot_gain=0.1)
    else:
        raise NotImplementedError(f"Controller type {controller_type} not implemented.")
    return controller_instance