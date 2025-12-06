
from src.mujoco_helper.transforms import rpy2r, r2rpy
import numpy as np
from PIL import Image
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

def transform_action(data, omy_env, ik_env, action_type):
    target_q = data['action'].numpy()
    if action_type == 'joint':
        return target_q
    if action_type == 'delta_joint':
        delta_joint = target_q[:-1] - omy_env.env.get_qpos_joints(omy_env.joint_names)
        delta_joint = np.concatenate([delta_joint, np.array([target_q[-1]])])
        return delta_joint
    ik_env.forward(target_q[:-1], joint_names=omy_env.joint_names)
    p, R = ik_env.get_pR_body(omy_env.tcp_link_name)
    next_eef_pose = np.concatenate([p, r2rpy(R)])
    next_eef_pose = np.append(next_eef_pose, target_q[-1])
    if action_type == 'eef_pose':
        return next_eef_pose
    current_p = omy_env.p0
    current_R = omy_env.R0
    dp = p - current_p
    dR = current_R.T @ R   
    drpy = r2rpy(dR)
    delta = np.concatenate([dp, drpy, np.array([target_q[-1]])])
    if action_type == 'delta_eef_pose':
        return delta
    raise NotImplementedError

def parse_object_info(data):
    obj_pose = data['obj_pose'].numpy()
    obj_names = data['obj_names']
    obj_names = obj_names.split(',')
    recp_q_states = data['obj_q_states'].numpy()
    recp_q_names = data['obj_q_names']
    recp_q_names = recp_q_names.split(',')
    return obj_pose, obj_names, recp_q_states, recp_q_names


def create_dataset(ROOT, add_images = True):
    features = {
        "observation.state": {
                    "dtype": "float32",
                    "shape": (7,),
                    "names": ["state"],
        },
        "action": {
            "dtype": "float32",
            "shape": (7,),
            "names": ["action"], # [q target, gripper]
        }
    }
    if add_images:
        features["observation.image"] = {
                "dtype": "image",
                "shape": (256, 256, 3),
                "names": ["height", "width", "channels"],
        }
        features["observation.wrist_image"] = {
                "dtype": "image",
                "shape": (256, 256, 3),
                "names": ["height", "width", "channels"],
        }
    else:
        # object numbers : 4 x 6
        # objects with q states: 3
        # -> total 27 dimensions
        features['observation.environment_state']= {
                    "dtype": "float32",
                    "shape": (27,),
                    "names": ["obj_pose"],
            }
        
    dataset = LeRobotDataset.create(
            repo_id="transformed_data",
            root = ROOT,
            robot_type="omy",
            fps=20,
            features=features,
            image_writer_threads=10,
            image_writer_processes=5
    )
    return dataset


def iterate_episodes(dataset,transformed_dataset, omy_env, ik_env,q_init, start_idx_ori, end_idx_ori, language_instruction, args, img_aug = False):
    omy_env.reset()
    ik_env.reset()
    for _ in range(10):
        omy_env.action_type = 'joint'
        omy_env.step(q_init)
        omy_env.step_env()
    omy_env.action_type = args.action_type
    current_step = start_idx_ori
    if img_aug:
        omy_env.agument_object_random_color()
    while omy_env.env.is_viewer_alive():
        omy_env.step_env()
        if img_aug and omy_env.env.loop_every(HZ = 1):
            omy_env.agument_object_random_color()
        if omy_env.env.loop_every(HZ = 20):
            success = omy_env.check_success()
            if current_step > end_idx_ori - 1:
                break
            data = dataset.hf_dataset[current_step]
            if current_step == start_idx_ori:
                objet_info = parse_object_info(data)
                omy_env.set_object_pose(*objet_info)
            action = transform_action(data, omy_env, ik_env,args.action_type)
            observation = omy_env.step(action, gripper_mode='continuous')
            agent_image, wrist_image = omy_env.grab_image(return_side=False)
        
            # # resize to 256x256
            frame = {
                "observation.state": observation,
                "action": action.astype(np.float32),
            }
            if args.observation_type == 'image':
                agent_image = Image.fromarray(agent_image)
                wrist_image = Image.fromarray(wrist_image)
                agent_image = agent_image.resize((256, 256))
                wrist_image = wrist_image.resize((256, 256))
                agent_image = np.array(agent_image)
                wrist_image = np.array(wrist_image)
                frame["observation.image"] = agent_image
                frame["observation.wrist_image"] = wrist_image
            else:
                obj_states, recp_q_poses = omy_env.get_object_pose(pad=10)
                # extract position and rpy
                # sort the object names in alphabetical order to maintain consistency
                obj_states_sorted = np.zeros((24,), dtype=np.float32)
                sorted_indices = np.argsort(obj_states['names'])
                for i, idx in enumerate(sorted_indices):
                    if 'pad' in obj_states['names'][idx]:
                        continue
                    obj_state = obj_states['poses'][idx]
                    obj_states_sorted[i*6:(i+1)*6] = obj_state
                # sort the receptacle q states in alphabetical order
                recp_q_poses_sorted = np.zeros((3,), dtype=np.float32)
                sorted_indices = np.argsort(recp_q_poses['names'])
                i = 0
                for _, idx in enumerate(sorted_indices):
                    if 'pad' in recp_q_poses['names'][idx]:
                        continue
                    recp_q_pose = recp_q_poses['poses'][idx]
                    recp_q_poses_sorted[i] = recp_q_pose
                    i += 1
                obj_states_final = np.concatenate([obj_states_sorted, recp_q_poses_sorted])
                frame["observation.environment_state"] = obj_states_final
            transformed_dataset.add_frame(
                frame, task=language_instruction
            )
            omy_env.render()
            current_step += 1
    return success


def make_teleoperation_dataset(ROOT):
    dataset = LeRobotDataset.create(
            repo_id="temp",
            root = ROOT,
            robot_type="omy",
            fps=20,
            features={
                "image": {
                    "dtype": "image",
                    "shape": (256, 256, 3),
                    "names": ["height", "width", "channels"],
                },
                "wrist_image": {
                    "dtype": "image",
                    "shape": (256, 256, 3),
                    "names": ["height", "width", "channels"],
                },
                "state": {
                    "dtype": "float32",
                    "shape": (10,),
                    "names": ["state"], # joint angles
                },
                "action": {
                    "dtype": "float32",
                    "shape": (7,),
                    "names": ["action"], # [q target, gripper]
                },
                "eef_pose": {
                    "dtype": "float32",
                    "shape": (7,),
                    "names": ["eef_pose"], # [x,y,z, r, p, y, gripper]
                },
                "obj_pose": {
                    "dtype": "float32",
                    "shape": (10, 6),
                    "names": ["obj_pose"], # just the initial position of the object. Not used in training.
                },
                "obj_names": {
                    "dtype": "string",
                    "shape": (1,),
                    "names": ["obj_names"], # names of the objects
                },
                "obj_q_names": {
                    "dtype": "string",
                    "shape": (1,),
                    "names": ["obj_q_names"], # names of the objects with q states
                },
                "obj_q_states": {
                    "dtype": "float32",
                    "shape": (10,),
                    "names": ["obj_q_states"], # q states of the objects
                },
                "config_file_name": {
                    "dtype": "string",
                    "shape": (1,),
                    "names": ["file_name"], # names of the objects
                }
            },
            image_writer_threads=10,
            image_writer_processes=5
    )
    return dataset