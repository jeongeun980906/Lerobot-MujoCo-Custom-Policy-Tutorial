import sys
import random
import numpy as np
import os
from PIL import Image
from src.env.env import RILAB_OMY_ENV
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from src.controllers import load_controller
import gc, argparse
import json
import copy
import time
from src.dataset.utils import make_teleoperation_dataset

def main(args):
    episode_id = 0
    NUM_trials_PER_TASK = args.num_trials
    ROOT = args.root_dir
    SEED = None
    config_file_path = './configs/train.json'
    with open(config_file_path) as f:
        env_conf = json.load(f)
    leader = load_controller('leader',env_conf)
    language_instruction = env_conf['language_instruction']
    omy_env = RILAB_OMY_ENV(cfg=env_conf, seed=SEED, 
                            action_type=env_conf['control_mode'], obs_type='eef_pose',
                            vis_mode = 'teleop')
    make_new = True
    if os.path.exists(ROOT):
        if args.resume:
            dataset = LeRobotDataset('temp', root=ROOT)
            # episode_id = dataset.num_episodes - 60
            episode_id = 0
            print(dataset.num_episodes)
            make_new = False
        else:
            import shutil
            print("REMOVE")
            shutil.rmtree(ROOT)
            make_new = True
    if make_new:
        print("CREATE")
        dataset = make_teleoperation_dataset(ROOT)
    action = np.zeros(12)
    record_flag = False
    just_reseted = True
    last_q = omy_env.get_full_joint_state() #np.zeros(12)
    # zero_pose = np.array([-0.6442722678184509, -2.0621182918548584, 2.297856569290161, -0.26436978578567505, 0.9549943804740906, 0.037058718502521515])
    last_obj_poses = np.zeros((10, 6))
    while leader.action_data is None:
        time.sleep(0.1)
    omy_env.reset()
    for _ in range(10):
        action = leader.get_action()
        omy_env.step(action)
        omy_env.step_env()
    leader.reset(omy_env)
    while omy_env.env.is_viewer_alive() and episode_id < NUM_trials_PER_TASK:
        omy_env.step_env()
        if omy_env.env.loop_every(HZ=20):
            key_list = omy_env.env.get_key_pressed_list()
            done = omy_env.check_success()
            if done or 90 in key_list:  # 'z' key to reset
                print("END EPISODE")
                if done:
                    dataset.save_episode()
                    episode_id += 1
                else: 
                    dataset.clear_episode_buffer()
                omy_env.reset()

                for _ in range(50):
                    action = leader.get_action()
                    omy_env.step(action)
                    omy_env.step_env()

                leader.reset(env=omy_env)
                record_flag = False
                just_reseted = True
                action = leader.get_action()
                joint_q = omy_env.step(action)
            action = leader.get_action() #  , reset, fail
            eef_pose = omy_env.step(action)
            joint_q_full = omy_env.get_full_joint_state()           
            
            agent_image,wrist_image = omy_env.grab_image(return_side=False)
            # # resize to 256x256
            agent_image = Image.fromarray(agent_image)
            wrist_image = Image.fromarray(wrist_image)
            agent_image = agent_image.resize((256, 256))
            wrist_image = wrist_image.resize((256, 256))
            agent_image = np.array(agent_image)
            wrist_image = np.array(wrist_image)
            obj_states, recp_q_poses = omy_env.get_object_pose(pad=10)
            obj_poses = np.array(obj_states['poses'])
            joint_q = joint_q_full[:6]
            if just_reseted and sum(abs(joint_q - omy_env.q_zero)) > 1e-1:
                print("Reset the leader to idle pose")
                omy_env.render("reset the leader to idle pose")
                continue
            else: just_reseted = False
            if not record_flag and sum(abs(joint_q - omy_env.q_zero)) > 1e-1 and not just_reseted:
                record_flag = True
                print("Start recording")
                
                # print(obj_init_poses, obj_names)
            if record_flag:
                if sum(abs(joint_q_full - last_q)) < 1e-3 and np.sum(abs(obj_poses - last_obj_poses)) < 1e-3:
                    # print("Stop recording", omy_env.env.tick)
                    print("Stop recording", omy_env.env.tick)
                #     # continue
                else:
                    dataset.add_frame( {
                            "image": agent_image,
                            "wrist_image": wrist_image,
                            "state": joint_q_full,
                            "action": action,
                            "eef_pose": eef_pose,
                            'obj_pose': np.array(obj_states['poses'],dtype=np.float32),
                            "obj_names": ','.join(obj_states['names']),
                            "obj_q_names": ','.join(recp_q_poses['names']),
                            "obj_q_states": np.array(recp_q_poses['poses'],dtype=np.float32),
                            "config_file_name": config_file_path,
                        }, task=language_instruction
                    )
            last_q = joint_q_full    
            last_obj_poses = obj_poses
            # based on the episode_id number, get the guide line
            omy_env.render(language_instruction, guideline= f' [Num Episode: {episode_id}/{args.num_trials}]')
        omy_env.env.sync_sim_wall_time()
    omy_env.env.close_viewer()
    dataset.stop_image_writer()
    # leader.close()
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_trials", type=int, default=50)
    parser.add_argument("--root_dir", type=str, default='./dataset/demo_data')
    parser.add_argument('--resume', action='store_true', default=False)
    args = parser.parse_args()
    os.makedirs(args.root_dir, exist_ok=True)
    main(args)
