import argparse
import sys
import random
import numpy as np
import os
from PIL import Image
import json
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from src.env.env import RILAB_OMY_ENV
from src.mujoco_helper.transforms import rpy2r, r2rpy
import torch
from src.dataset.utils import *
from src.mujoco_helper import MuJoCoParserClass
import gc
def main(args):
    dataset = LeRobotDataset('Jeongeun/deep_learning_2025',root = './dataset/demo_data')
    metadata = dataset.meta
    if os.path.exists(args.path):
        import shutil
        print("REMOVE")
        shutil.rmtree(args.path)
    transformed_dataset = create_dataset(args.path, add_images = (args.observation_type=='image'))
    config_file_path = './configs/train2.json'
    with open(config_file_path) as f:
        env_conf = json.load(f)
    omy_env = RILAB_OMY_ENV(cfg=env_conf, seed=0, 
                            action_type=args.action_type, 
                            obs_type=args.proprio_type,
                            vis_mode = 'teleop')
    ik_env = MuJoCoParserClass(name='IK_env',rel_xml_path='./asset/scene_table.xml')
    for episode_index in range(metadata.total_episodes):
        
        start_idx_ori = dataset.episode_data_index['from'][episode_index].item()
        end_idx_ori = dataset.episode_data_index['to'][episode_index].item()
        q_init = dataset.hf_dataset[start_idx_ori]['action'].numpy()
        language_instruction = dataset.hf_dataset[start_idx_ori]['task_index'].item()
        language_instruction = metadata.tasks[language_instruction]
        print(f"Episode {episode_index}, Instruction: {language_instruction}")
        # back to original colors

        success =   iterate_episodes(dataset, transformed_dataset, omy_env, ik_env, q_init, start_idx_ori, end_idx_ori, language_instruction, args)
        if success:
            transformed_dataset.save_episode()
        else:
            transformed_dataset.clear_episode_buffer()
        if args.observation_type == 'image':
            for _ in range(args.image_aug_num):
                # This will randomize object colors
                success =   iterate_episodes(dataset, transformed_dataset, omy_env, ik_env, q_init, start_idx_ori, end_idx_ori, language_instruction, args, img_aug=True)
                if success:
                    transformed_dataset.save_episode()
                else:
                    transformed_dataset.clear_episode_buffer()
        gc.collect()
        torch.cuda.empty_cache()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--action_type', type=str, default='delta_joint',choices=['joint','delta_joint','eef_pose','delta_eef_pose'],)
    parser.add_argument('--proprio_type', type=str, default='eef_pose', choices=['joint_pos','eef_pose'],)
    parser.add_argument('--observation_type', type=str, default='image',choices=['image','objet_pose'],)
    parser.add_argument('--path', type=str, default='./dataset/transformed_data',)
    parser.add_argument('--image_aug_num', type=int, default=5,)
    args = parser.parse_args()
    main(args)