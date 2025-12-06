from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

dataset = LeRobotDataset(
    'Jeongeun/deep_learning_2025_vision_joint', './dataset/transformed_data2'
)
# dataset = LeRobotDataset(
#     'Jeongeun/deep_learning_2025', './dataset/demo_data'
# )
dataset.push_to_hub(
    upload_large_folder=True
)