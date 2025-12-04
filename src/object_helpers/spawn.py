import numpy as np 
from src.mujoco_helper.utils import sample_xyzs, rpy2r, rotation_matrix
from src.mujoco_helper.transforms import r2rpy

class ObjectSpawner:
    def __init__(self, env, init_pose_config):
        """
        env: An environment instance that provides:
            - get_body_names(prefix)
            - set_p_base_body(body_name, p)
            - set_R_base_body(body_name, R)
        """
        self.env = env
        self.init_pose_config = init_pose_config

    def spawn_recp(self):
        # Spawn receptacles
        recepacle_config = self.init_pose_config['receptacles']
        recp_names = []
        recp_sizes = []
        recp_rots = []
        recp_xyzs = []
        # --- Get receptacle names and their configurations ---
        for idx, (name, pose) in enumerate(recepacle_config.items()):
            if 'x_range' in pose:
                x_range = pose['x_range']
            else:
                x_range = [0.3, 0.6]
            if 'y_range' in pose:
                y_range = pose['y_range']
            else:
                y_range = [-0.35, 0.35]
            if 'z' in pose:
                z = pose['z']
            else:
                z = 0.82
            recp_xyz = self._get_non_colliding_position(\
                placed_positions=recp_xyzs,
                x_range=x_range,
                y_range=y_range,
                min_dist=0.1,
                recp_xyzs=recp_xyzs,
                size_recps=recp_sizes,
            )
        
            body_name = f'body_obj_{name}'
            self.env.set_p_body(body_name=body_name, p=[recp_xyz[0], recp_xyz[1], z])
            if pose['init_rot'][0] == 'choice':
                angle = np.random.choice(pose['init_rot'][1])
            elif pose['init_rot'][0] == 'range':
                angle = np.random.uniform(*pose['init_rot'][1])
            else: 
                angle = 0
            self.env.set_R_body(body_name=body_name, 
                                    R=rpy2r(np.deg2rad([0, 0, angle])))
            
            if 'flat_rectangle' or 'bowl' in name: continue
            recp_names.append(body_name)
            recp_rots.append(angle)
            recp_xyzs.append(recp_xyz)
            site_name = f'top_site_{name}'
            ori_size = self.env.model.site(site_name).size
            ori_size = list(ori_size)
            # rotate the size to match the orientation
            recp_sizes.append(self.get_rotated_aabb_size(ori_size[:2], np.deg2rad(angle))+[ori_size[2]])
        self.env.forward(increase_tick=False)
        self.recp_names = recp_names
        self.recp_rots = recp_rots
        self.recp_xyzs = recp_xyzs
        self.recp_sizes = recp_sizes

    def spawn_objects(self):
        # --- Get object names to spawn (exclude the tray) ---
        obj_names = self.env.get_body_names(prefix='body_obj_')
        for recp_name in self.recp_names:
            if recp_name in obj_names:
                obj_names.remove(recp_name)


        # List to keep track of already placed objects to avoid collisions.
        placed_positions = []
        obj_configs = self.init_pose_config['objects']
        obj_configs_names = list(obj_configs.keys())
        # Spawn each object with a non-colliding position and random rotation.
        for name in obj_names:
            ori_name = name.split('body_obj_')[-1]
            ignore_collision = False
            if ori_name in obj_configs_names:
                values = obj_configs[ori_name]
                if 'constraints' in values:
                    constraints = values['constraints']
                    if len(constraints) == 2:
                        relation, obj = constraints
                        relative_pose = []
                    else:
                        relation, obj, relative_pose = constraints
                    if relation == "on":
                        if "site" in obj or 'region' in obj:
                            recp_xyz = self.env.get_p_site(site_name=obj)
                            recp_R = self.env.get_R_site(site_name=obj)
                            recp_size = self.env.model.site(obj).size
                        else:
                            recp_xyz = self.env.get_p_body(body_name=obj)
                            recp_R = self.env.get_R_body(body_name=obj)
                            recp_size = [0.12, 0.12, 0.01]
                        recp_R = r2rpy(recp_R)
                        recp_size = self.get_rotated_aabb_size(recp_size[:2], np.deg2rad(recp_R[2])) + [recp_size[2]]
                        if len(relative_pose) == 2:
                            # rotate the ranges with rotation
                            rot = recp_R[2]
                            relative_pose = np.array(relative_pose)
                            relative_pose = np.dot(np.array([[np.cos(rot), -np.sin(rot)], [np.sin(rot), np.cos(rot)]]), relative_pose)
                            x_range = [recp_xyz[0] + relative_pose[0][0], recp_xyz[0] + relative_pose[0][1]]
                            y_range = [recp_xyz[1] + relative_pose[1][0], recp_xyz[1] + relative_pose[1][1]]
                        else:
                            x_range = [recp_xyz[0] - recp_size[0]+ 0.05, recp_xyz[0] + recp_size[0] - 0.05]
                            y_range = [recp_xyz[1] - recp_size[1]+ 0.05, recp_xyz[1]  + recp_size[1] - 0.05]
                        if "z" in values:
                            z = values['z']
                        else:
                            z = recp_xyz[2] + 0.02
                        ignore_collision = True
                else:
                    if 'x_range' in values:
                        x_range = values['x_range']
                    else: 
                        x_range = [0.3, 0.6]
                    if 'y_range' in values:
                        y_range = values['y_range']
                    else:
                        y_range = [-0.35, 0.35]
                    if 'z' in values:
                        z = values['z']
                    else:
                        z = 0.82
                if 'yaw_range' in values:
                    yaw_range = values['yaw_range']
                    angle = np.random.uniform(*yaw_range)
                else: 
                    angle = 0
            else:
                continue
            if ignore_collision:
                pos = sample_xyzs(
                    n_sample=1,
                    x_range=x_range,
                    y_range=y_range,
                    z_range=[z, z],
                    min_dist=0.15,
                    xy_margin=0.00
                )[0][:2]
                # print(pos, self.recp_xyzs[:2], self.recp_sizes)
            else:
                # print(name, self.recp_xyzs, self.recp_sizes, x_range, y_range, placed_positions)
                # Find a position that doesn't overlap with previously placed objects.
                pos = self._get_non_colliding_position(
                    placed_positions=placed_positions,
                    x_range=x_range,
                    y_range=y_range,
                    min_dist=0.12,
                    recp_xyzs=self.recp_xyzs,
                    size_recps=self.recp_sizes
                )
            placed_positions.append(pos)
            # Set the object's position (using the same z as the tray for simplicity).
            self.env.set_p_base_body(body_name=name, p=[pos[0], pos[1], z])
            if 'roll' in values:
                roll = values['roll']
                pitch = values['pitch']
            else: 
                roll = 0
                pitch = 0
            self.env.set_R_base_body(body_name=name, R=rpy2r(np.deg2rad([roll, pitch, angle])))
            if 'color' in values:
                print(ori_name)
                geom_name = f"{ori_name}_1_geom"
                geom_idx = self.env.geom_names.index(geom_name)
                self.env.model.geom(geom_idx).rgba  = np.array(values['color'])
                print("CHANGE COLOR!")
            

    def _get_non_colliding_position(self, placed_positions, x_range, y_range, min_dist, recp_xyzs, size_recps, margin=0.1):
        """Attempts to sample a position that does not collide with already placed objects and recp_xyzs.
           Raises a ValueError if no valid position is found after a fixed number of attempts."""
        max_attempts = 10000
        for attempt in range(max_attempts):
            x = np.random.uniform(x_range[0], x_range[1])
            y = np.random.uniform(y_range[0], y_range[1])
            candidate = np.array([x, y])
            
            collision = False
            for placed_pos in placed_positions:
                if np.linalg.norm(candidate - placed_pos) < min_dist:
                    # print(candidate)
                    collision = True
                    break
              # Check against each rectangle.
            if not collision:
                for recp, size in zip(recp_xyzs, size_recps):
                    half_width = size[0] 
                    half_height = size[1] 
                    # print(size, recp)
                    # Check if candidate is within the rectangle bounds (assumed to be centered at recp).
                    if (recp[0] - half_width - margin <= candidate[0] <= recp[0] + half_width + margin and
                        recp[1] - half_height - margin <= candidate[1] <= recp[1] + half_height + margin):
                        collision = True
                        break
            
            # If no collision was detected, return the candidate.
            if not collision:
                return candidate

        # If no valid candidate is found after max_attempts, raise an error.
        raise ValueError(f"No valid non-colliding position found after {max_attempts} attempts.")
    def get_rotated_aabb_size(self, size, yaw):
        """
        Computes the axis-aligned bounding box dimensions for a rotated rectangle.

        Parameters:
        size: A tuple or list (width, height) representing the rectangle's dimensions.
        yaw: The rotation angle in radians (rotation about the z-axis).

        Returns:
        A tuple (new_width, new_height) representing the AABB dimensions.
        """
        w, h = size
        new_width = abs(w * np.cos(yaw)) + abs(h * np.sin(yaw))
        new_height = abs(w * np.sin(yaw)) + abs(h * np.cos(yaw))
        return [new_width, new_height]
