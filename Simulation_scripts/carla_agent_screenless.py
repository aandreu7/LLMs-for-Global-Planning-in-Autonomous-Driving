#!/usr/bin/python3

# Copyright (c) 2019 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.


from __future__ import print_function

import argparse
import logging
import numpy as np
import carla
import cv2
import os
import math
import json
import re
import fcntl
import tempfile
import uuid

from mc_utils_screenless import get_ego_vehicle, close_world
from mc_min_screenless import init_game, iter_game

# !!!!!! MODS !!!!!!
import sys
carla_agents_path = '/datafast/105-1/Datasets/INTERNS/aplanaj/CARLA_0.9.15/PythonAPI/carla'
if carla_agents_path not in sys.path:
    sys.path.append(carla_agents_path)
# !!!!!! MODS !!!!!!

from agents.navigation.global_route_planner import GlobalRoutePlanner # MOD

BEV_CAMERA_PARAMS_PER_MAP = {
    "Town01_Opt": (200.0, 200.0, 250.0),
    "Town02_Opt": (200, 200, 250),
    "Town03_Opt": (200, 200, 250),
    "Town04_Opt": (-50, 0, 500),
    "Town05_Opt": (0, 0, 300),
    "Town07": (-50, -30, 190),
    "Town15": (80, 0, 1000),
}

# Auxiliar functions
def estimate_distance(pos1, pos2):
    distance = np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
    return distance


def build_projection_matrix(w, h, fov, is_behind_camera=False):
    focal = w / (2.0 * np.tan(fov * np.pi / 360.0))
    K = np.identity(3)

    if is_behind_camera:
        K[0, 0] = K[1, 1] = -focal
    else:
        K[0, 0] = K[1, 1] = focal

    K[0, 2] = w / 2.0
    K[1, 2] = h / 2.0
    return K


def carla_world3D_to_cam3D(M, world_pt):
    # inverse matrix
    inv_transf_matrix = np.array(M.get_inverse_matrix())

    # world3D to cam3D
    cam_pt = np.dot(inv_transf_matrix, world_pt)

    # normalize
    cam_pt[0] /= cam_pt[3]
    cam_pt[1] /= cam_pt[3]
    cam_pt[2] /= cam_pt[3]

    return cam_pt


def carla_cam3D_to_cam2D(K, cam_pt):
    # UE4 system reference to standard reference
    cam_pt = [cam_pt[1], -cam_pt[2], cam_pt[0]]

    # cam3D to cam2D
    img_pt = np.dot(K, cam_pt)

    # normalize
    img_pt[0] /= img_pt[2]
    img_pt[1] /= img_pt[2]

    return img_pt


def rotate_points(points, yaw_deg, center=(0.0, 0.0)):
    """
    Rotate 2D points by a given yaw angle around a center.

    Args:
        points (array-like): list or array of shape (N, 2), e.g. [[x1, y1], [x2, y2], [x3, y3]]
        yaw_deg (float): rotation angle in degrees (positive = counter-clockwise)
        center (tuple): (x, y) rotation center (default is origin)

    Returns:
        np.ndarray: rotated points of shape (N, 2)
    """
    points = np.array(points)
    cx, cy = center
    yaw = np.radians(yaw_deg)

    # Rotation matrix (2D)
    R = np.array([
        [np.cos(yaw), -np.sin(yaw)],
        [np.sin(yaw),  np.cos(yaw)]
    ])

    # Translate to origin, rotate, then translate back
    translated = points - np.array([cx, cy])
    rotated = translated @ R.T
    rotated += np.array([cx, cy])
    
    return rotated


# File locking helpers to coordinate multiple processes writing the same files
def file_lock(lock_path):
    """Context manager that acquires an exclusive lock on lock_path.

    Usage:
        with file_lock('/some/dir/.lock'):
            # critical section
    """
    class _Lock:
        def __init__(self, path):
            self.path = path
            # ensure directory exists
            d = os.path.dirname(path)
            if d and not os.path.exists(d):
                try:
                    os.makedirs(d, exist_ok=True)
                except Exception:
                    pass
            self._fh = None

        def __enter__(self):
            # open lock file
            self._fh = open(self.path, 'w')
            # block until lock
            fcntl.flock(self._fh, fcntl.LOCK_EX)
            return self._fh

        def __exit__(self, exc_type, exc, tb):
            try:
                if self._fh:
                    fcntl.flock(self._fh, fcntl.LOCK_UN)
                    self._fh.close()
            except Exception:
                pass

    return _Lock(lock_path)


def atomic_write_json(path, data):
    """Atomically write JSON data to path by writing to a tmp file and renaming.
    """
    d = os.path.dirname(path) or '.'
    with tempfile.NamedTemporaryFile('w', delete=False, dir=d, prefix='.tmp_json_', suffix='.json') as tf:
        json.dump(data, tf, indent=4)
        tmpname = tf.name
    os.replace(tmpname, path)


def draw_pt(img, pt, angle, color=(0, 0, 255), shape='triangle'):
    if shape == 'triangle':
        width_triangle = 10
        height_triangle = 10

        vertices = np.array([[pt[0]-width_triangle/2., pt[1]-height_triangle/2.],
                            [pt[0]+width_triangle/2., pt[1]-height_triangle/2.],
                            [pt[0], pt[1]+height_triangle/2.]])

        vertices = np.array(rotate_points(vertices, angle, center=(pt[0], pt[1])), np.int32)

        vertices_pts = vertices.reshape((-1, 1, 2))
        cv2.polylines(img, [vertices_pts], isClosed=True, color=color, thickness=1)

        # fill it
        cv2.fillPoly(img, [vertices_pts], color=color)

    elif shape=='circle':
        pass

    elif shape=='square':
        width_square = 15
        height_square = 15

        top_left = (int(pt[0] - width_square/2.), int(pt[1] + height_square/2.))
        bottom_right = (int(pt[0] + width_square/2.), int(pt[1] - height_square/2.))
        img = cv2.rectangle(img, top_left, bottom_right, color, thickness=-1)

        # Add border, maximizing contrast
        img = cv2.rectangle(img, top_left, bottom_right, (0, 0, 0), thickness=2)

    # MOD ==============
    elif shape=='arrow':
        arrow_length = 24
        arrow_width = 7
        head_length = 12
        head_width = 18
        
        vertices = np.array([
            [pt[0] - arrow_width/2., pt[1] - arrow_length/2.],
            [pt[0] - arrow_width/2., pt[1] + arrow_length/2. - head_length],
            [pt[0] - head_width/2., pt[1] + arrow_length/2. - head_length],
            [pt[0], pt[1] + arrow_length/2.],
            [pt[0] + head_width/2., pt[1] + arrow_length/2. - head_length],
            [pt[0] + arrow_width/2., pt[1] + arrow_length/2. - head_length],
            [pt[0] + arrow_width/2., pt[1] - arrow_length/2.]
        ])
        
        vertices = np.array(rotate_points(vertices, angle, center=(pt[0], pt[1])), np.int32)
        vertices_pts = vertices.reshape((-1, 1, 2))
        
        # Borde negro para contraste (key color low value for filtering later)
        cv2.polylines(img, [vertices_pts], isClosed=True, color=(0, 0, 0), thickness=3)
        # Relleno de color
        cv2.fillPoly(img, [vertices_pts], color=color)
    # MOD ==============

    return img



# DRIVING agent
class AIDriverAgent():
    def __init__(self, args, sim_world, client):
        self.direction = 3
        self.control = carla.VehicleControl()
        self.v_sensors = []
        self.sim_world = sim_world
        self.ego = get_ego_vehicle()
        self.args = args
        self.saved_imgs = {}
        self.counter = 0
        self.last_indication = ""
        self.tm = client.get_trafficmanager(args.tm_port)
        self.ind_counter = 0
        self.intersections_data = [] # MOD
        self.is_in_junction = False # MOD
        self.existing_intersection_indices = set() # Para evitar duplicados
        self._load_existing_intersections()

        self.bev_camera_params = {}
        self.bev_camera_params['R_angle'] = (-90.0, -90.0, 0)
        self.bev_camera_params['t'] = BEV_CAMERA_PARAMS_PER_MAP.get(args.map, (200.0, 200.0, 250.0))

        self.bev_image_shape = (300, 300)

        self.v_sensor_params = [
                    {'type': 'sensor.camera.rgb', 'x': 0.0, 'y': 0.0, 'z': 2.0, 'roll': 0.0, 'pitch': 0.0, 'yaw': -60.0,
                    'width': 300, 'height': 300, 'fov': 60, 'id': 'img1', 'lens_circle_setting': False, 'attach_vehicle': True}, # right
                    {'type': 'sensor.camera.rgb', 'x': 0.0, 'y': 0.0, 'z': 2.0, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                    'width': 300, 'height': 300, 'fov': 60, 'id': 'img2', 'lens_circle_setting': False, 'attach_vehicle': True}, # front
                    {'type': 'sensor.camera.rgb', 'x': 0.0, 'y': 0.0, 'z': 2.0, 'roll': 0.0, 'pitch': 0.0, 'yaw': 60.0,
                    'width': 300, 'height': 300, 'fov': 60, 'id': 'img3', 'lens_circle_setting': False, 'attach_vehicle': True},  # left

                    {'type': 'sensor.camera.semantic_segmentation', 'x': 0.0, 'y': 0.0, 'z': 2.0, 'roll': 0.0, 'pitch': 0.0, 'yaw': -60.0,
                    'width': 300, 'height': 300, 'fov': 60, 'id': 'ss_img1', 'lens_circle_setting': False, 'attach_vehicle': True}, # ss right
                    {'type': 'sensor.camera.semantic_segmentation', 'x': 0.0, 'y': 0.0, 'z': 2.0, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                    'width': 300, 'height': 300, 'fov': 60, 'id': 'ss_img2', 'lens_circle_setting': False, 'attach_vehicle': True}, # ss front
                    {'type': 'sensor.camera.semantic_segmentation', 'x': 0.0, 'y': 0.0, 'z': 2.0, 'roll': 0.0, 'pitch': 0.0, 'yaw': 60.0,
                    'width': 300, 'height': 300, 'fov': 60, 'id': 'ss_img3', 'lens_circle_setting': False, 'attach_vehicle': True},  # ss left

                    {'type': 'sensor.camera.rgb', 'x': self.bev_camera_params['t'][0], 'y': self.bev_camera_params['t'][1], 'z': self.bev_camera_params['t'][2],
                    'roll': self.bev_camera_params['R_angle'][0], 'pitch': self.bev_camera_params['R_angle'][1], 'yaw': self.bev_camera_params['R_angle'][2],
                    'width': self.bev_image_shape[0], 'height': self.bev_image_shape[1], 'fov': 90, 'id': 'bev', 'lens_circle_setting': False, 'attach_vehicle': False},    # BEV

                    {'type': 'sensor.camera.semantic_segmentation', 'x': self.bev_camera_params['t'][0], 'y': self.bev_camera_params['t'][1], 'z': self.bev_camera_params['t'][2],
                    'roll': self.bev_camera_params['R_angle'][0], 'pitch': self.bev_camera_params['R_angle'][1], 'yaw': self.bev_camera_params['R_angle'][2],
                    'width': self.bev_image_shape[0], 'height': self.bev_image_shape[1], 'fov': 90, 'id': 'ss_bev', 'lens_circle_setting': False, 'attach_vehicle': False},    # ss BEV
                ]

        if args.save_data:
            self.create_dirs()
        self.init_sensors()

        ending_pos_world4 = [args.ego_ending_position[0], args.ego_ending_position[1], args.ego_ending_position[2], 1.0]
        np_ending_pos_cam = carla_world3D_to_cam3D(self.bev_camera_matrix, ending_pos_world4)
        self.ending_pix_pt = carla_cam3D_to_cam2D(self.bev_K, np_ending_pos_cam)

        self.special_img_counter = self.get_next_special_image_index()



# MODS !!!!!!!!!!!!!!!!!!!!!!
    def get_next_special_image_index(self):
        """Looks in the special_maps directory to find the highest index and returns the next one."""
        dir_path = self.paths.get('special_maps')
        # If dir missing, keep previous behavior
        if not dir_path or not os.path.exists(dir_path):
            return 0

        # Use a lock to avoid races when multiple processes try to allocate the next index
        lock_dir = os.path.dirname(dir_path) or '.'
        lockfile = os.path.join(lock_dir, '.intersections.lock')
        max_index = -1
        with file_lock(lockfile):
            for filename in os.listdir(dir_path):
                if filename.endswith(('.jpg', '.png')):
                    try:
                        # Extracts the numeric part of the filename, e.g., '000000123' from '000000123_rgb.jpg'
                        index_str = filename.split('_')[0]
                        current_index = int(index_str)
                        if current_index > max_index:
                            max_index = current_index
                    except (ValueError, IndexError):
                        # Ignore files that don't match the expected format
                        continue

        return max_index + 1
# MODS !!!!!!!!!!!!!!!!!!!!!!



    def create_img_msg(self, image, name):
        img_array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        img_array = np.reshape(img_array, (300, 300, 4))[:, :, :3]
        img_array = img_array / 255.

        if name == 'bev':
            img_array = draw_pt(img_array, self.ego_pix_pt, self.ego_yaw - 90, color=(0, 0, 255), shape='arrow')  # MOD
            img_array = draw_pt(img_array, self.ending_pix_pt, 0, color=(255, 0, 0), shape='square')

        return img_array

    def create_ss_img_msg(self, image, name):
        image.convert(carla.ColorConverter.CityScapesPalette)
        img_array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        img_array = np.reshape(img_array, (300, 300, 4))[:, :, :3]
        img_array = img_array / 255.

        if name == 'ss_bev':
            #img_array = draw_pt(img_array, self.ego_pix_pt, self.ego_yaw - 90, color=(0, 0, 255), shape='arrow')  # MOD
            #img_array = draw_pt(img_array, self.ending_pix_pt, 0, color=(255, 0, 0), shape='square')

            # Reconverts arrow & square to 0-1 scale if needed
            img_array = np.where(img_array > 1.0, img_array / 255., img_array)

            # ======== MODS =========
            # Filter allowed colors

            allowed_colors = np.array([
                [232, 35, 244], # Pink: background
                [255, 0, 0],    # Blue: square (note: color ordering preserved as in source)
                [0, 0, 255],    # Red: arrow
                [128, 64, 128], # Gray: road
                [0, 0, 0]       # Black (key color): figure borders
            ], dtype=np.float64) / 255.

            # Máscara: True si el color del píxel está en allowed_colors
            mask = np.zeros((300, 300), dtype=bool)
            for color in allowed_colors:
                mask |= np.all(img_array == color, axis=-1)

            # Reemplaza los que NO están en la lista con fondo rosa
            img_array[~mask] = (np.array([232, 35, 244], dtype=np.float64) / 255.)

        return img_array

    def save_ss_image(self, image, name):
        img = self.create_ss_img_msg(image, name)

        if self.args.show_sensors:
            self.saved_imgs[name] = img

        if self.args.save_data:
            cv2.imwrite(os.path.join(self.paths[name], 'clean_bev.png'), img * 255)
            #cv2.imwrite(os.path.join(self.paths[name], '%09d.png' % self.counter), img * 255)

    def save_image(self, image, name):
        img = self.create_img_msg(image, name)
        if self.args.show_sensors:
            self.saved_imgs[name] = img

        if self.args.save_data:
            cv2.imwrite(os.path.join(self.paths[name], '%09d.jpg' % self.counter), img * 255)

    def update_direction(self, key):
        if key == 'w':
            self.direction = 3  # go straight
        elif key == 'a':
            self.direction = 1  # left
        elif key == 'd':
            self.direction = 2  # right
        elif key == '1':
            self.direction = 5  # left lane
        elif key == '3':
            self.direction = 6  # right lane
        elif key == '2':
            self.direction = 4  # lane following

    def init_sensors(self):
        v_sensor_params = self.v_sensor_params

        # We create the camera through a blueprint that defines its properties
        for sensor_params in v_sensor_params:
            if sensor_params['type'] == 'sensor.camera.rgb':
                camera_init_trans = carla.Transform(carla.Location(x=sensor_params['x'], y=sensor_params['y'], z=sensor_params['z']), carla.Rotation(roll=sensor_params['roll'], pitch=sensor_params['pitch'], yaw=sensor_params['yaw']))
                camera_bp = self.sim_world.get_blueprint_library().find(sensor_params['type'])
                camera_bp.set_attribute('image_size_x', str(sensor_params['width']))
                camera_bp.set_attribute('image_size_y', str(sensor_params['height']))
                camera_bp.set_attribute('fov', str(sensor_params['fov']))
                if sensor_params['attach_vehicle']:
                    camera_sensor = self.sim_world.spawn_actor(camera_bp, camera_init_trans, attach_to=self.ego)
                else:
                    camera_sensor = self.sim_world.spawn_actor(camera_bp, camera_init_trans)
                camera_sensor.listen(lambda image, name=sensor_params['id']: self.save_image(image, name))
                self.v_sensors.append(camera_sensor)

            if sensor_params['type'] == 'sensor.camera.semantic_segmentation':
                camera_init_trans = carla.Transform(carla.Location(x=sensor_params['x'], y=sensor_params['y'], z=sensor_params['z']), carla.Rotation(roll=sensor_params['roll'], pitch=sensor_params['pitch'], yaw=sensor_params['yaw']))
                camera_bp = self.sim_world.get_blueprint_library().find(sensor_params['type'])
                camera_bp.set_attribute('image_size_x', str(sensor_params['width']))
                camera_bp.set_attribute('image_size_y', str(sensor_params['height']))
                camera_bp.set_attribute('fov', str(sensor_params['fov']))
                if sensor_params['attach_vehicle']:
                    camera_sensor = self.sim_world.spawn_actor(camera_bp, camera_init_trans, attach_to=self.ego)
                else:
                    camera_sensor = self.sim_world.spawn_actor(camera_bp, camera_init_trans)
                camera_sensor.listen(lambda image, name=sensor_params['id']: self.save_ss_image(image, name))
                self.v_sensors.append(camera_sensor)

            if sensor_params['id'] == 'bev':
                self.bev_camera_matrix = camera_init_trans
                self.bev_K = build_projection_matrix(sensor_params['width'], sensor_params['height'], sensor_params['fov'])

    def visualize_images(self):
        #print('WARNING: visualize images makes no sense on screenles mode')
        pass

    def update_ego_position(self):
        c_ego_pos_world = self.ego.get_transform().location
        self.ego_pos = (c_ego_pos_world.x, c_ego_pos_world.y, c_ego_pos_world.z)
        ego_rot = self.ego.get_transform().rotation
        self.ego_yaw = ego_rot.yaw
        self.ego_orientation = (ego_rot.roll, ego_rot.pitch, ego_rot.yaw)


        np_ego_pos_world4 = np.asarray([c_ego_pos_world.x, c_ego_pos_world.y, c_ego_pos_world.z, 1])

        np_ego_pos_cam = carla_world3D_to_cam3D(self.bev_camera_matrix, np_ego_pos_world4)
        self.ego_pix_pt = carla_cam3D_to_cam2D(self.bev_K, np_ego_pos_cam)

    def _load_existing_intersections(self):
        """Load existing intersection indices to avoid duplicates"""
        intersections_file = os.path.join('./special_info', 'intersections.json')
        if os.path.exists(intersections_file) and os.path.getsize(intersections_file) > 0:
            try:
                with open(intersections_file, 'r') as f:
                    existing = json.load(f)
                    # Guardar los índices de imágenes que ya existen
                    for entry in existing:
                        if 'front_imgs_rgb' in entry:
                            val = entry['front_imgs_rgb']
                            # extraer sólo el nombre de fichero y buscar el prefijo numérico
                            base = os.path.basename(val)
                            m = re.match(r"(\d+)", base)
                            if m:
                                idx = int(m.group(1))
                                self.existing_intersection_indices.add(idx)
                            else:
                                print(f"Warning: could not parse index from existing entry front_imgs_rgb='{val}'")
            except Exception as e:
                print(f"Warning: Error loading existing intersections: {e}")
                self.existing_intersection_indices = set()

    def create_dirs(self):
        self.paths = {}

        self.paths['info'] = os.path.join(self.args.data_path, 'info')
        if not os.path.exists(self.paths['info']):
            os.makedirs(self.paths['info'])

        # MODS: Create directories for special intersection data
        self.paths['special_info'] = os.path.join('.', 'special_info')
        if not os.path.exists(self.paths['special_info']):
            os.makedirs(self.paths['special_info'])

        self.paths['special_fronts'] = os.path.join('.', 'special_info', 'special_fronts')
        if not os.path.exists(self.paths['special_fronts']):
            os.makedirs(self.paths['special_fronts'])

        self.paths['special_maps'] = os.path.join('.', 'special_info', 'special_maps')
        if not os.path.exists(self.paths['special_maps']):
            os.makedirs(self.paths['special_maps'])

        self.paths['clean_bev_ss_images'] = os.path.join('.', 'clean_bev_ss_images')
        if not os.path.exists(self.paths['clean_bev_ss_images']):
            os.makedirs(self.paths['clean_bev_ss_images'])

        for sensor in self.v_sensor_params:
            self.paths[sensor['id']] = os.path.join(self.args.data_path, 'sensors', sensor['id'])
            if not os.path.exists(self.paths[sensor['id']]):
                os.makedirs(self.paths[sensor['id']])

    # MODS: store intersections and special images
    def write_intersections_json(self):
        # Determine output path and lock
        file_out = os.path.join(self.paths.get('special_info', '.'), 'intersections.json')
        lockfile = os.path.join(self.paths.get('special_info', '.') or '.', '.intersections.lock')

        # Only proceed if there's actually new intersection data to save.
        if not self.intersections_data:
            return

        with file_lock(lockfile):
            # Read existing intersection data if the file exists.
            all_intersections = []
            if os.path.exists(file_out) and os.path.getsize(file_out) > 0:
                try:
                    with open(file_out, 'r') as f:
                        all_intersections = json.load(f)
                        if not isinstance(all_intersections, list):
                            print(f"Warning: {file_out} does not contain a list. It will be overwritten.")
                            all_intersections = []
                except json.JSONDecodeError:
                    print(f"Warning: Could not decode JSON from {file_out}. The file will be overwritten.")
                    all_intersections = []
                except Exception as e:
                    print(f"An error occurred while reading {file_out}: {e}. The file will be overwritten.")
                    all_intersections = []

            # Solo añadir las intersecciones que no existan ya
            for new_intersection in self.intersections_data:
                if 'front_imgs_rgb' in new_intersection:
                    val = new_intersection['front_imgs_rgb']
                    base = os.path.basename(val)
                    m = re.match(r"(\d+)", base)
                    if not m:
                        print(f"Warning: Skipping malformed front_imgs_rgb='{val}' (no leading digits)")
                        continue
                    idx = int(m.group(1))
                    if idx not in self.existing_intersection_indices:
                        all_intersections.append(new_intersection)
                        self.existing_intersection_indices.add(idx)
                        print(f"Adding new intersection with index {idx}")
                else:
                    print(f"Warning: Skipping malformed intersection data without front_imgs_rgb")

            # Atomic write
            try:
                atomic_write_json(file_out, all_intersections)
            except Exception as e:
                print(f"Warning: could not write intersections json atomically: {e}")

    def add_intersection_data(self):
        required_keys = ['img1', 'img2', 'img3', 'ss_img1', 'ss_img2', 'ss_img3', 'bev', 'ss_bev']
        if not all(key in self.saved_imgs for key in required_keys):
            print('Sensor images not available.')
            return

        img1_rgb = self.saved_imgs['img1']
        img2_rgb = self.saved_imgs['img2']
        img3_rgb = self.saved_imgs['img3']

        img1_ss = self.saved_imgs['ss_img1']
        img2_ss = self.saved_imgs['ss_img2']
        img3_ss = self.saved_imgs['ss_img3']

        bev_rgb = self.saved_imgs['bev']
        bev_ss = self.saved_imgs['ss_bev']

        all_images = [img1_rgb, img2_rgb, img3_rgb, img1_ss, img2_ss, img3_ss, bev_rgb, bev_ss]
        if any(np.all(img == 0) for img in all_images):
            print('Black image received from sensors.')
            return

        combined_img_rgb = np.concatenate((img1_rgb, img2_rgb, img3_rgb), axis=1)
        combined_img_ss = np.concatenate((img1_ss, img2_ss, img3_ss), axis=1)

        # Use a file lock around index allocation and JSON update to avoid races
        special_maps_dir = self.paths.get('special_maps')
        special_fronts_dir = self.paths.get('special_fronts')
        special_info_dir = self.paths.get('special_info') or '.'

        if not special_maps_dir or not special_fronts_dir:
            print('Warning: special maps/fronts directories not configured. Skipping save.')
            return

        lockfile = os.path.join(special_info_dir, '.intersections.lock')

        with file_lock(lockfile):
            # allocate next index by scanning existing files (protected by lock)
            max_index = -1
            if os.path.exists(special_maps_dir):
                for filename in os.listdir(special_maps_dir):
                    if filename.endswith(('.jpg', '.png')):
                        try:
                            idx_str = filename.split('_')[0]
                            cur = int(idx_str)
                            if cur > max_index:
                                max_index = cur
                        except Exception:
                            continue
            next_idx = max_index + 1
            # Print the index assigned so concurrent runs can see which index was reserved
            print(f"Assigned next intersection index: {next_idx}")

            rgb_img_name = '%09d_rgb.jpg' % next_idx
            ss_img_name = '%09d_ss.png' % next_idx
            rgb_img_path = os.path.join(special_fronts_dir, rgb_img_name)
            ss_img_path = os.path.join(special_fronts_dir, ss_img_name)

            cv2.imwrite(rgb_img_path, combined_img_rgb * 255)
            cv2.imwrite(ss_img_path, combined_img_ss * 255)

            bev_rgb_img_name = '%09d_bev_rgb.jpg' % next_idx
            bev_ss_img_name = '%09d_bev_ss.png' % next_idx
            bev_rgb_img_path = os.path.join(special_maps_dir, bev_rgb_img_name)
            bev_ss_img_path = os.path.join(special_maps_dir, bev_ss_img_name)

            clean_bev_ss_image = None
            if self.args.map == "Town01_Opt":
                clean_bev_ss_image = os.path.join(self.paths['clean_bev_ss_images'], "Town01_Opt_clean_bev.png")
            elif self.args.map == "Town02_Opt":
                clean_bev_ss_image = os.path.join(self.paths['clean_bev_ss_images'], "Town02_Opt_clean_bev.png")
            elif self.args.map == "Town03_Opt":
                clean_bev_ss_image = os.path.join(self.paths['clean_bev_ss_images'], "Town03_Opt_clean_bev.png")
            elif self.args.map == "Town04_Opt":
                clean_bev_ss_image = os.path.join(self.paths['clean_bev_ss_images'], "Town04_Opt_clean_bev.png")
            elif self.args.map == "Town07":
                clean_bev_ss_image = os.path.join(self.paths['clean_bev_ss_images'], "Town07_Opt_clean_bev.png")
            elif self.args.map == "Town15":
                clean_bev_ss_image = os.path.join(self.paths['clean_bev_ss_images'], "Town15_Opt_clean_bev.png")
            else:
                print("WARNING: Clean Bev image path could not be correctly saved. Add map name.")

            cv2.imwrite(bev_rgb_img_path, bev_rgb * 255)
            cv2.imwrite(bev_ss_img_path, bev_ss * 255)

            # Build the entry
            intersection_entry = {
                'origin_position': {'x': float(self.ego_pix_pt[0]), 'y': float(self.ego_pix_pt[1])},
                'destination_position': {'x': float(self.ending_pix_pt[0]), 'y': float(self.ending_pix_pt[1])},
                'front_imgs_rgb': rgb_img_path,
                'front_imgs_ss': ss_img_path,
                'map_img_rgb': bev_rgb_img_path,
                'map_img_ss': bev_ss_img_path,
                'clean_bev_ss_image': clean_bev_ss_image,
                'vehicle_orientation_2d': float(self.ego_yaw),
                'ground_truth': {'type': '', 'possible_exits': '', 'correct_exit': self.last_indication},
                "weather": self.args.weather
            }

            # update persistent intersections.json atomically and avoid duplicates
            file_out = os.path.join(special_info_dir, 'intersections.json')
            all_intersections = []
            if os.path.exists(file_out) and os.path.getsize(file_out) > 0:
                try:
                    with open(file_out, 'r') as f:
                        all_intersections = json.load(f)
                        if not isinstance(all_intersections, list):
                            all_intersections = []
                except Exception:
                    all_intersections = []

            # check duplicate by index
            already = False
            for entry in all_intersections:
                if 'front_imgs_rgb' in entry:
                    base = os.path.basename(entry['front_imgs_rgb'])
                    m = re.match(r"(\d+)", base)
                    if m and int(m.group(1)) == next_idx:
                        already = True
                        break

            if not already:
                all_intersections.append(intersection_entry)
                try:
                    atomic_write_json(file_out, all_intersections)
                except Exception as e:
                    print(f"Warning: could not persist intersection data: {e}")

            # update in-memory and counter
            self.intersections_data.append(intersection_entry)
            self.existing_intersection_indices.add(next_idx)
            self.special_img_counter = next_idx + 1

    def update_info(self):
        control = self.ego.get_control()

        # update indication
        indication = self.tm.get_next_action(self.ego)
        if indication[0] != self.last_indication:
            self.last_indication = indication[0]
            self.ind_counter += 1

        # update speed
        v_speed = self.ego.get_velocity()
        self.speed = math.sqrt(v_speed.x**2 + v_speed.y**2 + v_speed.z**2)

        # update steering
        self.steer = control.steer

        # update acceleration
        accel = 0.0
        if control.throttle > 0.0:
            accel = control.throttle
        elif control.brake > 0.0:
            accel = -1.0 * control.brake
        else:
            accel = 0.0
        self.accel = accel

    def write_data(self):
        name_file = "%09d.json" % (self.counter)
        file_out = os.path.join(self.paths['info'], name_file)
        data = {}

        data['position'] = self.ego_pos
        data['orientation'] = self.ego_orientation
        data['steering'] = self.steer
        data['speed'] = self.speed
        data['acceleration'] = self.accel
        data['action'] = self.last_indication

        with open(file_out, "w") as f:
            json.dump(data, f)

    def destroy_node(self):
        success = False
        while (not success and self.ego.is_alive):
            success = self.ego.destroy()


# ==============================================================================
# -- game_loop() ---------------------------------------------------------------
# ==============================================================================

last_indication_global = ""

def iter_game_loop(args, driving_agent, ego, sim_world, client):
    k = iter_game(args, client, sim_world)
    driving_agent.update_ego_position()
    driving_agent.update_info()

    if k != '':
        driving_agent.update_direction(k)

    if args.mode == 'ai':
        ego.apply_control(driving_agent.control)
        cv2.waitKey(1)

    if args.show_sensors:
        driving_agent.visualize_images()

    """
    # MODS
    if args.save_data:
        driving_agent.write_data()
    """

    # Get correct indication, based on AI driver agent
    global last_indication_global
    if last_indication_global != driving_agent.last_indication:
        print(driving_agent.last_indication)
        last_indication_global = driving_agent.last_indication

    driving_agent.counter = driving_agent.counter + 1

    # MODS: Proactive intersection detection
    current_waypoint = sim_world.get_map().get_waypoint(ego.get_location(), project_to_road=True)

    if current_waypoint.is_junction:
        if not driving_agent.is_in_junction:
            print('Approaching intersection in, capturing data...')
            driving_agent.add_intersection_data()
            driving_agent.write_intersections_json()  # Save immediately to avoid data loss
            driving_agent.is_in_junction = True
    else:
        if driving_agent.is_in_junction:
            driving_agent.is_in_junction = False


def game_loop(args):
    original_settings = None
    driving_agent = None
    client = None

    try:
        client, sim_world, original_settings = init_game(args)
        driving_agent = AIDriverAgent(args, sim_world, client)
        ego = get_ego_vehicle()

        print("Getting Traffic Manager")
        tm = client.get_trafficmanager(args.tm_port)
        print("Got Traffic Manager")
        
        # MOD: ignore traffic lights for the ego vehicle to prevent stopping at red lights
        tm.ignore_lights_percentage(ego, 100.0)


        # !!!!!! MODS FOR CALCULATING REAL ROUTE !!!!!!

        # MOD: Calculates real route using map topology
        amap = sim_world.get_map()
        grp = GlobalRoutePlanner(amap, 2.0)

        init_loc = carla.Location(x=args.ego_starting_position[0], y=args.ego_starting_position[1], z=args.ego_starting_position[2])
        end_loc = carla.Location(x=args.ego_ending_position[0], y=args.ego_ending_position[1], z=args.ego_ending_position[2])
        
        # Trazar ruta: devuelve una lista de tuplas (waypoint, RoadOption)
        route_trace = grp.trace_route(init_loc, end_loc)
        
        # Extraer solo las location de los waypoints para el Traffic Manager
        route_path = [w[0].transform.location for w in route_trace]
        
        # Pasarle la lista COMPLETA de puntos al Traffic Manager
        tm.set_path(ego, route_path)

        ego.set_autopilot(True, args.tm_port)

        """
        # Initial route planning
        init_point = carla.Location(x=args.ego_starting_position[0], y=args.ego_starting_position[1], z=args.ego_starting_position[2])
        end_point = carla.Location(x=args.ego_ending_position[0], y=args.ego_ending_position[1], z=args.ego_ending_position[2])
        tm.set_path(ego, [init_point, end_point])
        """

        # !!!!!! MODS FOR CALCULATING REAL ROUTE !!!!!!


        end_diff = 1000.

        print("Bon voyage!") # MOD
        while(end_diff > 1.0):
            ego = get_ego_vehicle()
            iter_game_loop(args, driving_agent, ego, sim_world, client)
            ego_pos = ego.get_transform().location
            ego_pos = (ego_pos.x, ego_pos.y, ego_pos.z)
            end_diff = estimate_distance(ego_pos, args.ego_ending_position)

    finally:

        if original_settings:
            sim_world.apply_settings(original_settings)

        # close_ui(client)
        try:
            close_world(client, sim_world)
        except Exception as e:
            pass

        if driving_agent is not None:
            # MOD: persist collected intersection data
            try:
                driving_agent.write_intersections_json()
            except Exception as e:
                print(f"Warning: could not write intersections json: {e}")
            driving_agent.destroy_node()

# ==============================================================================
# -- main() --------------------------------------------------------------------
# ==============================================================================


def main():
    argparser = argparse.ArgumentParser(
        description='CARLA Manual Control Client')
    argparser.add_argument(
        '-v', '--verbose',
        action='store_true',
        dest='debug',
        help='print debug information')
    argparser.add_argument(
        '--host',
        metavar='H',
        default='127.0.0.1',
        help='IP of the host server (default: 127.0.0.1)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '-t', '--tm_port',
        metavar='T',
        default=8000,
        type=int,
        help='TCP TrafficManager port to listen to (default: 8000)')
    argparser.add_argument(
        '-m', '--mode',
        choices=['manual', 'auto', 'ai'],
        default='auto',
        type=str,
        help='choose initial driving mode of the vehicle (default: manual)')
    argparser.add_argument(
        '--res',
        metavar='WIDTHxHEIGHT',
        default='1280x720',
        help='window resolution (default: 1280x720)')
    argparser.add_argument(
        '--filter',
        metavar='PATTERN',
        default='vehicle.*',
        help='actor filter (default: "vehicle.*")')
    argparser.add_argument(
        '--generation',
        metavar='G',
        default='2',
        help='restrict to certain actor generation (values: "1","2","All" - default: "2")')
    argparser.add_argument(
        '--rolename',
        metavar='NAME',
        default='hero',
        help='actor role name (default: "hero")')
    
    argparser.add_argument(
        '--wait_for_hero',
        action='store_true',
        help='wait for a hero vehicle or create a new one')
    
    argparser.add_argument(
        '--map',
        metavar='TOWN',
        type=str,
        default='Town01',
        help='load a new town (default: Town01)')
    
    argparser.add_argument(
        '--gamma',
        default=2.2,
        type=float,
        help='Gamma correction of the camera (default: 2.2)')
    argparser.add_argument(
        '--sync',
        action='store_true',
        default=True,
        help='Activate synchronous mode execution')
    argparser.add_argument(
        '--show_sensors',
        action='store_true',
        default=False,
        help='Show installed sensors')
    argparser.add_argument(
        '--save_data',
        action='store_true',
        default=False,
        help='Save data to disk')
    argparser.add_argument(
        '--data_path',
        type=str,
        default='./out',
        help='Path to save data')
    argparser.add_argument(
        '--ego_yaw',
        default=None,
        type=float,
        help='Ego yaw location')
    argparser.add_argument(
        '--ego_starting_position',
        type=float,
        nargs=3,
        metavar=('x', 'y', 'z'),
        help='Ego starting position')
    argparser.add_argument(
        '--ego_ending_position',
        type=float,
        nargs=3,
        metavar=('x', 'y', 'z'),
        help='Ego ending position')
    argparser.add_argument(
        '--weather',
        type=str,
        help='Set simulation weather conditions'
    )

    args = argparser.parse_args()

    args.width, args.height = [int(x) for x in args.res.split('x')]

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)

    logging.info('listening to server %s:%s', args.host, args.port)

    try:
        print("Game loop starts")
        game_loop(args)

    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')


if __name__ == '__main__':

    main()
