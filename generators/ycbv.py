"""
Source Code from EfficientPose implementation (https://github.com/ybkscht/) licensed under the Apache License, Version 4.0

EfficientPose is licensed under a
Creative Commons Attribution-NonCommercial 4.0 International License.

The license can be found in the LICENSE file in the root directory of this source tree
or at http://creativecommons.org/licenses/by-nc/4.0/.
---------------------------------------------------------------------------------------------------------------------------------
---------------------------------------------------------------------------------------------------------------------------------

Based on:

Keras EfficientDet implementation (https://github.com/xuannianz/EfficientDet) licensed under the Apache License, Version 2.0
---------------------------------------------------------------------------------------------------------------------------------
The official EfficientDet implementation (https://github.com/google/automl) licensed under the Apache License, Version 2.0
---------------------------------------------------------------------------------------------------------------------------------
EfficientNet Keras implementation (https://github.com/qubvel/efficientnet) licensed under the Apache License, Version 2.0
---------------------------------------------------------------------------------------------------------------------------------
Keras RetinaNet implementation (https://github.com/fizyr/keras-retinanet) licensed under the Apache License, Version 2.0
"""

import os
import numpy as np
import cv2
import yaml
import random
import json
import copy
from plyfile import PlyData
from pathlib import Path
import sys
from tqdm import tqdm
sys.path.append(".")
sys.path.append("./bop_toolkit")
# from bop_toolkit_lib import inout

from generators.common import Generator

from collections import defaultdict

class DatasetConfig:
    model_folder = 'models'
    train_folder = 'train'
    test_folder = 'test'
    img_folder = 'rgb'
    depth_folder = 'depth'
    mask_folder = 'mask'
    mask_ext = 'png'
    img_ext = 'png'
    depth_ext = 'png'


config = defaultdict(lambda *_: DatasetConfig())

config['tless'] = tless = DatasetConfig()
tless.model_folder = 'models_cad'
tless.test_folder = 'test_primesense'
tless.train_folder = 'train_primesense'

config['hb'] = hb = DatasetConfig()
hb.test_folder = 'test_primesense'

config['itodd'] = itodd = DatasetConfig()
itodd.depth_ext = 'tif'
itodd.img_folder = 'gray'
itodd.img_ext = 'tif'

config['ycbv'] = ycbv = DatasetConfig()
ycbv.img_ext = 'png'

#Generator for the LINEMOD Dataset downloaded from here: https://github.com/j96w/DenseFusion
class YCBVGenerator(Generator):
    """
    Generator for the Linemod dataset

    """
    def __init__(self, 
                 dataset_base_path,
                 object_id,
                 image_extension = ".png",
                 shuffle_dataset = True,                    
                 symmetric_objects = {"master_chef_can",1, "bowl", 13, "mug", 14, "wood_block", 16, "large_marker", 18, "large_clamp", 19, "extra_large_clamp", 20, "foam_brick", 21},#{"glue", 11, "eggbox", 10}, #set with names and indices of symmetric objects
                 **kwargs):
        """
        Initializes a Linemod generator
        Args:
            dataset_base_path: path to the Linemod dataset
            object_id: Integer object id of the Linemod object on which to generate data
            image_extension: String containing the image filename extension
            shuffle_dataset: Boolean wheter to shuffle the dataset or not
             symmetric_objects: set with names and indices of symmetric objects
        
        """
        
        #check and set the rotation representation and the number of parameters to use
        self.init_num_rotation_parameters(**kwargs)
        
        self.dataset_base_path = dataset_base_path
        
        if not "train" in kwargs or kwargs["train"]:
            train = True
        else:
            train = False
            
        self.model_path = os.path.join(self.dataset_base_path, "models")
        self.object_id = object_id
        self.image_extension = image_extension
        self.shuffle_dataset = shuffle_dataset
        self.translation_parameter = 3
        self.symmetric_objects = symmetric_objects      
        
        #check if both paths exist
        if not self.check_path(self.dataset_base_path) or not self.check_path(self.dataset_base_path) or not self.check_path(self.model_path):
            return None
        
        #get dict with object ids as keys and object subdirs as values
       
        
        if not self.object_id in list(range(1, 22)):
            print("The given object id {} was not found in the dataset dir {}".format(self.object_id, self.dataset_path))
            return None
        #set the class and name dict for mapping each other
        self.class_to_name = {0: "object"}
        self.name_to_class = {"object": 0}
        self.name_to_mask_value = {"object": 255}
        self.object_ids_to_class_labels = {self.object_id: 0}
        self.class_labels_to_object_ids = {0: self.object_id}
                      
        #parse yaml files with ground truth annotations and infos about camera intrinsics and 3D BBox
        
        self.all_models_dict = self.parse_yaml(self.model_path, filename = "models_info.yml")
        
        #get the model with the given object id
        self.model_dict = self.all_models_dict[self.object_id]
        #load the complete 3d model from the ply file
        self.model_3d_points = self.load_model_ply(path_to_ply_file = os.path.join(self.model_path, "obj_{:06}.ply".format(self.object_id)))
        self.class_to_model_3d_points = {0: self.model_3d_points}
        self.name_to_model_3d_points = {"object": self.model_3d_points}
        
        #create dict with the class indices/names as keys and 3d model diameters as values
        self.class_to_model_3d_diameters, self.name_to_model_3d_diameters = self.create_model_3d_diameters_dict(self.all_models_dict, self.object_ids_to_class_labels, self.class_to_name)
        
        #create dict with the class indices/names as keys and model 3d bboxes as values
        self.class_to_model_3d_bboxes, self.name_to_model_3d_bboxes = self.create_model_3d_bboxes_dict(self.all_models_dict, self.object_ids_to_class_labels, self.class_to_name)
        
        #get the final input and annotation infos for the base generator
        self.image_paths, self.mask_paths, self.annotations, self.infos = self.prepare_dataset(dataset_path=Path(self.dataset_base_path), train=train, cfg=ycbv)        
        
        #shuffle dataset
        if self.shuffle_dataset:
            self.image_paths, self.mask_paths, self.annotations, self.infos = self.shuffle_sequences(self.image_paths, self.mask_paths, self.annotations, self.infos)
            
        
        #init base class
        Generator.__init__(self, **kwargs)
        
    
    def parse_yaml(self, object_path, filename = "gt.yml"):
        """
        Reads a yaml file
        Args:
            object_path: Path to the yaml file
            filename: filename of the yaml file
        Returns:
            yaml_dic: Dictionary containing the yaml file content
    
        """
        yaml_path = os.path.join(object_path, filename)
        
        if not os.path.isfile(yaml_path):
            print("Error: file {} does not exist!".format(yaml_path))
            return None
        
        with open(yaml_path) as fid:
            yaml_dic = yaml.safe_load(fid)
            
        return yaml_dic
        
    
    def load_json(self, path, keys_to_int=False):
        """Loads content of a JSON file.

        :param path: Path to the JSON file.
        :return: Content of the loaded JSON file.
        """
        # Keys to integers.
        def convert_keys_to_int(x):
            return {int(k) if k.lstrip('-').isdigit() else k: v for k, v in x.items()}

        with open(path, 'r') as f:
            if keys_to_int:
                content = json.load(f, object_hook=lambda x: convert_keys_to_int(x))
            else:
                content = json.load(f)

        return content
    
    def load_gt(self, path):
        content = self.load_json(path)
        
        
        return content
     
    def load_all_scenes_camera(self, path):
        
        content = self.load_json(path)
        for value in content: 
            value['cam_K_np'] = np.reshape(np.array(value['cam_K']), newshape = (3, 3))
        
        return content

    def get_bbox_3d(self, model_dict):
        """
        Converts the 3D model cuboid from the Linemod format (min_x, min_y, min_z, size_x, size_y, size_z) to the (num_corners = 8, num_coordinates = 3) format
        Args:
            model_dict: Dictionary containing the cuboid information of a single Linemod 3D model in the Linemod format
        Returns:
            bbox: numpy (8, 3) array containing the 3D model's cuboid, where the first dimension represents the corner points and the second dimension contains the x-, y- and z-coordinates.
    
        """
        #get infos from model dict
        min_point_x = model_dict["min_x"]
        min_point_y = model_dict["min_y"]
        min_point_z = model_dict["min_z"]
        
        size_x = model_dict["size_x"]
        size_y = model_dict["size_y"]
        size_z = model_dict["size_z"]
        
        bbox = np.zeros(shape = (8, 3))
        #untere ebende
        bbox[0, :] = np.array([min_point_x, min_point_y, min_point_z])
        bbox[1, :] = np.array([min_point_x + size_x, min_point_y, min_point_z])
        bbox[2, :] = np.array([min_point_x + size_x, min_point_y + size_y, min_point_z])
        bbox[3, :] = np.array([min_point_x, min_point_y + size_y, min_point_z])
        #obere ebene
        bbox[4, :] = np.array([min_point_x, min_point_y, min_point_z + size_z])
        bbox[5, :] = np.array([min_point_x + size_x, min_point_y, min_point_z + size_z])
        bbox[6, :] = np.array([min_point_x + size_x, min_point_y + size_y, min_point_z + size_z])
        bbox[7, :] = np.array([min_point_x, min_point_y + size_y, min_point_z + size_z])
        
        return bbox
    
    
    def get_bbox_3d_dict(self, class_idx_as_key = True):
        """
       Returns a dictionary which either maps the class indices or the class names to the 3D model cuboids
        Args:
            class_idx_as_key: Boolean indicating wheter to return the class indices or the class names as keys
        Returns:
            Dictionary containing the class indices or the class names as keys and the 3D model cuboids as values
    
        """
        if class_idx_as_key:
            return self.class_to_model_3d_bboxes
        else:
            return self.name_to_model_3d_bboxes
        
        
    def create_model_3d_bboxes_dict(self, all_models_dict, object_ids_to_class_labels, class_to_name):
        """
       Creates two dictionaries which are mapping the class indices, respectively the class names to the 3D model cuboids
        Args:
            all_models_dict: Dictionary containing all 3D model's bboxes in the Linemod dataset format (min_x, min_y, min_z, size_x, size_y, size_z)
            object_ids_to_class_labels: Dictionary mapping the Linemod object ids to the mdoel classes
            class_to_name: Dictionary mapping the model classes to their names
        Returns:
            Two dictionaries containing the model class indices or the class names as keys and the 3D model cuboids as values
    
        """
        class_to_model_3d_bboxes = dict()
        name_to_model_3d_bboxes = dict()
        
        for object_id, class_label in object_ids_to_class_labels.items():
            model_bbox = self.get_bbox_3d(all_models_dict[object_id])
            class_to_model_3d_bboxes[class_label] = model_bbox
            name_to_model_3d_bboxes[class_to_name[class_label]] = model_bbox
            
        return class_to_model_3d_bboxes, name_to_model_3d_bboxes
    
    
    def get_models_3d_points_dict(self, class_idx_as_key = True):
        """
       Returns either the 3d model points dict with class idx as key or the model name as key
        Args:
            class_idx_as_key: Boolean indicating wheter to return the class indices or the class names as keys
        Returns:
            Dictionary containing the class indices or the class names as keys and the 3D model points as values
    
        """
        if class_idx_as_key:
            return self.class_to_model_3d_points
        else:
            return self.name_to_model_3d_points
        
        
    def get_objects_diameter_dict(self, class_idx_as_key = True):
        """
       Returns either the diameter dict with class idx as key or the model name as key
        Args:
            class_idx_as_key: Boolean indicating wheter to return the class indices or the class names as keys
        Returns:
            Dictionary containing the class indices or the class names as keys and the 3D model diameters as values
    
        """
        if class_idx_as_key:
            return self.class_to_model_3d_diameters
        else:
            return self.name_to_model_3d_diameters
        
        
    def create_model_3d_diameters_dict(self, all_models_dict, object_ids_to_class_labels, class_to_name):
        """
       Creates two dictionaries containing the class idx and the model name as key and the 3D model diameters as values
        Args:
            all_models_dict: Dictionary containing all 3D model's bboxes and diameters in the Linemod dataset format
            object_ids_to_class_labels: Dictionary mapping the Linemod object ids to the model classes
            class_to_name: Dictionary mapping the model classes to their names
        Returns:
            Dictionary containing the class indices or the class names as keys and the 3D model diameters as values
    
        """
        class_to_model_3d_diameters = dict()
        name_to_model_3d_diameters = dict()
        
        for object_id, class_label in object_ids_to_class_labels.items():
            class_to_model_3d_diameters[class_label] = all_models_dict[object_id]["diameter"]
            name_to_model_3d_diameters[class_to_name[class_label]] = all_models_dict[object_id]["diameter"]
            
        return class_to_model_3d_diameters, name_to_model_3d_diameters
    
    
    def is_symmetric_object(self, name_or_object_id):
        """
       Check if the given object is considered to be symmetric or not
        Args:
            name_or_object_id: The name of the object or the id of the object
        Returns:
            Boolean indicating wheter the object is symmetric or not
    
        """
        return name_or_object_id in self.symmetric_objects
    
    
    def get_models_3d_points_list(self):
        """
       Returns a list with all models 3D points. In case of Linemod there is only a single element in the list
    
        """
        return [self.model_3d_points]
    
    
    def get_objects_diameter_list(self):
        """
       Returns a list with all models 3D diameters. In case of Linemod there is only a single element in the list
    
        """
        return [self.model_dict["diameter"]]
        
    
    def get_object_diameter(self):
        """
       Returns the object's 3D model diameter
    
        """
        return self.model_dict["diameter"]
    
        
    def get_num_rotation_parameters(self):
        """
       Returns the number of rotation parameters. For axis angle representation there are 3 parameters used
    
        """
        return self.rotation_parameter
    
    
    def get_num_translation_parameters(self):
        """
       Returns the number of translation parameters. Usually 3 
    
        """
        return self.translation_parameter
            
        
    def shuffle_sequences(self, image_paths, mask_paths, annotations, infos):
        """
       Takes sequences (e.g. lists) containing the dataset and shuffle them so that the corresponding entries still match
    
        """
        concatenated = list(zip(image_paths, mask_paths, annotations, infos))
        random.shuffle(concatenated)
        image_paths, mask_paths, annotations, infos = zip(*concatenated)
        
        return image_paths, mask_paths, annotations, infos
    
    
    def load_model_ply(self, path_to_ply_file):
        """
       Loads a 3D model from a plyfile
        Args:
            path_to_ply_file: Path to the ply file containing the object's 3D model
        Returns:
            points_3d: numpy array with shape (num_3D_points, 3) containing the x-, y- and z-coordinates of all 3D model points
    
        """
        model_data = PlyData.read(path_to_ply_file)
                                  
        vertex = model_data['vertex']
        points_3d = np.stack([vertex[:]['x'], vertex[:]['y'], vertex[:]['z']], axis = -1)
        
        return points_3d
        
    def get_scene_camera(self, scene):
        """Loads content of a JSON file with information about the scene camera.

        See docs/bop_datasets_format.md for details.

        :param path: Path to the JSON file.
        :return: Dictionary with the loaded content.
        """

        scene['cam_K'] = np.array(scene['cam_K'], np.float).reshape((3, 3))
        # scene['cam_R_w2c'] = np.array(scene['cam_R_w2c'], np.float).reshape((3, 3))
        # scene['cam_t_w2c'] = np.array(scene['cam_t_w2c'], np.float).reshape((3, 1))
        scene['cam_K_np'] = np.reshape(np.array(scene["cam_K"]), newshape = (3, 3))
        
        return scene

    def prepare_dataset(self, dataset_path: Path, train: bool, cfg: DatasetConfig, 
                        min_px_count_visib=1024, scene_ids=None, min_visib_fract=0.1,
                        show_progressbar=True):
        """
       Prepares the Linemod dataset and converts the data from the Linemod format to the model format
        Args:
            object_path: path to the single Linemod object
            data_examples: List containing all data examples of the used dataset split (train or test)
            gt_dict: Dictionary mapping the example id's to the corresponding ground truth data
            info_dict: Dictionary mapping the example id's to the intrinsic camera parameters
        Returns:
            image_paths: List with all rgb image paths in the dataset split
            mask_paths: List with all segmentation mask paths in the dataset split
            depth_paths: List with all depth image paths in the dataset split (Currently not used in the model)
            annotations: List with all annotation dictionaries in the dataset split
            infos: List with all info dictionaries (intrinsic camera parameters) in the dataset split
    
        """
        
        # data = instance.BopInstanceDataset(dataset_root=Path(dataset_path), test=True, cfg=instance.ycbv, object_id=self.object_id)
        image_paths = []
        mask_paths = []
        infos = []
        gts = []
        # for item in data.instances: #loop over a seqeunce                        
        #     mask_paths.append(item['mask_path'] )
        #     image_paths.append(item['rgb_path'])         
        #     infos.append(self.get_scene_camera(item['scene_cam']))
        #     gts.append({'cam_R_m2c': item['cam_R_obj'], 'cam_t_m2c': item['cam_t_obj']})
        
        annotations = self.convert_gt(gts, infos, mask_paths)       
        
        data_folder = dataset_path / (cfg.train_folder if train else cfg.test_folder)
        img_folder = cfg.img_folder
        mask_folder = cfg.mask_folder
        img_ext = cfg.img_ext if train else 'png'
        if scene_ids is None:
            scene_ids = sorted([int(p.name) for p in data_folder.glob('*')])
        for scene_id in tqdm(scene_ids, 'loading crop info'):
            scene_folder = data_folder / f'{scene_id:06d}'
            scene_gt = json.load((scene_folder / 'scene_gt.json').open())
            scene_gt_info = json.load((scene_folder / 'scene_gt_info.json').open())
            scene_camera = json.load((scene_folder / 'scene_camera.json').open())

            for img_id, poses in scene_gt.items():
                img_info = scene_gt_info[img_id]
                scene_cam = scene_camera[img_id]

                for pose_idx, pose in enumerate(poses):
                    if pose['obj_id'] == self.object_id:
                        
                        pose_info = img_info[pose_idx]
                        if pose_info['visib_fract'] < min_visib_fract:
                            continue
                        if pose_info['px_count_visib'] < min_px_count_visib:
                            continue

                        bbox_visib = pose_info['bbox_visib']
                        bbox_obj = pose_info['bbox_obj']

                        cam_R_obj = np.array(pose['cam_R_m2c'])
                        cam_t_obj = np.array(pose['cam_t_m2c'])

                        rgb_path = data_folder / f'{scene_id:06d}/{img_folder}/{int(img_id):06d}.{img_ext}'
                        rgb_path = rgb_path.__str__()
                        assert os.path.exists(rgb_path) is not None
                        
                        mask_path = data_folder / f'{scene_id:06d}/{mask_folder}/{int(img_id):06d}_{pose_idx:06d}.png'
                        mask_path = mask_path.__str__()
                        # mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                        assert os.path.exists(mask_path )
                        mask_paths.append(mask_path)
                        image_paths.append(rgb_path)         
                        infos.append(self.get_scene_camera(scene_cam))
                        gts.append({'cam_R_m2c': cam_R_obj, 'cam_t_m2c': cam_t_obj})
                                                
        annotations = self.convert_gt(gts, infos, mask_paths)       

        return image_paths, mask_paths, annotations, infos
    
    
    def insert_np_cam_calibration(self, filtered_infos):
        """
       Converts the intrinsic camera parameters in each dict of the given list into a numpy (3, 3) camera matrix
        Args:
            filtered_infos: List with all dictionaries containing the intrinsic camera parameters
        Returns:
            filtered_infos: List with all dictionaries containing the intrinsic camera parameters also as a numpy (3, 3) array
    
        """
        for info in filtered_infos:
            if not info is None:
                info["cam_K_np"] = np.reshape(np.array(info["cam_K"]), newshape = (3, 3))
        
        return filtered_infos
    
    
    def convert_gt(self, gt_list, info_list, mask_paths):
        """
       Prepares the annotations from the Linemod dataset format into the model format
        Args:
            gt_list: List with all ground truth dictionaries in the dataset split
            info_list: List with all info dictionaries (intrinsic camera parameters) in the dataset split
            mask_paths: List with all segmentation mask paths in the dataset split
        Returns:
            all_annotations: List with the converted ground truth dictionaries
    
        """
        all_annotations = []
        for gt, info, mask_path in tqdm(zip(gt_list, info_list, mask_paths)):
            if not gt is None:
                #init annotations in the correct base format. set number of annotations to one because linemod dataset only contains one annotation per image
                num_all_rotation_parameters = self.rotation_parameter + 2 #+1 for class id and +1 for is_symmetric flag
                annotations = {'labels': np.zeros((1,)),
                            'bboxes': np.zeros((1, 4)),
                            'rotations': np.zeros((1, num_all_rotation_parameters)),
                            'translations': np.zeros((1, self.translation_parameter)),
                            'translations_x_y_2D': np.zeros((1, 2))}
                
                #fill in the values
                #class label is always zero because there is only one possible object
                #get bbox from mask
                mask = cv2.imread(mask_path)
                annotations["bboxes"][0, :], _ = self.get_bbox_from_mask(mask)
                #transform rotation into the needed representation
                annotations["rotations"][0, :-2] = self.transform_rotation(np.array(gt["cam_R_m2c"]), self.rotation_representation)
                annotations["rotations"][0, -2] = float(self.is_symmetric_object(self.object_id))
                annotations["rotations"][0, -1] = float(0) #useless for linemod because there is only one object but neccessary to keep compatibility of the architecture with multi-object datasets
                
                annotations["translations"][0, :] = np.array(gt["cam_t_m2c"])
                annotations["translations_x_y_2D"][0, :] = self.project_points_3D_to_2D(points_3D = np.zeros(shape = (1, 3)), #transform the object origin point which is the centerpoint
                                                                                        rotation_vector = self.transform_rotation(np.array(gt["cam_R_m2c"]), "axis_angle"),
                                                                                        translation_vector = np.array(gt["cam_t_m2c"]),
                                                                                        camera_matrix = info["cam_K_np"])
            
            all_annotations.append(annotations)
        
        return all_annotations
    
    
    def convert_bboxes(self, bbox):
        """
       Convert bbox from (x1, y1, width, height) to (x1, y1, x2, y2) format
        Args:
            bbox: numpy array (x1, y1, width, height)
        Returns:
            new_bbox: numpy array (x1, y1, x2, y2)
    
        """
        new_bbox = np.copy(bbox)
        new_bbox[2] += new_bbox[0]
        new_bbox[3] += new_bbox[1]
        
        return new_bbox
    
        
    def parse_yaml(self, object_path, filename = "gt.yml"):
        """
       Reads a yaml file
        Args:
            object_path: Path to the yaml file
            filename: filename of the yaml file
        Returns:
            yaml_dic: Dictionary containing the yaml file content
    
        """
        yaml_path = os.path.join(object_path, filename)
        
        if not os.path.isfile(yaml_path):
            print("Error: file {} does not exist!".format(yaml_path))
            return None
        
        with open(yaml_path) as fid:
            yaml_dic = yaml.safe_load(fid)
            
        return yaml_dic
        
    
    def check_path(self, path):
        """
        Check if the given path exists
        """
        if not os.path.exists(path):
            print("Error: path {} does not exist!".format(path))
            return False
        else:
            return True
        
        
    def parse_examples(self, data_file):
        """
       Reads the Linemod dataset split (train or test) txt file containing the examples of this split
        Args:
            data_file: Path to the dataset split file
        Returns:
            data_examples: List containing all data example id's of the used dataset split
    
        """
        if not os.path.isfile(data_file):
            print("Error: file {} does not exist!".format(data_file))
            return None
        
        with open(data_file) as fid:
            data_examples = [example.strip() for example in fid if example != ""]
            
        return data_examples
        
        
    #needed function implementations of the generator base class
    def size(self):
        """ Size of the dataset.
        """
        return len(self.image_paths)

    def num_classes(self):
        """ Number of classes in the dataset.
        """
        return len(self.class_to_name)

    def has_label(self, label):
        """ Returns True if label is a known label.
        """
        return label in self.class_to_name

    def has_name(self, name):
        """ Returns True if name is a known class.
        """
        return name in self.name_to_class

    def name_to_label(self, name):
        """ Map name to label.
        """
        return self.name_to_class[name]

    def label_to_name(self, label):
        """ Map label to name.
        """
        return self.class_to_name[label]

    def image_aspect_ratio(self, image_index):
        """ Compute the aspect ratio for an image with image_index.
        """
        #image aspect ratio is fixed on Linemod dataset
        return 640. / 480.

    def load_image(self, image_index):
        """ Load an image at the image_index.
        """
        #load image and switch BGR to RGB
        image = cv2.imread(self.image_paths[image_index])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image
    
    def load_mask(self, image_index):
        """ Load mask at the image_index.
        """
        return cv2.imread(self.mask_paths[image_index])

    def load_annotations(self, image_index):
        """ Load annotations for an image_index.
        """
        return copy.deepcopy(self.annotations[image_index])
    
    def load_camera_matrix(self, image_index):
        """ Load intrinsic camera parameter for an image_index.
        """
        return np.copy(self.infos[image_index]["cam_K_np"])
        
    

if __name__ == "__main__":
    #test linemod generator
    # train_gen = YCBVGenerator("D:\\CV\\bop\\datasets\\ycbv", object_id = 1, rotation_representation = 'axis_angle')
    for i in range(1,22):
        test_gen = YCBVGenerator("D:\\CV\\bop\\datasets\\ycbv", object_id = i, rotation_representation = 'axis_angle', train = False)
        print(len(test_gen))
    
    # img, anno = test_gen[0]
    
    