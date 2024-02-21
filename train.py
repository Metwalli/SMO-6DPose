"""
Dense6DPose (c) by Metwalli Al-Selwi
contact: Metwalli.msn@gmail.com

Dense6DPose is licensed under a
Creative Commons Attribution-NonCommercial 4.0 International License.

The license can be found in the LICENSE file in the root directory of this source tree
or at http://creativecommons.org/licenses/by-nc/4.0/.
---------------------------------------------------------------------------------------------------------------------------------
---------------------------------------------------------------------------------------------------------------------------------

Based on:

EfficientPose implementation (https://github.com/ybkscht/EfficientPose) licensed under the Apache License, Version 4.0

Keras EfficientDet implementation (https://github.com/xuannianz/EfficientDet) licensed under the Apache License, Version 2.0
---------------------------------------------------------------------------------------------------------------------------------
The official EfficientDet implementation (https://github.com/google/automl) licensed under the Apache License, Version 2.0
---------------------------------------------------------------------------------------------------------------------------------
EfficientNet Keras implementation (https://github.com/qubvel/efficientnet) licensed under the Apache License, Version 2.0
---------------------------------------------------------------------------------------------------------------------------------
Keras RetinaNet implementation (https://github.com/fizyr/keras-retinanet) licensed under
    
Copyright 2017-2018 Fizyr (https://fizyr.com)

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import argparse
from email import generator
# from curses import init_color
import time
import os
import sys
import math

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler

from model import build_Model
from losses import smooth_l1, focal, transformation_loss
from efficientnet import BASE_WEIGHTS_PATH, WEIGHTS_HASHES

from loss_history import LossHistory
from utils.params import Params

def parse_args(args):
    """
    Parse the arguments.
    """
    date_and_time = time.strftime("%d_%m_%Y_%H_%M_%S")
    parser = argparse.ArgumentParser(description = 'Simple model training script.')
    subparsers = parser.add_subparsers(help = 'Arguments for specific dataset types.', dest = 'dataset_type')
    subparsers.required = True
    
    linemod_parser = subparsers.add_parser('linemod')
    linemod_parser.add_argument('linemod_path', help = 'Path to dataset directory (ie. /Datasets/Linemod_preprocessed).')
    linemod_parser.add_argument('--object-id', help = 'ID of the Linemod Object to train on', type = int, default = 8)
    
    linemod_parser = subparsers.add_parser('ycbv')
    linemod_parser.add_argument('ycbv_path', help = 'Path to dataset directory (ie. /bop/Datasets/ycbv).')
    linemod_parser.add_argument('--object-id', help = 'ID of the Linemod Object to train on', type = int, default = 1)
    
    occlusion_parser = subparsers.add_parser('occlusion')
    occlusion_parser.add_argument('occlusion_path', help = 'Path to dataset directory (ie. /Datasets/Linemod_preprocessed/).')

    parser.add_argument('--rotation-representation', help = 'Which representation of the rotation should be used. Choose from "axis_angle", "rotation_matrix" and "quaternion"', default = 'axis_angle')    

    parser.add_argument('--weights', help = 'File containing weights to init the model parameter')
    parser.add_argument('--freeze_backbone', help = 'Freeze training of backbone layers.', default= False, type=bool)
    parser.add_argument('--no-freeze-bn', help = 'Do not freeze training of BatchNormalization layers.', action = 'store_true')
    parser.add_argument('--backbone', help = 'The backbone featrure extraction model(efficientnet or densenet).', default = 'densenet')
    parser.add_argument('--model_dir', help = 'The directory which obtain params json file.', default = 'experiments/densenet')
    parser.add_argument('--config', help = 'The Config json file.', default = 'config/cfg_bop2019.json')

    parser.add_argument('--batch_size', help = 'Size of the batches.', default = 1, type = int)
    parser.add_argument('--lr', help = 'Learning rate', default = 1e-4, type = float)
    parser.add_argument('--no-color-augmentation', help = 'Do not use colorspace augmentation', action = 'store_true')
    parser.add_argument('--no-6dof-augmentation', help = 'Do not use 6DoF augmentation', action = 'store_true')
    parser.add_argument('--phi', help = 'Hyper parameter phi', default = 0, type = int, choices = (0, 1, 2, 3, 4, 5, 6))
    parser.add_argument('--gpu', help = 'Id of the GPU to use (as reported by nvidia-smi).')
    parser.add_argument('--epochs', help = 'Number of epochs to train.', type = int, default = 20)
    parser.add_argument('--steps', help = 'Number of steps per epoch.', type = int)
    parser.add_argument('--snapshot-path', help = 'Path to store snapshots of models during training', default = os.path.join("checkpoints", date_and_time))
    parser.add_argument('--save_tensorboard', help = 'Log directory for Tensorboard output', dest = 'save_tensorboard', action = 'store_true')
    parser.add_argument('--no-snapshots', help = 'Disable saving snapshots.', dest = 'snapshots', action = 'store_false')
    parser.add_argument('--evaluation', help = 'Disable per epoch evaluation.', dest = 'evaluation', action = 'store_false')
    parser.add_argument('--compute-val-loss', help = 'Compute validation loss during training', dest = 'compute_val_loss', action = 'store_true')
    parser.add_argument('--score-threshold', help = 'score threshold for non max suppresion', type = float, default = 0.5)
    parser.add_argument('--validation-image-save-path', help = 'path where to save the predicted validation images after each epoch', default = None)

    # Fit generator arguments
    parser.add_argument('--multiprocessing', help = 'Use multiprocessing in fit_generator.', action = 'store_true')
    parser.add_argument('--workers', help = 'Number of generator workers.', type = int, default = 4)
    parser.add_argument('--max-queue-size', help = 'Queue length for multiprocessing workers in fit_generator.', type = int, default = 10)
    
    print(vars(parser.parse_args(args)))
    return parser.parse_args(args)


def main(args = None):
    """
    Train an the model.

    Args:
        args: parseargs object containing configuration for the training procedure.
    """    
    # python train.py --phi 0 --epochs 500 --freeze_backbone false --backbone densenet occlusion d:cv\dataset\Linemod_preprocessed
    # python train.py --phi 0 --steps 181  --epochs 20 --snapshot-path checkpoints\01_06_2022_18_28_05 --backbone efficientnet occlusion d:cv\dataset\Linemod_preprocessed
    # parse arguments
    # args = ['--phi', '0', '--backbone', 'densenet', 'occlusion', 'D:\CV\DATASET\Linemod_preprocessed']
    # args = ['--phi', '0', '--backbone', 'densenet', 'ycbv', 'D:\CV\\bop\\datasets\\ycbv']
    # assert args is None, "Model Dir should be provided"

    # args = []
    # model_dir = "experiments/densenet"
    # params = Params(os.path.join(model_dir, 'params.json'))
    # args.append(params)
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)

    
    # load the user configs
    # tf.compat.v1.disable_eager_execution()
    
    # create the generators
    print("\nCreating the Generators...")
    train_generator, validation_generator = create_generators(args)
    print("Done!")
    
    num_rotation_parameters = train_generator.get_num_rotation_parameters()
    num_classes = train_generator.num_classes()
    num_anchors = train_generator.num_anchors

    # num_rotation_parameters = 3
    # num_classes = 8
    # num_anchors = 9

    # optionally choose specific GPU
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    allow_gpu_growth_memory()
    print("\nBuilding the Model...")
    # mirrored_strategy = tf.distribute.MirroredStrategy()
    # with mirrored_strategy.scope():
    model, prediction_model = build_Model(args.phi,
                                            num_classes = num_classes,
                                            num_anchors = num_anchors,
                                            freeze_bn = not args.no_freeze_bn,
                                            score_threshold = args.score_threshold,
                                            num_rotation_parameters = num_rotation_parameters,
                                            print_architecture = True,
                                            backbone = args.backbone,
                                            freeze_backbone= args.freeze_backbone)
    print("Done!")
    # load pretrained weights
    if args.weights:
        print('Loading the Model ...')
        if args.weights == 'imagenet' and args.backbone == 'efficientnet':
            model_name = 'efficientnet-b{}'.format(args.phi)
            file_name = '{}_weights_tf_dim_ordering_tf_kernels_autoaugment_notop.h5'.format(model_name)
            file_hash = WEIGHTS_HASHES[model_name][1]
            weights_path = keras.utils.get_file(file_name,
                                                BASE_WEIGHTS_PATH + file_name,
                                                cache_subdir='models',
                                                file_hash=file_hash)
            model.load_weights(weights_path, by_name=True)
            # for i in range(1, [227, 329, 329, 374, 464, 566, 656][args.phi]):
            #     model.layers[i].trainable = False
        else:
            
            model.load_weights(args.weights, by_name=True)

    # compile model
    model.compile(optimizer=Adam(learning_rate = args.lr, clipnorm = 0.001), 
                  loss={'regression': smooth_l1(),
                        'classification': focal(),
                        'transformation': transformation_loss(model_3d_points_np = train_generator.get_all_3d_model_points_array_for_loss(),
                                                              num_rotation_parameter = num_rotation_parameters)},
                  loss_weights = {'regression' : 1.0,
                                  'classification': 1.0,
                                  'transformation': 0.02})

    # create the callbacks
    callbacks = create_callbacks(
        model,
        prediction_model,
        validation_generator,
        args,
    )

    # if not args.compute_val_loss:
    #     validation_generator = None
    # elif args.compute_val_loss and validation_generator is None:
    #     raise ValueError('When you have no validation data, you should not specify --compute-val-loss.')

    INIT_LR = args.lr
    EPOCHS = args.epochs

    def cosine_decay(epoch):
        initial_lrate = INIT_LR
        lrate = 0.5 * initial_lrate * (1 + math.cos(epoch*math.pi/EPOCHS))
        return lrate

    
    # Retrieve Loss History
    history_filename = os.path.join(args.snapshot_path, "train_fit_{backbone}_history.json".format(backbone=args.backbone))
    overwriting = os.path.exists(history_filename) and args.weights is None
    # assert not overwriting, "Weights found in model_dir, aborting to avoid overwrite"
    loss_history = LossHistory(history_filename)
    init_epoch = loss_history.get_initial_epoch()
    if init_epoch > 0 and not args.weights:
        savedCheckpoint = get_checkpoint(validation_generator, args)
        print('Loading the Checkpoint ...')
        model.load_weights(savedCheckpoint, by_name=True)

    callbacks.append(loss_history)

    # lrate = LearningRateScheduler(cosine_decay)
    # callbacks.append(lrate)


    # start training
    return model.fit(
        x = train_generator,
        steps_per_epoch = args.steps,
        initial_epoch = init_epoch,
        epochs = args.epochs,
        verbose = 1,
        callbacks = callbacks,
        workers = args.workers,
        use_multiprocessing = args.multiprocessing,
        max_queue_size = args.max_queue_size,
        validation_data = validation_generator,
        validation_steps = args.steps,
        validation_freq = 2
    )

def get_checkpoint(validation_generator, args):
    
    if args.dataset_type == 'linemod':
        if validation_generator.is_symmetric_object(args.object_id):
            metric_to_monitor = "ADD-S"
        else:
            metric_to_monitor = "ADD"
    else:
        metric_to_monitor = "ADD(-S)" 
    checkpint_path = os.path.join(args.snapshot_path, args.backbone, args.dataset_type, 'phi_{phi}_{dataset_type}_best_{metric}.h5'.format(phi = str(args.phi), metric = metric_to_monitor, dataset_type = args.dataset_type))
    return checkpint_path

def allow_gpu_growth_memory():
    """
        Set allow growth GPU memory to true

    """
    
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.set_logical_device_configuration(gpu,
                    [tf.config.LogicalDeviceConfiguration(memory_limit=4096 * 4)]
                )
                # tf.config.experimental.set_memory_growth(gpu, True)
            print(len(gpus), "Physical GPUs,", len(tf.config.list_logical_devices('GPU')), "Logical GPUs")
        except RuntimeError as e:
            print(e)

def create_callbacks(training_model, prediction_model, validation_generator, args):
    """
    Creates the callbacks to use during training.

    Args:
        training_model: The model that is used for training.
        prediction_model: The model that should be used for validation.
        validation_generator: The generator for creating validation data.
        args: parseargs args object.

    Returns:
        A list of callbacks used for training.
    """
    callbacks = []

    tensorboard_callback = None
    snapshot_path = os.path.join(args.snapshot_path, args.backbone)
    tensorboard_dir = os.path.join(args.snapshot_path, "logs")
    if args.dataset_type == "linemod":
        snapshot_path = os.path.join(snapshot_path, "object_" + str(args.object_id))
        if args.validation_image_save_path:
            save_path = os.path.join(args.validation_image_save_path, "object_" + str(args.object_id))
        else:
            save_path = args.validation_image_save_path
        if args.save_tensorboard:
            tensorboard_dir = os.path.join(tensorboard_dir, "object_" + str(args.object_id))
            
        if validation_generator.is_symmetric_object(args.object_id):
            metric_to_monitor = "ADD-S"
            mode = "max"
        else:
            metric_to_monitor = "ADD"
            mode = "max"
    elif args.dataset_type == "occlusion":
        snapshot_path = os.path.join(snapshot_path, "occlusion")
        if args.validation_image_save_path:
            save_path = os.path.join(args.validation_image_save_path, args.backbone, "occlusion")
        else:
            save_path = args.validation_image_save_path
        if args.save_tensorboard:
            tensorboard_dir = os.path.join(tensorboard_dir, "occlusion")
            
        metric_to_monitor = "ADD(-S)"
        mode = "max"
    elif args.dataset_type == "ycbv":
        snapshot_path = os.path.join(snapshot_path, "object_" + str(args.object_id))
        if args.validation_image_save_path:
            save_path = os.path.join(args.validation_image_save_path, "object_" + str(args.object_id))
        else:
            save_path = args.validation_image_save_path
        if args.save_tensorboard:
            tensorboard_dir = os.path.join(tensorboard_dir, "object_" + str(args.object_id))
            
        if validation_generator.is_symmetric_object(args.object_id):
            metric_to_monitor = "ADD-S"
            mode = "max"
        else:
            metric_to_monitor = "ADD"
            mode = "max"
        
    if args.validation_image_save_path:
        os.makedirs(save_path, exist_ok = True)
    
    tensorboard_callback = keras.callbacks.TensorBoard(
        log_dir = tensorboard_dir,
        histogram_freq = 2,
        write_graph = True,
        write_grads = False,
        write_images = False,
        embeddings_freq = 0,
        embeddings_layer_names = None,
        embeddings_metadata = None
    )
    callbacks.append(tensorboard_callback)

    if args.evaluation and validation_generator:
        from eval.eval_callback import Evaluate
        evaluation = Evaluate(validation_generator, prediction_model, tensorboard = tensorboard_callback, save_path = None)
        callbacks.append(evaluation)

    # save the model
    if args.snapshots:
        # ensure directory created first; otherwise h5py will error after epoch.
        os.makedirs(snapshot_path, exist_ok = True)
        best_checkpoint = keras.callbacks.ModelCheckpoint(os.path.join(snapshot_path, 'phi_{phi}_{dataset_type}_best_{metric}.h5'.format(phi = str(args.phi), metric = metric_to_monitor, dataset_type = args.dataset_type)),
                                                     verbose = 1,
                                                     save_weights_only = True,
                                                     save_best_only = True,
                                                     monitor = metric_to_monitor,
                                                     mode = mode)
        callbacks.append(best_checkpoint)

        # last_checkpoint = ModelCheckpoint(os.path.join(snapshot_path, 'phi_{phi}_{dataset_type}_last_{metric}.h5'.format(phi = str(args.phi), metric = metric_to_monitor, dataset_type = args.dataset_type)),
        #                           monitor = metric_to_monitor,
        #                           save_freq = 'epoch',
        #                           save_weights_only = True,
        #                           verbose = 1, 
        #                           mode = mode)
        # callbacks.append(last_checkpoint)

    callbacks.append(keras.callbacks.ReduceLROnPlateau(
        monitor    = 'MixedAveragePointDistanceMean_in_mm',
        factor     = 0.5,
        patience   = 25,
        verbose    = 1,
        mode       = 'min',
        min_delta  = 0.0001,
        cooldown   = 0,
        min_lr     = 1e-7
    ))

    return callbacks


def create_generators(args):
    """
    Create generators for training and validation.

    Args:
        args: parseargs object containing configuration for generators.
    Returns:
        The training and validation generators.
    """
    common_args = {
        'batch_size': args.batch_size,
        'phi': args.phi,
    }

    if args.dataset_type == 'linemod':
        from generators.linemod import LineModGenerator
        train_generator = LineModGenerator(
            args.linemod_path,
            args.object_id,
            rotation_representation = args.rotation_representation,
            use_colorspace_augmentation = not args.no_color_augmentation,
            use_6DoF_augmentation = not args.no_6dof_augmentation,
            **common_args
        )

        validation_generator = LineModGenerator(
            args.linemod_path,
            args.object_id,
            train = False,
            shuffle_dataset = False,
            shuffle_groups = False,
            rotation_representation = args.rotation_representation,
            use_colorspace_augmentation = False,
            use_6DoF_augmentation = False,
            **common_args
        )
    elif args.dataset_type == 'occlusion':
        from generators.occlusion import OcclusionGenerator
        train_generator = OcclusionGenerator(
            args.occlusion_path,
            rotation_representation = args.rotation_representation,
            use_colorspace_augmentation = not args.no_color_augmentation,
            use_6DoF_augmentation = not args.no_6dof_augmentation,
            **common_args
        )

        validation_generator = OcclusionGenerator(
            args.occlusion_path,
            train = False,
            shuffle_dataset = False,
            shuffle_groups = False,
            rotation_representation = args.rotation_representation,
            use_colorspace_augmentation = False,
            use_6DoF_augmentation = False,
            **common_args
        )
    elif args.dataset_type == 'ycbv':
        from generators.ycbv import YCBVGenerator
        train_generator = YCBVGenerator(
            args.ycbv_path,
            args.object_id,
            rotation_representation = args.rotation_representation,
            use_colorspace_augmentation = not args.no_color_augmentation,
            use_6DoF_augmentation = not args.no_6dof_augmentation,
            **common_args
        )

        validation_generator = YCBVGenerator(
            args.ycbv_path,
            args.object_id,
            train = False,
            shuffle_dataset = False,
            shuffle_groups = False,
            rotation_representation = args.rotation_representation,
            use_colorspace_augmentation = False,
            use_6DoF_augmentation = False,
            **common_args
        )
        
    else:
        raise ValueError('Invalid data type received: {}'.format(args.dataset_type))

    return train_generator, validation_generator


if __name__ == '__main__':
    main()
