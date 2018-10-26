import imp
from ruamel import yaml
# import yaml
import os
import numpy as np
import argparse
from easydict import EasyDict as edict
import configparser

class Configuration():
    def __init__(self):
        self.Parse_args()
        self.exp_folder = os.path.join(self.args.exp_folder, self.args.exp_name)
        if not os.path.exists(self.exp_folder):
            os.makedirs(self.exp_folder)

    def Load(self):
        if self.args.config_file is not None:
            # Read a user specific config File
            # cf = imp.load_source('config', self.args.config_file)
            # cf = configparser.ConfigParser()
            # cf.read(self.args.config_file)
            # print(cf)
            # cf = edict(cf)
            with open(self.args.config_file, 'r') as f:
                cf = yaml.load(f, Loader=yaml.Loader)
                # print(cf.crop_train)
                # print(type(cf.crop_train))
                # print(len(cf.crop_train))
                cf = edict(cf)
        else:
            # Read the deafault config file
            # cf = imp.load_source('config', 'config/configFile.py')
            with open('config/configFile.yml', 'r') as f:
                cf = edict(yaml.load(f))
        cf = self.Parser_to_config(cf)
        cf.exp_folder = os.path.join(cf.exp_folder, cf.exp_name)
        cf.tensorboard_path = os.path.join(cf.exp_folder,'tensorboard/')
        cf.debug = self.args.debug
        cf.log_file = os.path.join(cf.exp_folder, "logfile.log")
        cf.log_file_stats = os.path.join(cf.exp_folder, "logfile_stats.log")
        cf.log_file_debug = os.path.join(cf.exp_folder, "logfile_debug.log")
        if not os.path.exists(os.path.join(cf.exp_folder, 'json_stats/')):
            os.makedirs(os.path.join(cf.exp_folder, 'json_stats/'))
        if not os.path.exists(os.path.join(cf.exp_folder, 'json_stats/train_history/')):
            os.makedirs(os.path.join(cf.exp_folder, 'json_stats/train_history/'))
        cf.train_json_path = os.path.join(cf.exp_folder, "json_stats/train_history/")
        cf.val_json_file = os.path.join(cf.exp_folder, "json_stats/val_stats.json")
        cf.test_json_file = os.path.join(cf.exp_folder, "json_stats/test_stats.json")
        cf.best_json_file = os.path.join(cf.exp_folder, "json_stats/best_model_stats.json")
        # Copy config file TODO: create a file saver for parse config
        # shutil.copyfile(cf.config_file, os.path.join(cf.exp_folder, "config.py"))

        if cf.predict_path_output is None or cf.predict_path_output is None:
            cf.predict_path_output = os.path.join(cf.exp_folder,'predictions/')
            if not os.path.exists(cf.predict_path_output):
                os.makedirs(cf.predict_path_output)
        # cf.original_size = cf.size_image_test
        if cf.input_model_path is None:
            cf.input_model_path = os.path.join(cf.exp_folder, cf.model_name + '.pth')
        if cf.output_model_path is None:
            cf.output_model_path = cf.exp_folder
        else:
            if not os.path.exists(cf.output_model_path):
                os.makedirs(cf.output_model_path)
        if cf.map_labels is not None:
            cf.map_labels = {value: idx for idx, value in enumerate(cf.map_labels)}
        # if cf.pretrained_model is None:
        #     cf.pretrained_model = 'None'
        if not cf.pretrained_model.lower() in ('none', 'basic', 'custom'):
            raise ValueError('Unknown pretrained_model definition')
        if cf.pretrained_model == 'basic':
            cf.basic_pretrained_model = True
        else:
            cf.basic_pretrained_model = False
        if cf.basic_models_path is None:
            cf.basic_models_path = './pretrained_model/'
        return cf

    def Parse_args(self):
        # Input arguments
        parser = argparse.ArgumentParser(description="Pytorch framework for Semantic segmentation, classification "
                                                     "and detection")
        parser.add_argument("--config_file",
                            type=str,
                            default=None,
                            help="configuration YALM file path")

        parser.add_argument("--exp_name",
                            type=str,
                            default='Sample',
                            help="Experiment name")

        parser.add_argument("--exp_folder",
                            type=str,
                            default='/home/jlgomez/Experiments/',
                            help="Experiment folder path")

        parser.add_argument("--problem_type",
                            type=str,
                            help="Type of problem, Options: ['segmentation','classification']")

        parser.add_argument("--debug",
                            dest='debug',
                            action='store_true',
                            help="experiment mode")

        # Model
        parser.add_argument("--model", dest='model_type',
                            type=str,
                            help="Model name, Options: ['DenseNetFCN', 'FCN8']")

        ### load options
        parser.add_argument("--resume_experiment",
                            type=bool,
                            help="Restore the best model obtained in the experiment defined if exist")
        parser.add_argument("--pretrained_model",
                            type=str,
                            help="Pretarined Model, Options: 'None': from scratch, 'basic': pretraned from imagenet, "
                                 "'custom': personal model")
        parser.add_argument("--input_model_path",
                            type=str,
                            help="Path and pretrained file to load [None uses experiment path and model name by "
                                 "default")
        parser.add_argument("--load_weight_only",
                            type=bool,
                            help="True: loads only weights and parameters [recommended], False: loads all the network")
        parser.add_argument("--basic_models_path",
                            type=str,
                            help="Path to download and store the basic models (ImageNet weights)")
        ### Save options
        parser.add_argument("--save_weight_only",
                            type=bool,
                            help="True: stores only weights and parameters, False: store all the network structure and "
                                 "weights")
        parser.add_argument("--model_name",
                            type=str,
                            help="Base name of the model file to save")
        parser.add_argument("--output_model_path",
                            type=str,
                            help="Path to store the model using model_name [None uses the default experiment path]")

        # Loss type
        parser.add_argument("--loss_type",
                            type=str,
                            help="Loss function, options: ['cross_entropy_segmentation','focal_segmentation']")
        # General parameters
        parser.add_argument("--train_samples",
                            type=int,
                            help="Number of samples to train, -1 uses all the samples available inside the dataset "
                                 "files")
        parser.add_argument("--valid_samples",
                            type=int,
                            help="Number of samples to validate, -1 uses all the samples available inside the dataset "
                                 "files")
        parser.add_argument("--test_samples",
                            type=int,
                            help="Number of samples to test, -1 uses all the samples available inside the dataset "
                                 "files")
        parser.add_argument("--train_batch_size",
                            type=int,
                            help="Train batch size"
                                 "files")
        parser.add_argument("--valid_batch_size",
                            type=int,
                            help="Validation batch size")
        parser.add_argument("--test_batch_size",
                            type=int,
                            help="Test batch size")
        parser.add_argument("--train",
                            type=bool,
                            help="Activate/Deactivate train step")
        parser.add_argument("--validation",
                            type=bool,
                            help="Activate/Deactivate validation step")
        parser.add_argument("--test",
                            type=bool,
                            help="Activate/Deactivate test step")
        parser.add_argument("--predict_test",
                            type=bool,
                            help="Generate predictions from test, doesn't need gt")
        parser.add_argument("--predict_path_output",
                            type=str,
                            help="Path to store the predictions. 'None' uses the default output in the experiment "
                                 "folder /predictions")

        # Image properties
        parser.add_argument('--size_image_train', nargs='+', type=int, help='Global train dataset image size: '
                                                                  'e.g. --size_image_train X Y')
        parser.add_argument('--size_image_valid', nargs='+', type=int, help='Global validation dataset image size: '
                                                                  'e.g. --size_image_valid X Y')
        parser.add_argument('--size_image_test', nargs='+', type=int, help='Global test dataset image size: '
                                                                  'e.g. --size_image_test X Y')
        parser.add_argument('--resize_image_train', nargs='+', type=int, help='Resize train image to size: '
                                                                  'e.g. --resize_image_train X Y')
        parser.add_argument('--resize_image_valid', nargs='+', type=int, help='Resize validation image to size: '
                                                                  'e.g. --resize_image_valid X Y')
        parser.add_argument('--resize_image_test', nargs='+', type=int, help='Resize test image to size: '
                                                                  'e.g. --resize_image_test X Y')
        parser.add_argument('--crop_train', nargs='+', type=int, help='Crop size for training: '
                                                                  'e.g. --crop_train X Y')
        parser.add_argument("--grayscale",
                            type=bool,
                            help="True: If dataset is in grayscale")

        # Dataset properties
        parser.add_argument("--dataset",
                            type=str,
                            help="Dataset name to read bounding boxes (only for detection)")
        parser.add_argument("--train_images_txt",
                            type=str,
                            help="File that contains the train images")
        parser.add_argument("--train_gt_txt",
                            type=str,
                            help="File that contains the ground truth from train images")
        parser.add_argument("--valid_images_txt",
                            type=str,
                            help="File that contains the validation images")
        parser.add_argument("--valid_gt_txt",
                            type=str,
                            help="File that contains the ground truth from validation images")
        parser.add_argument("--test_images_txt",
                            type=str,
                            help="File that contains the test images")
        parser.add_argument("--test_gt_txt",
                            type=str,
                            help="File that contains the ground truth from test images")

        parser.add_argument('--labels', nargs='+', type=str, help='Specify list of class labels: e.g. '
                                                                  '--labels class1 class2 class3 ...'
                                                                  'e.g. --crop_train X Y')
        parser.add_argument('--map_labels', nargs='+', type=str, help='Specify list of class labels: e.g. '
                                                                  '--labels class1 class2 class3 ...'
                                                                  'e.g. --crop_train X Y')
        parser.add_argument("--num_classes",
                            type=int,
                            help="Number of classes")
        parser.add_argument("--shuffle",
                            type=bool,
                            help="Shuffle train data in the training step")
        parser.add_argument("--void_class",
                            type=int,
                            help="Void class id or value")

        # Training
        parser.add_argument("--epochs",
                            type=int,
                            help="Max number of epochs, use 0 to save directly a model, useful to make conversions")
        parser.add_argument("--initial_epoch",
                            type=int,
                            help="Defines the starting epoch number")
        parser.add_argument("--valid_samples_epoch",
                            type=int,
                            help="Number of validation images used to validate an epoch")

        ### Optimizer ###
        parser.add_argument("--optimizer",
                            type=str,
                            help="Optimizer approach, Options available ['SGD','Adam','RMSProp']")
        parser.add_argument("--momentum",dest='momentum1',
                            type=float,
                            help="Momentum or first momentum for the optimizer")
        parser.add_argument("--momentum2",
                            type=float,
                            help="Second momentum for Adam")
        parser.add_argument("--learning_rate",
                            type=float,
                            help="Learning rate")
        parser.add_argument("--learning_rate_bias",
                            type=float,
                            help="Learning rate for the bias")
        parser.add_argument("--weight_decay",
                            type=float,
                            help="Weight decay")
        ### Scheduler
        parser.add_argument("--scheduler",
                            type=str,
                            help="Training scheduler, Options available "
                                 "['ReduceLROnPlateau','Step','MultiStep','Exponential', None]")
        parser.add_argument("--decay",
                            type=float,
                            help="Learnng rate decay to apply (lr*decay)")
        parser.add_argument("--sched_patience",
                            type=int,
                            help="ReduceLROnPlateau option: epoch patience without loss change until a lr decrement")
        parser.add_argument("--step_size",
                            type=int,
                            help="Step option: epoch counter to decrease lr")
        parser.add_argument("--milestone", nargs='+',
                            type=int,
                            help="MultiStep option: define different milestones (epochs) to decrease lr: "
                                 "e.g --milestone 50 30 10")
        ### Save criteria
        parser.add_argument("--save_condition",
                            type=str,
                            help="Reference metric to save the model: Options "
                                 "['always','(x)_loss','(x)_mAcc','(x)_mIoU'] x = valid or train_loss")
        ### Early Stopping
        parser.add_argument("--early_stopping",
                            type=bool,
                            help="Enable early stopping when no imporvement is detected")
        parser.add_argument("--stop_condition",
                            type=str,
                            help="Reference metric to stop the training: Options "
                                 "['always','(x)_loss','(x)_mAcc','(x)_mIoU'] x = valid or train_loss")
        parser.add_argument("--patience",
                            type=int,
                            help="Number of epochs without improvement to apply early stopping")

        # Image preprocess
        parser.add_argument("--rescale",
                            type=bool,
                            help="Divide images values to 255 to have values between 0 and 1 ")
        parser.add_argument("--mean", nargs='+',
                            type=float,
                            help="List with the mean values of the traing data"
                                 "e.g --mean 0.5 0.5 0.5")
        parser.add_argument("--std", nargs='+',
                            type=float,
                            help="List with the std values of the traing data"
                                 "e.g --std 0.5 0.5 0.5")

        # Data augmentation
        parser.add_argument("--hflips",
                            type=bool,
                            help="Horitzontal flips")

        self.args = parser.parse_args()

    def Parser_to_config(self, cf):
        cf.config_path = self.args.config_file
        cf.exp_name = self.args.exp_name
        cf.exp_folder = self.args.exp_folder
        if self.args.problem_type is not None:
            cf.problem_type = self.args.problem_type
            # Model
        if self.args.model_type is not None:
            cf.model_type = self.args.model_type
            ### load options
        if self.args.resume_experiment is not None:
            cf.resume_experiment = self.args.resume_experiment
        if self.args.pretrained_model is not None:
            cf.pretrained_model = self.args.pretrained_model
        if self.args.input_model_path is not None:
            cf.input_model_path = self.args.input_model_path
        if self.args.load_weight_only is not None:
            cf.load_weight_only = self.args.load_weight_only
        if self.args.basic_models_path is not None:
            cf.basic_models_path = self.args.basic_models_path
            ### Save options
        if self.args.save_weight_only is not None:
            cf.save_weight_only = self.args.save_weight_only
        if self.args.model_name is not None:
            cf.model_name = self.args.model_name
        if self.args.output_model_path is not None:
            cf.output_model_path = self.args.output_model_path
            # Loss type
        if self.args.loss_type is not None:
            cf.loss_type = self.args.loss_type
            # General parameters
        if self.args.train_samples is not None:
            cf.train_samples = self.args.train_samples
        if self.args.valid_samples is not None:
            cf.valid_samples = self.args.valid_samples
        if self.args.test_samples is not None:
            cf.test_samples = self.args.test_samples
        if self.args.train_batch_size is not None:
            cf.train_batch_size = self.args.train_batch_size
        if self.args.valid_batch_size is not None:
            cf.valid_batch_size = self.args.valid_batch_size
        if self.args.test_batch_size is not None:
            cf.test_batch_size = self.args.test_batch_size
        if self.args.train is not None:
            cf.train = self.args.train
        if self.args.validation is not None:
            cf.validation = self.args.validation
        if self.args.test is not None:
            cf.test = self.args.test
        if self.args.predict_test is not None:
            cf.predict_test = self.args.predict_test
        if self.args.predict_path_output is not None:
            cf.predict_path_output = self.args.predict_path_output
            # Image properties
        if self.args.size_image_train is not None:
            cf.size_image_train = self.args.size_image_train
        if self.args.size_image_valid is not None:
            cf.size_image_valid = self.args.size_image_valid
        if self.args.size_image_test is not None:
            cf.size_image_test = self.args.size_image_test
        if self.args.resize_image_train is not None:
            cf.resize_image_train = self.args.resize_image_train
        if self.args.resize_image_valid is not None:
            cf.resize_image_valid = self.args.resize_image_valid
        if self.args.resize_image_test is not None:
            cf.resize_image_test = self.args.resize_image_test
        if self.args.crop_train is not None:
            cf.crop_train = self.args.crop_train
        if self.args.grayscale is not None:
            cf.grayscale = self.args.grayscale
            # Dataset properties
        if self.args.dataset is not None:
            cf.dataset = self.args.dataset
        if self.args.train_images_txt is not None:
            cf.train_images_txt = self.args.train_images_txt
        if self.args.train_gt_txt is not None:
            cf.train_gt_txt = self.args.train_gt_txt
        if self.args.valid_images_txt is not None:
            cf.valid_images_txt = self.args.valid_images_txt
        if self.args.valid_gt_txt is not None:
            cf.valid_gt_txt = self.args.valid_gt_txt
        if self.args.test_images_txt is not None:
            cf.test_images_txt = self.args.test_images_txt
        if self.args.test_gt_txt is not None:
            cf.test_gt_txt = self.args.test_gt_txt
        if self.args.labels is not None:
            self.args.labels=self.args.labels[0].split(',')
            cf.labels = self.args.labels
        if self.args.map_labels is not None:
            self.args.map_labels=self.args.map_labels[0].split(',')
            cf.map_labels = self.args.map_labels
        if self.args.num_classes is not None:
            cf.num_classes = self.args.num_classes
        if self.args.shuffle is not None:
            cf.shuffle = self.args.shuffle
        if self.args.void_class is not None:
            cf.void_class = self.args.void_class
           # Training
        if self.args.epochs is not None:
            cf.epochs = self.args.epochs
        if self.args.initial_epoch is not None:
            cf.initial_epoch = self.args.initial_epoch
        if self.args.valid_samples_epoch is not None:
            cf.valid_samples_epoch = self.args.valid_samples_epoch
            ### Optimizer ###
        if self.args.optimizer is not None:
            cf.optimizer = self.args.optimizer
        if self.args.momentum1 is not None:
            cf.momentum1 = self.args.momentum1
        if self.args.momentum2 is not None:
            cf.momentum2 = self.args.momentum2
        if self.args.learning_rate is not None:
            cf.learning_rate = self.args.learning_rate
        if self.args.learning_rate_bias is not None:
            cf.learning_rate_bias = self.args.learning_rate_bias
        if self.args.weight_decay is not None:
            cf.weight_decay = self.args.weight_decay
            ### Scheduler
        if self.args.scheduler is not None:
            cf.scheduler = self.args.scheduler
        if self.args.decay is not None:
            cf.decay = self.args.decay
        if self.args.sched_patience is not None:
            cf.sched_patience = self.args.sched_patience
        if self.args.step_size is not None:
            cf.step_size = self.args.step_size
        if self.args.model_name is not None:
            cf.milestone = self.args.milestone
            ### Save criteria
        if self.args.save_condition is not None:
            cf.save_condition = self.args.save_condition
            ### Early Stopping
        if self.args.early_stopping is not None:
            cf.early_stopping = self.args.early_stopping
        if self.args.stop_condition is not None:
            cf.stop_condition = self.args.stop_condition
        if self.args.patience is not None:
            cf.patience = self.args.patience
            # Image preprocess
        if self.args.rescale is not None and self.args.rescale == True:
            cf.rescale = 1 / 255.
        if self.args.mean is not None:
            cf.mean = self.args.mean
        if self.args.std is not None:
            cf.std = self.args.std
            # Data augmentation
        if self.args.hflips is not None:
            cf.hflips = self.args.hflips
        return cf