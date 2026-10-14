from dataclasses import dataclass, field
from typing import List
import argparse
import toml
import copy

@dataclass
class DataConfig:
    """ Contains the dataset configuration:
        - data paths
        - file names
        - transforms
    """

    out_path: str = '/path/to/output/directory'

    data_path: str = "path/to/high_resolution_data"
    data_path_lowres: str = "path/to/low_resolution_data"
    data_path_test: str = "path/to/test_data_directory" # This should contain directories with low and high resolution images
    results_path: str = out_path + '/results'
    tensorboard_path: str = out_path + '/tb_logger'
    checkpoint_path: str = out_path + '/checkpoint'
    config_path: str = out_path + '/config-files'

    index_csv: str = "path/to/indexed_files.csv" # Path to CSV file containing indices for data selection

    sample_dimension: int = field(default=(None, None))
    
    transforms: List = field(default_factory= lambda: ['log', 'normalize_minus1_to_plus1'])

    epsilon: float = 0.0001

    predict_variable: str = None

    n_workers: int = 10

    use_float16: bool = False

    pad_input: int = field(default=(0, 0, 0, 0))

    lazy: bool = False


@dataclass
class TrainingConfig:
    """ Contains the training configuration:
        - data splits
        - hyperparameters: learning rate, batch size, number of epochs
    """

    n_epochs: int = 200

    batch_size: int  = 6

    lr: float = 4e-5

    grad_clip_norm: float = 1

    warmup: float = 0

    check_val_every_n_epoch: int = 1

    use_regularization: bool = False


@dataclass
class DiffusionConfig:
    """ Contains the diffusion configuration:
        - UNet architechture: number of channels, up and down blocks
        - variance schedule parameters
        - eponential moving average (EMA) decay
    """

    name: str ='checkpoint_name'

    which_model_G: str = "wave"
    finetune_norm: bool = False
    unet: dict = field(default_factory = lambda:{
        "in_channel": 32,
        "out_channel": 16,
        "inner_channel": 128,
        "norm_groups": 16,
        "channel_multiplier": [1, 2, 4, 4, 8, 8, 16],
        "attn_res": [16],
        "res_blocks": 3,
        "dropout": 0,
        "beta_schedule": {
            "train": {
            "schedule": "linear",
            "n_timestep": 2000,
            "linear_start": 1e-6,
            "linear_end": 1e-2
            },
            "val": {
            "schedule": "linear",
            "n_timestep": 2000,
            "linear_start": 1e-6,
            "linear_end": 1e-2
            }
        },
        "diffusion": {
            "image_size": 192,
            "channels": 4,
            "conditional": True
        }
    })

    diffusion_model: str = 'consistency' # 've' or 'consistency'
    diffusion_checkpoint_path: str = ''
    consistency_conversion_output_dir: str = ''

    resume_ckpt_path: str = ''

    sigma_max: float = 1000
    sigma_min: float = 1e-2

    data_std: float = 0.5
    time_min: float = 0.002
    time_max: float = 80
    clip_output: bool = False

    ema_rate: float = 0.999
    use_ema: bool = False
    
@dataclass
class Config(DataConfig, TrainingConfig, DiffusionConfig):
    """ Wrapper class that combines the configuration. """
    None


def parse_command_line():
    """ Parses the command line options and overwrites the default configuration. """

    parser = argparse.ArgumentParser()

    parser.add_argument("-rw", "--n_worker",
                        help="Number of worker processes", type=int)

    parser.add_argument("-tf", "--target_filename",
                        help="Filename of the target .nc file", type=str)

    parser.add_argument("-ef", "--esm_filename",
                        help="Filename of the ESM .nc file", type=str)

    parser.add_argument("-ts", "--trainig_start",
                        help="Start year of training data", type=int)

    parser.add_argument("-te", "--trainig_end",
                        help="End year of training data", type=int)

    parser.add_argument("-vs", "--valid_start",
                        help="Start year of valid data", type=int)

    parser.add_argument("-ve", "--valid_end",
                        help="End year of validation data", type=int)

    parser.add_argument("-clat", "--crop_data_latitude", nargs='+',
                        help="List of indeces to crop field in latitude dimension", type=int)

    parser.add_argument("-clon", "--crop_data_longitude", nargs='+',
                        help="List of indeces to crop field in longitude dimension", type=int)

    parser.add_argument("-fp", "--use_float_16",
                        help="Convert data to float 16 precision", action='store_true')

    parser.add_argument("-n", "--name",
                        help="The name of the model", type=str)

    parser.add_argument("-dm", "--diffusion_model",
                        help="The name of the model", type=str)

    parser.add_argument("-ep", "--n_epochs",
                        help="Number of training epochs", type=int)

    parser.add_argument("-bs", "--batch_size",
                        help="Training batch size", type=int)

    parser.add_argument("-nr", "--network_resolution",
                        help="Neural network resolution", type=int)

    parser.add_argument("-c", "--channels", nargs='+',
                        help="List of network channels", type=int)

    parser.add_argument("-dbt", "--down_block_types", nargs='+',
                        help="List of network down sampling blocks" , type=str)

    parser.add_argument("-ubt", "--up_block_types", nargs='+',
                        help="List of network up sampling blocks" , type=str)

    parser.add_argument("-sgmin", "--sigma_min",
                        help="Minimum sigma value for stochastic process", type=float)

    parser.add_argument("-sgmax", "--sigma_max",
                        help="Maximum sigma value for stochastic process", type=float)

    parser.add_argument("-wu", "--warmup",
                        help="Number of warmup steps", type=float)

    parser.add_argument("-std", "--standardize",
                        help="Standardize data to zero mean and standard deviation of 1", action='store_true')

    parser.add_argument("-norm", "--normalize",
                        help="Normalize data to [-1, 1]", action='store_true')

    parser.add_argument("-ema", "--use_ema",
                        help="Use EMA model.", action='store_true')

    parser.add_argument("-cp", "--checkpoint_path",
                        help="Path to the model checkpoint .ckpt file", type=str)

    parser.add_argument("-of", "--output_filename",
                        help="Filename of the model output samples file", type=str)

    parser.add_argument("-ns", "--num_sde_steps",
                        help="Number of sde integration steps", type=int)

    parser.add_argument("-nb", "--num_batches",
                        help="Number of batches to sample", type=int)


    args = parser.parse_args()

    config = Config()
   
    if args.normalize and 'normalize_minus1_to_plus1' not in config.transforms:
        config.transforms.append('normalize_minus1_to_plus1')

    for arg in vars(args):
        value = getattr(args, arg)
        if value is not None:
            if type(value) == list:
                value = tuple(value)
            setattr(config, arg, value)

    return config
