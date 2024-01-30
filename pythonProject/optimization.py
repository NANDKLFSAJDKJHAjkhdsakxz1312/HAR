import logging
import wandb
import gin
import math
from train import Trainer
from input_pipeline.datasets import load
from input_pipeline.preprocessing import preprocessor
from models import lstm_model
from utils import utils_params, utils_misc


def train_func():
    with wandb.init() as run:
        gin.clear_config()
        # Hyperparameters
        bindings = []
        for key, value in run.config.items():
            bindings.append(f'{key}={value}')

        # generate folder structures
        run_paths = utils_params.gen_run_folder(run.id)
        print(run_paths)
        # set loggers
        utils_misc.set_loggers(run_paths['path_logs_train'], logging.INFO)

        # gin-config
        gin.parse_config_files_and_bindings(['configs/config.gin'], bindings)
        utils_params.save_config(run_paths['path_gin'], gin.config_str())

        preprocessor()

        # setup pipeline
        ds_train, ds_val, ds_test = load()
        # model
        batch_size = gin.query_parameter('prepare.batch_size')
        model = lstm_model(input_shape=(250, 6), num_classes=12, batch_size=batch_size)

        trainer = Trainer(model, ds_train, ds_val, run_paths)
        for _ in trainer.train():
            continue


sweep_config = {
    'name': 'hapt_tuning',
    'method': 'grid',
    'metric': {
        'name': 'val_acc',
        'goal': 'maximize'
    },
    'parameters': {
        'num_recurrent_layers': {
            'values': [1, 2, 3]
        },
        'num_fc_layers': {
            'values': [1, 2]
        },
        'num_hidden_units': {
            'values': [32, 64]
        },
        'dropout_rate': {
            'values': [0.2, 0.5]
        },
        'learning_rate': {
            'values': [0.001, 0.01]
        },
        'stateful': {
            'values': [True, False]
        }
    }
}
sweep_id = wandb.sweep(sweep_config, project="project2_optimization")

wandb.agent(sweep_id, function=train_func)