import gin
import logging
from absl import app, flags
import  tensorflow as tf
from train import Trainer
from input_pipeline import datasets
from input_pipeline import preprocessing
from utils import utils_params, utils_misc
from models import create_crnn_model
from evaluation.eval import evaluate

FLAGS = flags.FLAGS
flags.DEFINE_boolean('train', True, 'Specify whether to train or evaluate a model.')

import wandb
api_key = "31bd8c2f1d2200322f4dd81550c6bc45928ba2d0"
wandb.login(key=api_key)
project_name = "human_activity_recognition"
wandb.init(project=project_name)
print('jj')

def main(argv):

    # generate folder structures
    #run_paths = utils_params.gen_run_folder()
    #print(run_paths)
    run_paths = {'path_model_id': 'D:\\HAR\\experiments\\run_2024-01-28T19-28-59-232815', 'path_logs_train': 'D:\\HAR\\experiments\\run_2024-01-28T19-28-59-232815\\logs\\run.log', 'path_ckpts_train': 'D:\\HAR\\experiments\\run_2024-01-28T19-28-59-232815\\ckpts', 'path_gin': 'D:\\HAR\\experiments\\run_2024-01-28T19-28-59-232815\\config_operative.gin'}

    # set loggers
    utils_misc.set_loggers(run_paths['path_logs_train'], logging.INFO)

    # gin-config
    gin.parse_config_files_and_bindings(['configs/config.gin'], [])
    utils_params.save_config(run_paths['path_gin'], gin.config_str())

    # setup pipeline
    ds_train, ds_val, ds_test = datasets.load()

    # model
    model = create_crnn_model(input_shape=(250, 6), num_classes=12)



    if FLAGS.train:
        trainer = Trainer(model, ds_train, ds_val, run_paths)
        for _ in trainer.train():
            continue
        print('starting eval***************************************')
        evaluate(model,
                 run_paths,
                 ds_test)
    wandb.finish()

if __name__ == "__main__":
    app.run(main)
