import gin
import logging
from absl import app, flags
from train import Trainer
from input_pipeline import datasets
from utils import utils_params, utils_misc
from models import lstm_model
from input_pipeline.preprocessing import preprocessor
from eval import evaluate

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
    run_paths = utils_params.gen_run_folder()
    print(run_paths)

    # set loggers
    utils_misc.set_loggers(run_paths['path_logs_train'], logging.INFO)

    # gin-config
    gin.parse_config_files_and_bindings(['configs/config.gin'], [])
    utils_params.save_config(run_paths['path_gin'], gin.config_str())

    preprocessor()

    # setup pipeline
    ds_train, ds_val, ds_test = datasets.load()

    # model
    batch_size = gin.query_parameter('prepare.batch_size')
    model = lstm_model(input_shape=(250, 6), num_classes=12,batch_size=batch_size )



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
