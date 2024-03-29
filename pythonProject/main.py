import gin
import logging
from absl import app, flags
from train import Trainer
from input_pipeline_s2l import datasets
from utils import utils_params, utils_misc
from pythonProject.architectures.models_rcnn import create_crnn_model
from input_pipeline_s2l.preprocessing import preprocessor
from eval import evaluate

FLAGS = flags.FLAGS
flags.DEFINE_boolean('train', True, 'Specify whether to train or evaluate a model.')

import wandb
api_key = "31bd8c2f1d2200322f4dd81550c6bc45928ba2d0"
wandb.login(key=api_key)
project_name = "human_activity_recognition"
wandb.init(project=project_name)
print('jj')
run_paths = None

def main(argv):

    # generate folder structures
    global run_paths
    run_paths = utils_params.gen_run_folder('rcnn')
    print(run_paths)

    # set loggers
    utils_misc.set_loggers(run_paths['path_logs_train'], logging.INFO)

    # gin-config
    gin.parse_config_files_and_bindings(['configs/config.gin'], [])
    utils_params.save_config(run_paths['path_gin'], gin.config_str())

    _ = preprocessor()

    # setup pipeline
    ds_train, ds_val, ds_test = datasets.load()

    # model
    batch_size = gin.query_parameter('prepare.batch_size')
    model = create_crnn_model((250,6),12)
    #model = lstm_model(input_shape=(250, 6), num_classes=12,batch_size=batch_size )
    model.summary()


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
