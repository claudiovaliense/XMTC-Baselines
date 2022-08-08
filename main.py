import hydra
import os
from omegaconf import OmegaConf

from source.helper.EvalHelper import EvalHelper
from source.helper.FitHelper import FitHelper
from source.helper.PredictHelper import PredictHelper
from source.helper.PreprocessHelper import PreprocessHelper


def preprocess(params):
    preprocess_helper = PreprocessHelper(params)
    preprocess_helper.perform_preprocess()

def fit(params):
    fit_helper = FitHelper(params)
    fit_helper.perform_fit()

def predict(params):
    predict_helper = PredictHelper(params)
    predict_helper.perform_predict()

def eval(params):
    eval_helper = EvalHelper(params)
    eval_helper.perform_eval()


@hydra.main(config_path="settings", config_name="settings.yaml", version_base=None)
def perform_tasks(params):
    os.chdir(hydra.utils.get_original_cwd())
    OmegaConf.resolve(params)

    if "preprocess" in params.tasks:
        preprocess(params)

    if "fit" in params.tasks:
        fit(params)

    if "predict" in params.tasks:
        predict(params)

    if "eval" in params.tasks:
        eval(params)

if __name__ == '__main__':
    perform_tasks()