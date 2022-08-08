import pickle

import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from pecos.xmc.xtransformer.model import XTransformer
from scipy import sparse

from source.helper.Helper import Helper


class PredictHelper:

    def __init__(self, params):
        self.params = params
        self.helper = Helper(params)



    def perform_predict(self):

        for fold_id in self.params.data.folds:

            print(
                f"Predicting {self.params.model.name} over {self.params.data.name} (fold {fold_id}).")

            texts, texts_rpr, labels_rpr = self.helper.get_texts_labels(fold_id=fold_id, split="test")

            model = self.helper.load_model(fold_id=fold_id)

            # prediction is a csr_matrix with shape=(N, L)
            prediction = model.predict(
                texts,
                texts_rpr,
                kwargs=OmegaConf.to_container(self.params.model.pred_params, resolve=True))
            self.helper.checkpoint_prediction(prediction,fold_id)


