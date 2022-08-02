import pickle

import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from pecos.xmc.xtransformer.model import XTransformer
from pecos.xmc.xtransformer.module import MLProblemWithText
from pecos.utils import smat_util
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from pecos.utils.featurization.text.vectorizers import Vectorizer


class FitHelper:

    def __init__(self, params):
        self.params = params

    def perform_fit(self):

        for fold_id in self.params.data.folds:
            print(
                f"Fitting {self.params.model.name} over {self.params.data.name} (fold {fold_id}) with fowling self.params\n"
                f"{OmegaConf.to_yaml(self.params)}\n")

            texts = []
            with open(f"{self.params.data.dir}fold_/{fold_id}/test_texts.txt", 'w') as f:
                for text in f:
                    texts.append(text)

            texts_rpr = smat_util.load_matrix(f"{self.params.data.dir}fold_/{fold_id}/train_texts_rpr.npz")
            labels_rpr = smat_util.load_matrix(f"{self.params.data.dir}fold_/{fold_id}/train_labels_rpr.npz")

            problem = MLProblemWithText(X_text=texts, Y=labels_rpr, X_feat=texts_rpr)
            train_params = XTransformer.TrainParams.from_dict(
                OmegaConf.to_container(self.params.model.train_params, resolve=True)
            )
            model = XTransformer.train(problem,
                                       train_params=train_params)
            self._checkpoint_model(model, fold_id)


    def _checkpoint_model(self, model, fold_id):
        model.save(
            f"{self.params.model_checkpoint.dir}"
            f"{self.params.model.name}_"
            f"{self.params.data.name}_"
            f"{fold_id}"
        )


    def _load_vectorizer(self):
        return Vectorizer.load(f"{self.params.data.dir}/vectorizer")

