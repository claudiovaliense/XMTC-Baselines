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

    def _get_samples(self, split, fold_id):
        ids = self._load_ids(split, fold_id)
        samples_df = pd.DataFrame(self._load_samples())
        return samples_df[samples_df["idx"].isin(ids)]

    def _load_ids(self, split, fold_id):
        with open(f"{self.params.data.dir}fold_{fold_id}/{split}.pkl", "rb") as ids_file:
            return pickle.load(ids_file)

    def _load_samples(self):
        with open(f"{self.params.data.dir}samples.pkl", "rb") as samples_file:
            return pickle.load(samples_file)


    def _get_texts_labels(self, fold_id):
        samples_df = self._get_samples(split="train", fold_id=fold_id)
        texts_labels = samples_df.groupby(by=["text"])["label_idx"].apply(list)

        vectorizer = self._load_vectorizer(fold_id=fold_id)

        row_idx = 0
        texts, rows, cols, data = [], [], [], []

        for text, label_ids in texts_labels.items():
            for label_idx in label_ids:
                rows.append(row_idx)
                cols.append(label_idx)
                data.append(1.0)
            texts.append(text)
            row_idx += 1

        texts_rpr = sparse.csr_matrix(vectorizer.transform(texts), dtype=np.float32)
        labels_rpr = sparse.csr_matrix((data, (rows, cols)), shape=(len(texts), self.params.data.num_labels),
                                       dtype=np.float32)
        texts_rpr.sort_indices()
        labels_rpr.sort_indices()
        return texts, texts_rpr, labels_rpr

    def perform_fit(self):

        for fold_id in self.params.data.folds:
            print(
                f"Fitting {self.params.model.name} over {self.params.data.name} (fold {fold_id}) with fowling self.params\n"
                f"{OmegaConf.to_yaml(self.params)}\n")

            texts, texts_rpr, labels_rpr = self._get_texts_labels(fold_id)


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

    def _load_vectorizer(self, fold_id):
        with open(f"{self.params.data.dir}fold_{fold_id}/vectorizer.pkl", "rb") as vectorizer_file:
            return pickle.load(vectorizer_file)

