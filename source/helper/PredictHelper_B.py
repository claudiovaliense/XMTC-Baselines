import pickle

import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from pecos.xmc.xtransformer.model import XTransformer
from scipy import sparse


class PredictHelper:

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

    def _vectorize(self, texts):
        vectorizer = self._load_vectorizer()
        texts_rpr = vectorizer.predict(texts)
        return texts_rpr

    def _reshape(self, samples_df, num_labels):
        text_labels = samples_df.groupby(by=["text"])["label_idx"].apply(list)
        row_idx = 0
        texts, rows, cols, data = [], [], [], []

        for text, label_ids in text_labels.items():
            for label_idx in label_ids:
                rows.append(row_idx)
                cols.append(label_idx)
                data.append(1.0)
            texts.append(text)
            row_idx += 1

        texts_rpr = self._vectorize(texts)
        labels_rpr = sparse.csr_matrix((data, (rows, cols)), shape=(len(texts), num_labels), dtype=np.float32)
        return texts, texts_rpr, labels_rpr

    def perform_predict(self):

        for fold_id in self.params.data.folds:

            print(
                f"Predicting {self.params.model.name} over {self.params.data.name} (fold {fold_id}) with fowling self.params\n"
                f"{OmegaConf.to_yaml(self.params)}\n")

            texts, texts_rpr, labels_rpr = self._reshape(
                samples_df=self._get_samples(split="test", fold_id=fold_id),
                num_labels=self.params.data.num_labels
            )

            print(len(texts))
            print(texts_rpr.shape)
            print(labels_rpr.shape)

            # model = self._load_model(fold_id)
            #
            # # prediction is a csr_matrix with shape=(N, L)
            # prediction = model.predict(
            #     texts,
            #     texts_rpr,
            #     kwargs=OmegaConf.to_container(self.params.model.pred_params, resolve=True))
            # self._checkpoint(prediction,fold_id)

    def _checkpoint(self, prediction, fold_id):
        prediction_path=f"{self.params.prediction.dir}{self.params.model.name}{self.params.data.name}_{fold_id}"
        with open(prediction_path, "wb") as prediction_file:
            pickle.dump(prediction, prediction_file)


    def _load_model(self, fold_id):
        return XTransformer.load(
            f"{self.params.model_checkpoint.dir}"
            f"{self.params.model.name}_"
            f"{self.params.data.name}_"
            f"{fold_id}"
        )

    def _load_vectorizer(self):
        with open(f"{self.params.data.dir}tfidf.vzr", "rb") as vectorizer_file:
            return pickle.load(vectorizer_file)

