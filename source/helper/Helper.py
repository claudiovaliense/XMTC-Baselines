import pickle

import numpy as np
import pandas as pd
from pecos.xmc.xtransformer.model import XTransformer
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer


class Helper:

    def __init__(self, params):
        self.params = params

    # samples
    def get_samples(self, fold_id, split):
        ids = self.load_ids(fold_id, split)
        samples_df = pd.DataFrame(self.load_samples())
        samples_df = samples_df[samples_df["idx"].isin(ids)].reset_index(drop=True)
        return samples_df

    def load_ids(self, fold_id, split):
        with open(f"{self.params.data.dir}fold_{fold_id}/{split}.pkl", "rb") as ids_file:
            return pickle.load(ids_file)

    def load_samples(self):
        with open(f"{self.params.data.dir}samples.pkl", "rb") as samples_file:
            return pickle.load(samples_file)

    def get_texts_labels(self, fold_id, split):
        samples_df = self.get_samples(fold_id=fold_id, split=split)
        vectorizer = self.load_vectorizer(fold_id=fold_id)

        rows, cols, data = [], [], []

        for row_idx, row in samples_df.iterrows():
            for label_idx in row["labels_ids"]:
                rows.append(row_idx)
                cols.append(label_idx)
                data.append(1.0)

        texts_rpr = sparse.csr_matrix(vectorizer.transform(samples_df["text"]), dtype=np.float32)
        labels_rpr = sparse.csr_matrix((data, (rows, cols)), shape=(samples_df.shape[0], self.params.data.num_labels),
                                       dtype=np.float32)
        texts_rpr.sort_indices()
        labels_rpr.sort_indices()
        return samples_df["text"].tolist(), texts_rpr, labels_rpr



    def _checkpoint_ids_map(self, ids_map, fold_id):
        with open(f"{self.params.data.dir}fold_{fold_id}/ids_map.pkl", "wb") as ids_map_file:
            pickle.dump(ids_map, ids_map_file)

    # model
    def checkpoint_model(self, model, fold_id):
        model.save(
            f"{self.params.model_checkpoint.dir}"
            f"{self.params.model.name}_"
            f"{self.params.data.name}_"
            f"{fold_id}"
        )

    def load_model(self, fold_id):
        return XTransformer.load(
            f"{self.params.model_checkpoint.dir}"
            f"{self.params.model.name}_"
            f"{self.params.data.name}_"
            f"{fold_id}"
        )

    # prediction
    def checkpoint_prediction(self, prediction, fold_id):

        with open(f"{self.params.prediction.dir}"
                  f"{self.params.model.name}_"
                  f"{self.params.data.name}_"
                  f"{fold_id}.prd", "wb") as prediction_file:
            pickle.dump(prediction, prediction_file)

    def load_prediction(self, fold_id):

        with open(f"{self.params.prediction.dir}"
                  f"{self.params.model.name}_"
                  f"{self.params.data.name}_"
                  f"{fold_id}.prd", "rb") as prediction_file:
            return pickle.load(prediction_file)



    # vectorizer
    def get_vectorizer(self, texts):
        return TfidfVectorizer(
            analyzer="word", ngram_range=(1, 2), max_df=0.98, max_features=50000
        ).fit(texts)

    def checkpoint_vectorizer(self, vectorizer, fold_id):
        with open(f"{self.params.data.dir}fold_{fold_id}/vectorizer.pkl", "wb") as vectorizer_file:
            pickle.dump(vectorizer, vectorizer_file)

    def load_vectorizer(self, fold_id):
        with open(f"{self.params.data.dir}fold_{fold_id}/vectorizer.pkl", "rb") as vectorizer_file:
            return pickle.load(vectorizer_file)

    # text and labels cls
    def load_labels_cls(self):
        with open(f"{self.params.data.dir}label_cls.pkl", "rb") as label_cls_file:
            return pickle.load(label_cls_file)

    def load_texts_cls(self):
        with open(f"{self.params.data.dir}text_cls.pkl", "rb") as text_cls_file:
            return pickle.load(text_cls_file)

    # results
    def checkpoint_results(self, results):
        """
        Checkpoints stats on disk.
        :param stats: dataframe
        """
        pd.DataFrame(results).to_csv(
            self.params.result.dir + self.params.model.name + "_" + self.params.data.name + ".rts",
            sep='\t', index=False, header=True)

    # ranking
    def checkpoint_rankings(self, ranking):
        ranking_path = f"{self.params.ranking.dir}" \
                       f"{self.params.model.name}_" \
                       f"{self.params.data.name}.rnk"
        with open(ranking_path, "wb") as ranking_file:
            pickle.dump(ranking, ranking_file)

