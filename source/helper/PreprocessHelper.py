import pickle

import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from sklearn.feature_extraction.text import TfidfVectorizer


class PreprocessHelper:

    def __init__(self, params):
        self.params = params

    def _get_texts(self, split,  fold_id):
        ids = self._load_ids(split, fold_id)
        samples_df = pd.DataFrame(self._load_samples())
        samples_df = samples_df[samples_df["idx"].isin(ids)]
        samples_df = samples_df[["text_idx", "text"]].drop_duplicates()
        return samples_df

    # def _get_texts(self, split, fold_id):
    #     ids = self._load_ids(split, fold_id)
    #     samples_df = pd.DataFrame(self._load_samples())
    #
    #     return samples_df[samples_df["idx"].isin(ids)]["text"].unique().tolist()

    def _load_ids(self, split, fold_id):
        with open(f"{self.params.data.dir}fold_{fold_id}/{split}.pkl", "rb") as ids_file:
            return pickle.load(ids_file)

    def _load_samples(self):
        with open(f"{self.params.data.dir}samples.pkl", "rb") as samples_file:
            return pickle.load(samples_file)
        

    def _get_vectorizer(self, texts):
        return TfidfVectorizer(
            analyzer="word",ngram_range=(1,2),max_df=0.98, max_features=50000
        ).fit(texts)

    def perform_preprocess(self):

        for fold_id in self.params.data.folds:
            print(
                f"Preprocess {self.params.model.name} over {self.params.data.name} (fold {fold_id}) with fowling self.params\n"
                f"{OmegaConf.to_yaml(self.params)}\n")

            train_texts = self._get_texts(split="train", fold_id=fold_id)["text"]
            test_texts = self._get_texts(split="test", fold_id=fold_id)
            vectorizer = self._get_vectorizer(train_texts)
            self._checkpoint_vectorizer(vectorizer, fold_id)
            ids_map = pd.Series(test_texts["text_idx"].values, index=test_texts.index).to_dict()
            self._checkpoint_ids_map(
                ids_map, fold_id
            )


    def _checkpoint_vectorizer(self, vectorizer, fold_id):
        with open(f"{self.params.data.dir}fold_{fold_id}/vectorizer.pkl", "wb") as vectorizer_file:
            pickle.dump(vectorizer, vectorizer_file)

    def _checkpoint_ids_map(self, ids_map, fold_id):
        with open(f"{self.params.data.dir}fold_{fold_id}/ids_map.pkl", "wb") as ids_map_file:
            pickle.dump(ids_map, ids_map_file)
