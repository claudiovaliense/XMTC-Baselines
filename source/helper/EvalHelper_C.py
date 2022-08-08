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
from tqdm import tqdm

from source.helper.Helper import Helper


class EvalHelper:

    def __init__(self, params):
        self.params = params
        self.helper = Helper(params)
        self.relevance_map = self._load_relevance_map()
        self.labels_cls = self.helper.load_labels_cls()
        self.texts_cls = self.helper.load_texts_cls()

    def _load_relevance_map(self):
        with open(f"{self.params.data.dir}relevance_map.pkl", "rb") as relevances_file:
            return pickle.load(relevances_file)

    def _load_labels_cls(self):
        with open(f"{self.params.data.dir}labels_cls.pkl", "rb") as labels_cls_file:
            return pickle.load(labels_cls_file)

    def _load_texts_cls(self):
        with open(f"{self.params.data.dir}texts_cls.pkl", "rb") as texts_cls_file:
            return pickle.load(texts_cls_file)



    def _load_ids_map(self, fold_id):
        with open(f"{self.params.data.dir}fold_{fold_id}/ids_map.pkl", "rb") as ids_map_file:
            return pickle.load(ids_map_file)

    def perform_eval(self):

        thresholds = [1, 5, 10]
        label_cls = ["all", "full", "few", "tail"]
        stats = []

        for fold_id in self.params.data.folds:
            print(
                f"Evaluating {self.params.model.name} over {self.params.data.name} (fold {fold_id}) with fowling params\n"
                f"{OmegaConf.to_yaml(self.params)}\n")

            prediction = self._load_prediction(fold_id)
            ids_map = self._load_ids_map(fold_id)
            ranking = self._retrieve(prediction,ids_map)

            for cls in label_cls:
                stat = {}
                relevances = self._eval_relevance(ranking, cls)
                stat["fold"] = fold_id
                stat["cls"] = cls
                for k in thresholds:
                    stat[f"MRR@{k}"] = self.mrr_at_k(relevances.values(), k, len(relevances))
                    stat[f"RCL@{k}"] = self.recall_at_k(relevances.values(), k, len(relevances))
                stats.append(stat)



        self.checkpoint_stats(
            pd.DataFrame(
                stats,
                columns=["fold", "cls", "MRR@1", "MRR@5", "MRR@10", "RCL@1", "RCL@5", "RCL@10"]
            ).sort_values(by=["cls"])
        )

    def _retrieve (self, prediction, ids_map):
        ranking = {}
        rows, cols = prediction.nonzero()
        for row, col in tqdm(zip(rows, cols), desc="Ranking"):
            ranking.setdefault(
                ids_map[row], []
            ).append(col)

        return ranking

    def _get_relevant_position(self, text_idx, retrieved_ids):

        for position, label_idx in enumerate(retrieved_ids):
            if label_idx in self.relevance_map[text_idx]:
                return position+1
        return 1e9

    def _eval_relevance (self, ranking, cls):
        relevances = {}
        for text_idx, labels_ids in ranking.items():
            if cls =="all":
                relevances[text_idx]=self._get_relevant_position(text_idx, labels_ids)
            if cls in self.texts_cls[text_idx]:
                # filter labels_ids of cls
                filtered_labels_ids = filter(lambda label_idx: cls == self.labels_cls[label_idx], labels_ids)
                relevances[text_idx]=self._get_relevant_position(text_idx, filtered_labels_ids)

        return relevances

    def mrr_at_k(self, positions, k, num_samples):
        """
        Evaluates the MMR considering only the positions up to k.
        :param positions:
        :param k:
        :param num_samples:
        :return:
        """
        # positions_at_k = [p for p in positions if p <= k]
        positions_at_k = [p if p <= k else 0 for p in positions]
        rrank = 0.0
        for pos in positions_at_k:
            if pos != 0:
                rrank += 1.0 / pos

        return rrank / num_samples

    def mrr(self, ranking):
        """
        Evaluates the MMR considering only the positions up to k.
        :param positions:
        :param num_samples:
        :return:
        """
        return np.mean(ranking)

    def recall_at_k(self, positions, k, num_samples):
        """
        Evaluates the Recall considering only the positions up to k
        :param positions:
        :param k:
        :param num_samples:
        :return:
        """
        return 1.0 * sum(i <= k for i in positions) / num_samples

    def precision_at_k(self, positions, k):

        assert k >= 1
        positions = np.asarray(positions)[:k] != 0
        if positions.size != k:
            raise ValueError('Relevance score length < k')
        return np.mean(positions)

    def checkpoint_stats(self, stats):
        """
        Checkpoints stats on disk.
        :param stats: dataframe
        """
        stats.to_csv(
            self.params.stat.dir + self.params.model.name + "_" + self.params.data.name + ".result",
            sep='\t', index=False, header=True)

    def checkpoint_ranking(self, ranking):
        ranking_path = f"{self.params.ranking.dir}" \
                       f"{self.params.model.name}_" \
                       f"{self.params.data.name}.rnk"
        with open(ranking_path, "wb") as ranking_file:
            pickle.dump(ranking, ranking_file)


