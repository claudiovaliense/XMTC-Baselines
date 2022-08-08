import pickle

import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from ranx import Qrels
from ranx import Run
from ranx import evaluate
from tqdm.contrib import tzip

from source.helper.Helper import Helper


class EvalHelper:

    def __init__(self, params):
        self.params = params
        self.helper = Helper(params)
        self.relevance_map = self._load_relevance_map()
        self.labels_cls = self._load_labels_cls()
        self.texts_cls = self._load_texts_cls()
        self.metrics = self._get_metrics()

    def _get_metrics(self):
        metrics = []
        for metric in self.params.eval.metrics:
            for threshold in self.params.eval.thresholds:
                metrics.append(f"{metric}@{threshold}")
        return metrics

    def _retrieve (self, prediction, ids_map, cls):
        ranking = {}
        rows, cols = prediction.nonzero()
        for row, col in tzip(rows,cols, desc="Ranking"):
            text_idx = ids_map[row]
            label_idx = col
            if (cls in self.labels_cls[label_idx] and cls in self.texts_cls[text_idx]) or cls=="all":
                if f"text_{text_idx}" in ranking:
                    ranking[f"text_{text_idx}"][f"label_{label_idx}"] = prediction[row,label_idx]
                else:
                    ranking[f"text_{text_idx}"] = {}
                    ranking[f"text_{text_idx}"][f"label_{label_idx}"] = prediction[row,label_idx]
        return ranking

    def perform_eval(self):


        results = []

        for fold_id in self.helper.params.data.folds:
            print(
                f"Evaluating {self.params.model.name} over {self.params.data.name} (fold {fold_id}) with fowling params\n"
                f"{OmegaConf.to_yaml(self.params)}\n")

            prediction = self.helper.load_prediction(fold_id)
            ids_map = self._load_ids_map(fold_id)

            for cls in self.params.eval.label_cls:
                ranking = self._retrieve(prediction, ids_map, cls)
                filtered_dictionary = {key: value for key, value in self.relevance_map.items() if key in ranking.keys()}
                qrels = Qrels(filtered_dictionary, name=cls)
                run = Run(ranking, name=cls)
                result = evaluate(qrels, run, self.metrics, threads=12)
                result["fold"]=fold_id
                result["cls"]=cls
                results.append(result)


        self.helper.checkpoint_results(results)

    def _load_relevance_map(self):
        with open(f"{self.params.data.dir}relevance_map.pkl", "rb") as relevances_file:
            data = pickle.load(relevances_file)
        relevance_map = {}
        for text_idx, labels_ids in data.items():
            d = {}
            for label_idx in labels_ids:
                d[f"label_{label_idx}"]=1.0
            relevance_map[f"text_{text_idx}"]=d
        return  relevance_map

    def _load_labels_cls(self):
        with open(f"{self.params.data.dir}label_cls.pkl", "rb") as label_cls_file:
            return pickle.load(label_cls_file)

    def _load_texts_cls(self):
        with open(f"{self.params.data.dir}text_cls.pkl", "rb") as text_cls_file:
            return pickle.load(text_cls_file)



    def _load_ids_map(self, fold_id):
        test_samples_df = self.helper.get_samples(fold_id=fold_id, split="test")
        return pd.Series(
            test_samples_df["text_idx"].values,
            index=test_samples_df.index
        ).to_dict()



