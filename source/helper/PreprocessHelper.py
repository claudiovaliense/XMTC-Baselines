from omegaconf import OmegaConf
from source.helper.Helper import Helper


class PreprocessHelper:

    def __init__(self, params):
        self.helper = Helper(params)

    def perform_preprocess(self):

        for fold_id in self.helper.params.data.folds:
            print(
                f"Preprocess {self.helper.params.model.name} over {self.helper.params.data.name} (fold {fold_id}) with fowling self.params\n"
                f"{OmegaConf.to_yaml(self.helper.params)}\n")

            #vectorizer
            train_samples_df = self.helper.get_samples(fold_id=fold_id,split="train")

            vectorizer = self.helper.get_vectorizer(train_samples_df["text"])
            self.helper.checkpoint_vectorizer(vectorizer, fold_id)

            # # id maps to test
            # test_samples_df = self._get_texts(split="test", fold_id=fold_id).reset_index(drop=True)
            # ids_map = pd.Series(test_samples_df["text_idx"].values, index=test_samples_df.index).to_dict()
            # self._checkpoint_ids_map(
            #     ids_map, fold_id
            # )


