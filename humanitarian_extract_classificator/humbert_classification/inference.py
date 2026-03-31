import os
import sys
import torch
import gdown
from typing import List
from tqdm import tqdm
from copy import copy
import gc
from nltk.tokenize import sent_tokenize, word_tokenize

_GDRIVE_MODEL_ID = "1nY_rV2Eqjxe4dCmF2V_5YCJjcTYFMLG8"

# Directory of this file — needed so torch.load can find sibling modules
# (pooling, TransformerModel, dataset_creation) by their original flat names
# during unpickling.
_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))

_DEFAULT_MODEL_PATH = os.path.join(
    os.path.expanduser("~"), ".cache", "humanitarian_extract_classificator", "humbert_debiased.pt"
)


_NOT_KEPT_TAGS = [
    "secondary_tags->Age-><18 years",
    "secondary_tags->Age-><18 years old",
    "secondary_tags->Age->18-24 years old",
    "secondary_tags->Age->25-59 years old",
    "secondary_tags->Age->5-11 years old",
    "secondary_tags->Age->12-17 years old",
    "secondary_tags->Displaced->Others of concern",
    "secondary_tags->Displaced->Pendular",
    "secondary_tags->Gender->All",
    "first_level_tags->sectors->Cross",
    "subsectors->Logistics->Supply chain",
    "first_level_tags->pillars_1d->Covid-19",
    "subpillars_1d->Covid-19->Cases",
    "subpillars_1d->Covid-19->Contact tracing",
    "subpillars_1d->Covid-19->Deaths",
    "subpillars_1d->Covid-19->Hospitalization & care",
    "subpillars_1d->Covid-19->Prevention campaign",
    "subpillars_1d->Covid-19->Research and outlook",
    "subpillars_1d->Covid-19->Restriction measures",
    "subpillars_1d->Covid-19->Testing",
    "subpillars_1d->Covid-19->Vaccination",
]

_TAG_RENAMING_DICT = {
    "subsectors": "Sectors",
    "first_level_tags->": "",
    "subpillars": "pillars",
    "pillars_1d": "Pillars 1D",
    "pillars_2d": "Pillars 2D",
    "sectors": "Sectors",
    "secondary_tags": "Secondary Tags",
    "specific_needs_groups": "Specific Needs Groups",
}


def _clean_tag(original_tag: str) -> str:
    clean_tag = copy(original_tag).replace("/", "-")
    for k, v in _TAG_RENAMING_DICT.items():
        clean_tag = clean_tag.replace(k, v)
    return clean_tag.title().replace("Wash", "WASH").replace("Idp", "IDP")


def _postprocess_classification_predictions(original_tags: List[str]):
    clean_tags = [
        _clean_tag(t) for t in original_tags if t not in _NOT_KEPT_TAGS
    ]
    return sorted(list(set(clean_tags)))


def _postprocess_ratio_predictions(ratio_dict: dict) -> dict:
    """Clean tag names in a ratio dict, keeping the max ratio when two raw
    tags map to the same cleaned name, and dropping not-kept tags."""
    clean: dict = {}
    for original_tag, ratio in ratio_dict.items():
        if original_tag in _NOT_KEPT_TAGS:
            continue
        clean_tag = _clean_tag(original_tag)
        if clean_tag not in clean or ratio > clean[clean_tag]:
            clean[clean_tag] = ratio
    return clean


class ClassificationInference:
    def __init__(self, model_path: str = None):
        if model_path is None:
            model_path = _DEFAULT_MODEL_PATH

        if not os.path.exists(model_path):
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            print(f"Model not found at '{model_path}'. Downloading from Google Drive...")
            gdown.download(id=_GDRIVE_MODEL_ID, output=model_path, quiet=False)

        # torch.load unpickling requires the module classes (pooling, TransformerModel,
        # dataset_creation) to be importable by their original flat names.
        sys.path.insert(0, _MODULE_DIR)
        try:
            self.classification_model = torch.load(model_path, weights_only=False)
        finally:
            sys.path.remove(_MODULE_DIR)

        self.classification_model.tokenizer.split_special_tokens = False
        self.classification_model.tokenizer.pad_token = "<pad>"
        self.classification_model.tokenizer.pad_token_id = 1
        self.classification_model.trained_architecture.common_backbone.attn_implementation = "eager"
        self.classification_model.trained_architecture.common_backbone.encoder.gradient_checkpointing = (
            False
        )
        self.classification_model.trained_architecture.eval()
        self.classification_model.val_params["num_workers"] = 0
        self.classification_model.tokenizer._in_target_context_manager = False
        for param in self.classification_model.trained_architecture.parameters():
            param.requires_grad = False

        if torch.cuda.is_available():
            self.classification_model.testing_device = "cuda"
        else:
            self.classification_model.testing_device = "cpu"

        self.classification_model.trained_architecture.to(
            self.classification_model.testing_device
        )

    def generate_predictions_by_sentence(
        self,
        test_entries: List[str],
        prediction_ratio: float = 1.0,
        return_ratio: bool = False,
    ):
        all_sentences = []
        entry_indices = []
        for i, entry in enumerate(test_entries):
            sentences = sent_tokenize(entry)
            all_sentences.extend(sentences)
            entry_indices.extend([i] * len(sentences))

        max_len = self._get_max_len(all_sentences)
        all_predictions = self.classification_model.custom_predict(
            all_sentences,
            max_len=max_len,
            prediction_ratio=prediction_ratio,
            return_ratio=return_ratio,
        )

        if return_ratio:
            # Merge sentence-level ratio dicts per entry, keeping the max ratio
            # per tag across all sentences belonging to the same entry.
            predictions: List[dict] = [{} for _ in test_entries]
            for i, preds in enumerate(all_predictions):
                entry_idx = entry_indices[i]
                for tagname, ratio in preds.items():
                    if ratio > predictions[entry_idx].get(tagname, 0.0):
                        predictions[entry_idx][tagname] = ratio
        else:
            predictions_list: List[list] = [[] for _ in test_entries]
            for i, preds in enumerate(all_predictions):
                entry_idx = entry_indices[i]
                predictions_list[entry_idx].extend(preds)
            predictions = [sorted(list(set(ep))) for ep in predictions_list]

        return predictions

    def _get_max_len(self, excerpts: List[str]):
        return int(min(512, 1.5 * max([len(word_tokenize(e)) for e in excerpts])))

    def __call__(
        self,
        excerpts: List[str],
        predict_by_sentence: bool = True,
        batch_size: int = 4,
        prediction_ratio: float = 1.05,
        return_ratio: bool = False,
    ):
        """
        prediction_ratio: threshold ratio (predicted probability / optimal threshold).
            Labels whose ratio exceeds this value are returned when return_ratio=False.
        return_ratio: when True, return a List[Dict[str, float]] of cleaned-tag -> ratio
            dicts instead of binary tag lists.  The max prediction for each excerpt is
            simply ``max(d, key=d.get)`` on the returned dict.
        """
        self.batch_size = batch_size
        self.classification_model.val_params["batch_size"] = batch_size
        final_outputs = []
        for i in tqdm(
            range(0, len(excerpts), self.batch_size),
            desc="Getting classification predictions",
        ):
            excerpts_one_batch = excerpts[i : i + self.batch_size]
            if predict_by_sentence:
                batch_outputs = self.generate_predictions_by_sentence(
                    excerpts_one_batch,
                    prediction_ratio=prediction_ratio,
                    return_ratio=return_ratio,
                )
            else:
                max_len = self._get_max_len(excerpts_one_batch)
                batch_outputs = self.classification_model.custom_predict(
                    excerpts_one_batch,
                    max_len=max_len,
                    prediction_ratio=prediction_ratio,
                    return_ratio=return_ratio,
                )
            final_outputs.extend(batch_outputs)

            gc.collect()

        if return_ratio:
            final_outputs = [
                _postprocess_ratio_predictions(entry_ratios)
                for entry_ratios in final_outputs
            ]
        else:
            final_outputs = [
                _postprocess_classification_predictions(entry_tags)
                for entry_tags in final_outputs
            ]
        return final_outputs
