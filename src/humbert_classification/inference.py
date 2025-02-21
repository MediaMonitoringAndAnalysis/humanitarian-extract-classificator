import torch
from typing import List
from tqdm import tqdm
from copy import copy
import gc


def _postprocess_classification_predictions(original_tags: List[str]):

    not_kept_tags = [
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

    tag_renaming_dict = {
        "subsectors": "Sectors",
        "first_level_tags->": "",
        "subpillars": "pillars",
        "pillars_1d": "Pillars 1D",
        "pillars_2d": "Pillars 2D",
        "sectors": "Sectors",
        "secondary_tags": "Secondary Tags",
        "specific_needs_groups": "Specific Needs Groups",
    }

    clean_tags = []

    for original_tag in original_tags:
        if original_tag not in not_kept_tags:
            clean_tag = copy(original_tag).replace("/", "-")
            for k, v in tag_renaming_dict.items():
                clean_tag = clean_tag.replace(k, v)
            clean_tags.append(
                clean_tag.title().replace("Wash", "WASH").replace("Idp", "IDP")
            )

    clean_tags = sorted(list(set(clean_tags)))
    return clean_tags


class ClassificationInference:
    def __init__(self, model_path: str = "humbert_debiased.pt"):
        self.classification_model = torch.load(model_path)
        self.classification_model.tokenizer.split_special_tokens = False
        self.classification_model.trained_architecture.common_backbone.attn_implementation="eager"
        self.classification_model.trained_architecture.common_backbone.encoder.gradient_checkpointing = (
            False
        )
        self.classification_model.trained_architecture.eval()
        # self.classification_model.trained_architecture.freeze()
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

    def __call__(
        self,
        excerpts: List[str],
        batch_size: int = 4,
        prediction_ratio: float = 1.05,
    ) -> List[List[str]]:
        """
        prediction_ratio: if None: return ratio between predited probability and optimal threshold, otherwise return labels with ratio > prediction_ratio
        """
        self.batch_size = batch_size
        self.classification_model.val_params["batch_size"] = batch_size
        all_ratios = []
        for i in tqdm(
            range(0, len(excerpts), self.batch_size),
            desc="Getting classification predictions",
        ):
            excerpts_one_batch = excerpts[i : i + self.batch_size]
            max_len = min(
                256, int(1.3 * max([len(x.split()) for x in excerpts_one_batch]))
            )

            all_ratios_one_batch = self.classification_model.custom_predict(
                excerpts_one_batch, max_len=max_len
            )
            all_ratios.extend(all_ratios_one_batch)
            
            gc.collect()

        final_tags = [
            [k for k, v in x.items() if v > prediction_ratio] for x in all_ratios
        ]
        final_tags = [
            _postprocess_classification_predictions(one_entry_tags)
            for one_entry_tags in final_tags
        ]
        return final_tags
