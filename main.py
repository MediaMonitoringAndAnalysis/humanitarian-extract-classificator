from flask import Flask, request, jsonify
from typing import List
import os
import json
import sys
import torch

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import logging

# torch._logging.set_logs(dynamo = logging.INFO)

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Enable anomaly detection
torch.autograd.set_detect_anomaly(True)

from src.api_classification.zero_shot_second_level import _generate_classification_prompts


def humbert_classification(
    text: List[str], batch_size: int = 32, prediction_ratio: float = 1.00
):

    # Change directory to the src/humbert_classification folder
    # needed to run these models
    os.chdir("src/humbert_classification")

    # Import and initialize the classification pipeline
    from inference import ClassificationInference

    classification_pipeline = ClassificationInference()

    classification_results: List[List[str]] = classification_pipeline(
        text, batch_size, prediction_ratio
    )

    os.chdir("../../")

    return classification_results



def level2_classification(
    entries: List[str],
    level1_classification: List[List[str]],
    definition_file_path: os.PathLike,
    save_folder_path: os.PathLike,
    api_key: str,
    model: str = "gpt-4o-mini",
    pipeline="OpenAI",
) -> List[List[str]]:
    """
    This function is used to classify the text into a level2 category.
    It uses the level1 classification and the definition file to classify the text.
    """

    with open(definition_file_path, "r") as f:
        level1_to_level2_definitions = json.load(f)

    # useful for not rerunning everything again incase of an error or a bug
    tmp_save_folder_path = os.path.join(save_folder_path, "tmp")
    os.makedirs(tmp_save_folder_path, exist_ok=True)

    for (
        level1_classification,
        level2_definitions,
    ) in level1_to_level2_definitions.items():

        save_path = os.path.join(tmp_save_folder_path, f"{level1_classification}.json")

        if not os.path.exists(save_path):
            _generate_classification_prompts(
                entries,
                level1_classification,
                level2_definitions,
                save_path,
                api_key,
                pipeline,
                model,
            )

    final_classifications = [[] for _ in entries]
    # reload the classifications from the tmp folder and merge them with the level1 classification to have a multilabel classification
    for one_level1_classification in level1_to_level2_definitions.keys():
        with open(os.path.join(tmp_save_folder_path, f"{one_level1_classification}.json"), "r") as f:
            one_level_classifications = json.load(f)

        for i, classification in enumerate(one_level_classifications):
            final_classifications[i].extend(classification)

    with open(os.path.join(save_folder_path, "level2_classifications.json"), "w") as f:
        json.dump(final_classifications, f)

    return final_classifications
