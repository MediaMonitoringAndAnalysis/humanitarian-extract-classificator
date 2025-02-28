from flask import Flask, request, jsonify
from typing import List, Dict
import shutil
import os
import json
import sys
import dotenv
import torch
import pandas as pd
from tqdm import tqdm

dotenv.load_dotenv()

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# import logging
# torch._logging.set_logs(dynamo = logging.INFO)

# Configure logging
# logging.basicConfig(level=logging.DEBUG)

# Enable anomaly detection
torch.autograd.set_detect_anomaly(True)

from src.api_classification.zero_shot_second_level import (
    _generate_one_label_zero_shot_classification_results,
    _load_level2_definitions_dataset,
)

from src.api_classification.zero_shot_assessment_problems import (
    _generate_zero_shot_assessment_problems_classification_prompts,
    _load_level2_problems_dataset,
)


def humbert_classification(
    text: List[str], batch_size: int = 32, prediction_ratio: float = 1.00
):

    # Change directory to the src/humbert_classification folder
    # needed to run these models

    os.chdir(os.path.join("src", "humbert_classification"))
    sys.path.append(os.getcwd())

    # print(os.getcwd())

    # Import and initialize the classification pipeline
    from inference import ClassificationInference

    classification_pipeline = ClassificationInference()

    classification_results: List[List[str]] = classification_pipeline(
        text, batch_size=batch_size, prediction_ratio=prediction_ratio
    )

    os.chdir("../../")
    sys.path.append(os.getcwd())

    return classification_results




def level2_classification(
    entries: List[str],
    level1_classifications: List[List[str]],
    api_key: str = os.getenv("openai_api_key"),
    hf_dataset_name: str = "Sfekih/humanitarian_taxonomy_level2_definitions",
    hf_token: str = os.getenv("hf_token"),
    save_folder_path: os.PathLike = os.path.join(
        "data", "predictions", "classification_results"
    ),
    model: str = "gpt-4o-mini",
    pipeline="OpenAI",
) -> List[List[str]]:
    """
    This function is used to classify the text into a level2 category.
    It uses the level1 classification and the definition file to classify the text.
    """

    level1_to_level2_definitions = _load_level2_definitions_dataset(
        hf_dataset_name=hf_dataset_name, hf_token=hf_token
    )

    # useful for not rerunning everything again incase of an error or a bug
    tmp_save_folder_path = os.path.join(save_folder_path, "tmp")
    os.makedirs(tmp_save_folder_path, exist_ok=True)

    for (
        tagname_level1,
        level2_data,
    ) in tqdm(
        level1_to_level2_definitions.items(),
        total=len(level1_to_level2_definitions),
        desc="Level 2 Classification",
    ):

        save_path = os.path.join(tmp_save_folder_path, f"{tagname_level1.replace('/', '_')}.json")

        if not os.path.exists(save_path):
            _generate_one_label_zero_shot_classification_results(
                entries=entries,
                tagname_level1=tagname_level1,
                level2_classifications=level1_classifications,
                level2_definitions=level2_data,
                save_path=save_path,
                api_key=api_key,
                api_pipeline=pipeline,
                model=model,
            )

    final_classifications = [[] for _ in entries]
    # reload the classifications from the tmp folder and merge them with the level1 classification to have a multilabel classification
    for one_level1_classification in level1_to_level2_definitions.keys():
        with open(
            os.path.join(tmp_save_folder_path, f"{one_level1_classification.replace('/', '_')}.json"), "r"
        ) as f:
            one_level_classifications = json.load(f)

        for i, classification in enumerate(one_level_classifications):
            final_classifications[i].extend(classification)

    with open(os.path.join(save_folder_path, "level2_classifications.json"), "w") as f:
        json.dump(final_classifications, f)

    # delete the tmp folder
    shutil.rmtree(tmp_save_folder_path)

    return final_classifications


def level2_problems_classification(
    entries: List[str],
    level2_classifications: List[List[str]],
    api_key: str = os.getenv("openai_api_key"),
    hf_dataset_name: str = "Sfekih/humanitarian_problems_questions",
    hf_token: str = os.getenv("hf_token"),
    save_folder_path: os.PathLike = os.path.join(
        "data", "predictions", "classification_results"
    ),
    model: str = "gpt-4o-mini",
    pipeline="OpenAI",
) -> List[List[str]]:
    """
    This function is used to classify the text into a problems classification from the level2 classification.
    It uses the level2 classification and the definition and questions file to classify the text.
    """

    level1_to_level2_definitions = _load_level2_problems_dataset(
        hf_dataset_name=hf_dataset_name, hf_token=hf_token
    )

    # useful for not rerunning everything again incase of an error or a bug
    tmp_save_folder_path = os.path.join(save_folder_path, "tmp")
    os.makedirs(tmp_save_folder_path, exist_ok=True)

    n_problem_definitions = sum(
        len(level2_data) 
        for level2_data in level1_to_level2_definitions.values()
    )

    tqdm_total = tqdm(
        total=n_problem_definitions, desc="Level 2 Problems Classification"
    )
    with tqdm_total:
        for (
            tagname_level1,
            level2_data,
        ) in level1_to_level2_definitions.items():

            for tagname_level2, level2problems in level2_data.items():

                save_path = os.path.join(
                    tmp_save_folder_path,
                    f"{tagname_level1}_{tagname_level2}_problems.json".replace(
                        "/", "_"
                    ),
                )

                if not os.path.exists(save_path):

                    _generate_zero_shot_assessment_problems_classification_prompts(
                        entries=entries,
                        tagname_level1=tagname_level1,
                        tagname_level2=tagname_level2,
                        level2_classifications=level2_classifications,
                        level2_problems_definitions=level2problems,
                        save_path=save_path,
                        api_key=api_key,
                        api_pipeline=pipeline,
                        model=model,
                    )

                tqdm_total.update(1)

    final_classifications = [[] for _ in entries]
    # reload the classifications from the tmp folder and merge them with the level1 classification to have a multilabel classification
    for one_level1_classification in level1_to_level2_definitions.keys():
        for tagname_level2 in level1_to_level2_definitions[
            one_level1_classification
        ].keys():

            with open(
                os.path.join(
                    tmp_save_folder_path,
                    f"{one_level1_classification}_{tagname_level2}_problems.json".replace(
                        "/", "_"
                    ),
                ),
                "r",
            ) as f:
                one_level_classifications = json.load(f)

            for i, classification in enumerate(one_level_classifications):
                final_classifications[i].extend(classification)

    with open(
        os.path.join(save_folder_path, "level2_problems_classifications.json"), "w"
    ) as f:
        json.dump(final_classifications, f)

    # delete the tmp folder
    shutil.rmtree(tmp_save_folder_path)

    return final_classifications
