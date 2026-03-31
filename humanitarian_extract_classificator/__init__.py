import os
import shutil
import json
from typing import List

from humanitarian_extract_classificator.api_classification.zero_shot_second_level import (
    _generate_one_label_zero_shot_classification_results,
    _load_level2_definitions_dataset,
)
from humanitarian_extract_classificator.api_classification.zero_shot_assessment_problems import (
    _generate_zero_shot_assessment_problems_classification_prompts,
    _load_level2_problems_dataset,
)

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def humbert_classification(
    text: List[str],
    batch_size: int = 32,
    prediction_ratio: float = 1.00,
    model_path: str = None,
    return_ratio: bool = False,
) -> List[List[str]]:
    """
    Classify humanitarian text excerpts using the HumBERT debiased model.

    Args:
        text: List of text excerpts to classify.
        batch_size: Number of excerpts to process per batch.
        prediction_ratio: Decision boundary ratio (1.0 = max F1, >1.0 favors precision,
            <1.0 favors recall).
        model_path: Optional path to the model file. Defaults to
            ~/.cache/humanitarian_extract_classificator/humbert_debiased.pt and
            auto-downloads on first use.

    Returns:
        List of label lists, one per input excerpt.
    """
    from humanitarian_extract_classificator.humbert_classification.inference import (
        ClassificationInference,
    )

    pipeline = ClassificationInference(model_path=model_path)
    return pipeline(
        text,
        batch_size=batch_size,
        prediction_ratio=prediction_ratio,
        return_ratio=return_ratio,
    )


def level2_classification(
    entries: List[str],
    level1_classifications: List[List[str]],
    api_key: str = None,
    hf_dataset_name: str = "Sfekih/humanitarian_taxonomy_level2_definitions",
    hf_token: str = None,
    save_folder_path: os.PathLike = os.path.join(
        "data", "predictions", "classification_results"
    ),
    model: str = "gpt-4o-mini",
    pipeline: str = "OpenAI",
) -> List[List[str]]:
    """
    Classify humanitarian text excerpts into level-2 categories using an LLM.

    Args:
        entries: List of text excerpts.
        level1_classifications: Level-1 classification results (output of humbert_classification).
        api_key: OpenAI (or compatible) API key.
        hf_dataset_name: Hugging Face dataset with level-2 definitions.
        hf_token: Hugging Face token for private datasets.
        save_folder_path: Directory to save intermediate and final results.
        model: LLM model name.
        pipeline: API pipeline name ("OpenAI" or compatible).

    Returns:
        List of label lists, one per input excerpt.
    """
    from tqdm import tqdm

    if api_key is None:
        api_key = os.getenv("OPENAI_API_KEY")
    if hf_token is None:
        hf_token = os.getenv("hf_token")

    level1_to_level2_definitions = _load_level2_definitions_dataset(
        hf_dataset_name=hf_dataset_name, hf_token=hf_token
    )

    tmp_save_folder_path = os.path.join(save_folder_path, "tmp")
    os.makedirs(tmp_save_folder_path, exist_ok=True)

    for tagname_level1, level2_data in tqdm(
        level1_to_level2_definitions.items(),
        total=len(level1_to_level2_definitions),
        desc="Level 2 Classification",
    ):
        save_path = os.path.join(
            tmp_save_folder_path, f"{tagname_level1.replace('/', '_')}.json"
        )
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
    for one_level1_classification in level1_to_level2_definitions.keys():
        with open(
            os.path.join(
                tmp_save_folder_path,
                f"{one_level1_classification.replace('/', '_')}.json",
            ),
            "r",
        ) as f:
            one_level_classifications = json.load(f)
        for i, classification in enumerate(one_level_classifications):
            final_classifications[i].extend(classification)

    with open(os.path.join(save_folder_path, "level2_classifications.json"), "w") as f:
        json.dump(final_classifications, f)

    shutil.rmtree(tmp_save_folder_path)
    return final_classifications


def level2_problems_classification(
    entries: List[str],
    level2_classifications: List[List[str]],
    api_key: str = None,
    hf_dataset_name: str = "Sfekih/humanitarian_problems_questions",
    hf_token: str = None,
    save_folder_path: os.PathLike = os.path.join(
        "data", "predictions", "classification_results"
    ),
    model: str = "gpt-4o-mini",
    pipeline: str = "OpenAI",
) -> List[List[str]]:
    """
    Classify humanitarian text excerpts into problem categories using an LLM.

    Args:
        entries: List of text excerpts.
        level2_classifications: Level-2 classification results (output of level2_classification).
        api_key: OpenAI (or compatible) API key.
        hf_dataset_name: Hugging Face dataset with level-2 problem definitions.
        hf_token: Hugging Face token for private datasets.
        save_folder_path: Directory to save intermediate and final results.
        model: LLM model name.
        pipeline: API pipeline name ("OpenAI" or compatible).

    Returns:
        List of label lists, one per input excerpt.
    """
    from tqdm import tqdm

    if api_key is None:
        api_key = os.getenv("OPENAI_API_KEY")
    if hf_token is None:
        hf_token = os.getenv("hf_token")

    level1_to_level2_definitions = _load_level2_problems_dataset(
        hf_dataset_name=hf_dataset_name, hf_token=hf_token
    )

    tmp_save_folder_path = os.path.join(save_folder_path, "tmp")
    os.makedirs(tmp_save_folder_path, exist_ok=True)

    n_problem_definitions = sum(
        len(level2_data) for level2_data in level1_to_level2_definitions.values()
    )

    with tqdm(
        total=n_problem_definitions, desc="Level 2 Problems Classification"
    ) as pbar:
        for tagname_level1, level2_data in level1_to_level2_definitions.items():
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
                pbar.update(1)

    final_classifications = [[] for _ in entries]
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

    shutil.rmtree(tmp_save_folder_path)
    return final_classifications


__all__ = [
    "humbert_classification",
    "level2_classification",
    "level2_problems_classification",
]
