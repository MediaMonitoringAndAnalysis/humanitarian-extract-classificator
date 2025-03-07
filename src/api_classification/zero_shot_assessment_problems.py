from typing import List, Dict
import os
import json
from llm_multiprocessing_inference import get_answers
from datasets import load_dataset
import pandas as pd
from collections import defaultdict


classification_system_prompt = """
You are a humanitarian expert.
You are given a text and a JSON dictionary where the keys are the problems and the values are descriptive questions that are related to the problem.
%s
You need to classify the text into the problems which are related to the questions.
The output is a JSON List[str] of problems (keys of the JSON dictionary). The task is a multilabel classification task.
The output is a list with no entries, one entry, or multiple entries.
If unsure about a problem, return it anyways (high recall is crucial).
"""


def _load_level2_problems_dataset(
    hf_dataset_name: str = "Sfekih/humanitarian_problems_questions", hf_token: str = os.getenv("hf_token")
) -> Dict[str, Dict[str, str]]:
    dataset = load_dataset(hf_dataset_name, token=hf_token)
    dataset_df = pd.DataFrame(dataset["train"])
    level1_to_level2_problems = defaultdict(lambda: defaultdict(dict))
    for index, row in dataset_df.iterrows():
        level1 = f"{row['task']}->{row['level1']}"
        level1_to_level2_problems[level1][row["level2"]][row["problem"]] = row[
            "question(s)"
        ]
    return level1_to_level2_problems

def _generate_zero_shot_assessment_problems_classification_prompts(
    entries: List[str],
    tagname_level1: str,
    tagname_level2: str,
    level2_classifications: List[List[str]],
    level2_problems_definitions: Dict[str, str],
    save_path: os.PathLike,
    api_key: str,
    api_pipeline: str = "OpenAI",
    model: str = "gpt-4o-mini",
):

    classifications_of_level1 = [[] for _ in entries]

    ids_of_entries_with_tag = [
        i
        for i, entry_tags in enumerate(level2_classifications)
        if str(tagname_level2) in str(entry_tags)
    ]

    classification_prompts = []
    for i in ids_of_entries_with_tag:
        classification_prompts.append(
            [
                {
                    "role": "system",
                    "content": classification_system_prompt
                    % level2_problems_definitions,
                },
                {"role": "user", "content": entries[i]},
            ]
        )

    classifications: List[List[str]] = get_answers(
        prompts=classification_prompts,
        default_response=[],
        response_type="structured",
        model=model,
        api_key=api_key,
        api_pipeline=api_pipeline,
        show_progress_bar=False,
    )

    final_classifications = []
    for one_entry_classifications in classifications:
        one_entry_classifications = [
            f"{tagname_level1}->{tagname_level2}->{one_problem}"
            for one_problem in one_entry_classifications
        ]
        final_classifications.append(one_entry_classifications)

    for i, classification in zip(ids_of_entries_with_tag, final_classifications):
        classifications_of_level1[i] = classification

    # save the classifications of the level1
    with open(save_path, "w") as f:
        json.dump(classifications_of_level1, f)
