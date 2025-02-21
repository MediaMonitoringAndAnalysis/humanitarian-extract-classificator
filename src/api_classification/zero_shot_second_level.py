from typing import List, Dict
import os
import json
from openai_multiproc_inference import get_answers



classification_system_prompt = """
You are a humanitarian expert.
You are given a text and a JSON dictionary where the keys are the tags and the values are the definitions of the tags.
%s
You need to classify the text into the tags that are present in the definition file.
The output is a JSON List. The task is a multilabel classification task.
The output is a list with no entries, one entry, or multiple entries.
If unsure about a tag, return it anyways (high recall is crucial).
"""


def _generate_classification_prompts(
    entries: List[str],
    level1_classification: List[List[str]],
    level2_definitions: Dict[str, str],
    save_path: os.PathLike,
    api_key: str,
    pipeline: str = "OpenAI",
    model: str = "gpt-4o-mini",
):

    classifications_of_level1 = [[] for _ in entries]

    ids_of_entries_with_tag = [
        i
        for i, entry_tags in enumerate(level1_classification)
        if level1_classification in entry_tags
    ]

    classification_prompts = []
    for i in ids_of_entries_with_tag:
        classification_prompts.append(
            [
                {
                    "role": "system",
                    "content": classification_system_prompt % level2_definitions,
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
        pipeline=pipeline,
    )

    for i, classification in zip(ids_of_entries_with_tag, classifications):
        classifications_of_level1[i] = classification

    # save the classifications of the level1
    with open(save_path, "w") as f:
        json.dump(classifications_of_level1, f)
