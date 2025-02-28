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


def _generate_one_label_zero_shot_classification_results(
    entries: List[str],
    tagname_level1: str,
    level2_classifications: List[List[str]],
    level2_definitions: Dict[str, str],
    save_path: os.PathLike,
    api_key: str,
    api_pipeline: str = "OpenAI",
    model: str = "gpt-4o-mini",
):

    classifications_of_level1 = [[] for _ in entries]

    ids_of_entries_with_tag = [
        i
        for i, entry_tags in enumerate(level2_classifications)
        if str(tagname_level1) in str(entry_tags)
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
        api_pipeline=api_pipeline,
        show_progress_bar=False,
    )
    
    final_classifications = []
    for one_entry_classifications in classifications:
        one_entry_classifications = [f"{tagname_level1}->{one_tag}" for one_tag in one_entry_classifications]
        final_classifications.append(one_entry_classifications)
    
    for i, classification in zip(ids_of_entries_with_tag, final_classifications):
        classifications_of_level1[i] = classification

    # save the classifications of the level1
    with open(save_path, "w") as f:
        json.dump(classifications_of_level1, f)


