from main import (
    humbert_classification,
    level2_classification,
    level2_problems_classification,
)
import json


if __name__ == "__main__":

    with open("data/test_inputs/example_excerpts.json", "r") as f:
        test_examples = json.load(f)

    test_examples = test_examples[:10]

    # normal pred ratio is 1.0, but we set it to 0.000000001 to run predictions on the test examples
    humbert_outputs = humbert_classification(
        test_examples, prediction_ratio=0.9
    )
    print(humbert_outputs)

    level2_outputs = level2_classification(
        entries=test_examples, level1_classifications=humbert_outputs
    )

    print(level2_outputs)
    
    level2_problems_outputs = level2_problems_classification(
        entries=test_examples, level2_classifications=level2_outputs
    )

    print(level2_problems_outputs)
