# Humanitarian Classification Framework - Inference

This repository provides the tools necessary for performing inference using classification models trained on the **Humanitarian Classification Framework**. The models were trained on the [HumSet dataset](https://aclanthology.org/2022.findings-emnlp.321/) and leverage debiased classification techniques developed as part of a custom training pipeline ([Bias Measurement and Mitigation for Humanitarian Text Classification](https://github.com/sfekih/bias-measurement-mitigation-humanitarian-text-classification)).

## Overview

The **Humanitarian Classification Framework** is designed to classify humanitarian texts into predefined categories to support analysts, organizations, and decision-makers in the humanitarian sector. By utilizing debiased models, the framework ensures that classification results are fairer and less influenced by inherent biases in the data.

## Features

- **Pre-trained models** on humanitarian data (HumSet).
- **Debiasing techniques** integrated into the model pipeline to enhance fairness and robustness.
- **Easy-to-use inference scripts** for fast classification of humanitarian documents.
- **Auto model download** — the HumBERT model is fetched automatically on first use.

## Installation

```bash
pip install .
```

Or directly from the repository:

```bash
pip install git+https://github.com/MediaMonitoringAndAnalysis/humanitarian-extract-classificator
```

The HumBERT model (`humbert_debiased.pt`) is downloaded automatically to `~/.cache/humanitarian_extract_classificator/` on first use. No manual download required.

## Usage

```python
from humanitarian_extract_classificator import humbert_classification
from typing import List

test_examples: List[str] = [
    "This is a test sentence",
    "This is another test sentence",
]

# prediction_ratio=1.0 maximises F1 score
# prediction_ratio>1.0 favours precision
# prediction_ratio<1.0 favours recall
outputs = humbert_classification(test_examples, prediction_ratio=1.0)
print(outputs)
```

### Level-2 classification (requires OpenAI API key)

```python
from humanitarian_extract_classificator import (
    humbert_classification,
    level2_classification,
    level2_problems_classification,
)
import os

os.environ["OPENAI_API_KEY"] = "..."

level1 = humbert_classification(test_examples)
level2 = level2_classification(entries=test_examples, level1_classifications=level1)
problems = level2_problems_classification(entries=test_examples, level2_classifications=level2)
```

## License

This repository is licensed under the AGPL v3. You are free to use, modify, and distribute the code as long as you comply with the terms of this license. Any derivative works must also be open-sourced under the same license.

For questions, support, or access to the models, please reach out to `selimfek@gmail.com`.
