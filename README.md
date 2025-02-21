# Humanitarian Classification Framework - Inference

This repository provides the tools necessary for performing inference using classification models trained on the **Humanitarian Classification Framework**. The models were trained on the [HumSet dataset](https://aclanthology.org/2022.findings-emnlp.321/) and leverage debiased classification techniques developed as part of a custom training pipeline ([Bias Measurement and Mitigation for Humanitarian Text Classification](https://github.com/sfekih/bias-measurement-mitigation-humanitarian-text-classification)).

## Overview

The **Humanitarian Classification Framework** is designed to classify humanitarian texts into predefined categories to support analysts, organizations, and decision-makers in the humanitarian sector. By utilizing debiased models, the framework ensures that classification results are fairer and less influenced by inherent biases in the data.

This repository provides the resources to load and run the classification models on new, unseen humanitarian extracts (of 1 to 4 sentences).

## Features

- **Pre-trained models** on humanitarian data (HumSet).
- **Debiasing techniques** integrated into the model pipeline to enhance fairness and robustness.
- **Easy-to-use inference scripts** for fast classification of humanitarian documents.

## How to Use

1. **Clone the Repository**

   First, clone the repository to your local machine:

   ```bash
   git clone https://github.com/yourusername/humanitarian-classification-inference.git
   cd humanitarian-classification-inference
   ```

2. **Install Dependencies**

   The repository includes a requirements.txt file to help set up the necessary Python packages:

   ```
   pip install -r requirements.txt
   ```
   
3. **Load Pre-trained Models**
Pre-trained models are not included in the repository. To access the models, please contact `selimfek@gmail.com``.

After obtaining the models, place them in the main repository folder.

4. **Inference**
for inference:

```
from main import humbert_classification
from typing import List


test_examples: List[str] = [
    "This is a test sentence",
    "This is another test sentence",
]

# The prediction_ratio is the decision boundary between the ratio between the predicte dprobability ands the optimal decision boundary threshold. 
# predicion_ratio=1.0 will maximize the F1 score
# predicion_rati>1.0 will favor precision.
# predicion_rati<1.0 will favor recall.
predicion_ratio: float = 1.0

outputs = humbert_classification(test_examples, prediction_ratio=predicion_ratio)
```


## TODO: add readme for level2 classification
...

## License

This repository is licensed under the AGPL v3. You are free to use, modify, and distribute the code as long as you comply with the terms of this license. Any derivative works must also be open-sourced under the same license.

For questions, support, or access to the models, please reach out to `selimfek@gmail.com`.