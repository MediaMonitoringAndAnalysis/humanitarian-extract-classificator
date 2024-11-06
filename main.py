from flask import Flask, request, jsonify
from typing import List
import os
import sys
import torch

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import logging

# torch._logging.set_logs(dynamo = logging.INFO)

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Enable anomaly detection
torch.autograd.set_detect_anomaly(True)


# Import and initialize the classification pipeline
from inference import ClassificationInference



def hum_classification(text: List[str], batch_size: int = 32, prediction_ratio: float = 1.00):
    
    classification_pipeline = ClassificationInference()

    classification_results: List[List[str]] = classification_pipeline(
        text, batch_size, prediction_ratio
    )

    return classification_results


