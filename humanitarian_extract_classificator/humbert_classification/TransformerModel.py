import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Dict, List
from transformers import AutoModel
from dataset_creation import ExcerptsDataset
import numpy as np
from pooling import Pooling


def get_tag_id_to_layer_id(ids_each_level):
    tag_id = 0
    list_id = 0
    tag_to_list = {}
    for id_list in ids_each_level:
        for i in range(len(id_list)):
            tag_to_list.update({tag_id + i: list_id})
        tag_id += len(id_list)
        list_id += 1
    return tag_to_list


def _flatten(ids_each_level):
    return [tag for id_one_level in ids_each_level for tag in id_one_level]


class TransformerArchitecture(torch.nn.Module):
    """
    base architecture, used for finetuning the transformer model.
    """

    def __init__(
        self,
        model_name_or_path,
        ids_each_level,
        dropout_rate: float,
        transformer_output_length: int,
        n_freezed_layers: int,
    ):
        super().__init__()
        self.ids_each_level = ids_each_level
        self.n_level0_ids = len(ids_each_level)
        self.n_heads = len(_flatten(self.ids_each_level))

        self.tag_id_to_layer_id = get_tag_id_to_layer_id(ids_each_level)

        self.common_backbone = AutoModel.from_pretrained(model_name_or_path)
        self.common_backbone.encoder.layer = self.common_backbone.encoder.layer[:-1]

        # freeze embeddings
        for param in self.common_backbone.embeddings.parameters():
            param.requires_grad = False
        # freeze n_freezed_layers first layers
        if n_freezed_layers > 0:
            for layer in self.common_backbone.encoder.layer[:n_freezed_layers]:
                for param in layer.parameters():
                    param.requires_grad = False

        self.pool = Pooling(
            word_embedding_dimension=transformer_output_length,
            pooling_mode_mean_tokens=True,
            pooling_mode_cls_token=True,
        )

        self.LayerNorm_specific_hidden = torch.nn.ModuleList(
            [
                torch.nn.LayerNorm(transformer_output_length * 2)
                for _ in range(self.n_level0_ids)
            ]
        )

        self.dropout = torch.nn.Dropout(dropout_rate)

        self.specific_layer = torch.nn.ModuleList(
            [
                AutoModel.from_pretrained(model_name_or_path).encoder.layer[-1]
                for _ in range(self.n_level0_ids)
            ]
        )

        self.output_layer = torch.nn.ModuleList(
            [
                torch.nn.Linear(transformer_output_length * 2, len(id_one_level))
                for id_one_level in _flatten(ids_each_level)
            ]
        )

        self.activation_function = nn.SELU()

    def forward(self, inputs):

        fith_layer_transformer_output = self.common_backbone(
            inputs["ids"],
            attention_mask=inputs["mask"],
        ).last_hidden_state

        encoder_outputs = [
            self.pool(
                {
                    "token_embeddings": self.specific_layer[i](
                        fith_layer_transformer_output.clone()
                    )[0],
                    "attention_mask": inputs["mask"],
                }
            )["sentence_embedding"]
            for i in range(self.n_level0_ids)
        ]

        if (
            "return_transformer_only" in inputs.keys()
            and inputs["return_transformer_only"]
        ):
            return torch.cat(encoder_outputs, dim=1)

        else:
            classification_heads = torch.cat(
                [
                    self.output_layer[tag_id](
                        self.LayerNorm_specific_hidden[self.tag_id_to_layer_id[tag_id]](
                            self.dropout(
                                self.activation_function(
                                    encoder_outputs[
                                        self.tag_id_to_layer_id[tag_id]
                                    ].clone()
                                )
                            )
                        )
                    )
                    for tag_id in range(self.n_heads)
                ],
                dim=1,
            )

            return classification_heads.cpu()


class LoggedTransformerModel(torch.nn.Module):
    """
    Logged transformers structure, done for space memory optimization
    Only contains needed varibles and functions for inference
    """

    def __init__(self, trained_model) -> None:
        super().__init__()
        self.trained_architecture = trained_model.model
        # self.trained_architecture.eval()
        # self.trained_architecture.freeze()
        trained_model.tokenizer.split_special_tokens = False
        self.tokenizer = trained_model.tokenizer
        self.tagname_to_tagid = trained_model.tagname_to_tagid
        self.val_params = trained_model.val_params

    def forward(self, inputs):
        output = self.trained_architecture(inputs)
        return output

    def get_loaders(self, dataset, params, max_len: int):
        set = ExcerptsDataset(dataset, self.tagname_to_tagid, self.tokenizer, max_len)
        loader = DataLoader(set, **params, pin_memory=True)
        return loader

    def custom_predict(
        self,
        dataset,
        max_len: int,
        prediction_ratio: float = 1.0,
    ):
        """
        1) get raw predictions
        2) postprocess them to output an output compatible with what we want in the inference
        """

        predictions_loader = self.get_loaders(dataset, self.val_params, max_len)

        logit_predictions = []

        with torch.inference_mode():
            # for batch in tqdm(
            #     predictions_loader,
            #     total=len(predictions_loader.dataset) // predictions_loader.batch_size,
            # ):
            for batch in predictions_loader:
                logits = self(
                    {
                        "ids": batch["ids"].to(self.testing_device),
                        "mask": batch["mask"].to(self.testing_device),
                        "return_transformer_only": False,
                    }
                ).cpu()
                logit_predictions.append(logits)

        if len(logit_predictions) > 0:
            logit_predictions = torch.cat(logit_predictions, dim=0)
        else:
            logit_predictions = torch.tensor([])

        logit_predictions = torch.sigmoid(logit_predictions)

        thresholds = np.array(list(self.optimal_thresholds.values()))
        final_predictions = logit_predictions.numpy() / thresholds

        outputs = [
            # {
            #     tagname: final_predictions[i, tagid]
            #     for tagname, tagid in self.tagname_to_tagid.items()
            # }
            [
                tagname
                for tagname, tagid in self.tagname_to_tagid.items()
                if final_predictions[i, tagid] >= prediction_ratio
            ]
            for i in range(logit_predictions.shape[0])
        ]

        return outputs
