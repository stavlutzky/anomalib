"""Callback that attaches necessary pre/post-processing to the model."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


from typing import Any

import torch
from lightning import Callback
from lightning.pytorch import Trainer
from lightning.pytorch.utilities.types import STEP_OUTPUT

from anomalib.data.utils import boxes_to_anomaly_maps, boxes_to_masks, masks_to_boxes
from anomalib.models import AnomalyModule
import os
import pickle

class _PostProcessorCallback(Callback):
    """Applies post-processing to the model outputs.

    Note: This callback is set within the Engine.
    """

    def __init__(self,predict_path=None) -> None:
        super().__init__()
        self.predict_path = predict_path

    def on_validation_batch_end(
        self,
        trainer: Trainer,
        pl_module: AnomalyModule,
        outputs: STEP_OUTPUT | None,
        batch: Any,  # noqa: ANN401
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        del batch, batch_idx, dataloader_idx  # Unused arguments.

        if outputs is not None:
            self.post_process(trainer, pl_module, outputs)

    def on_test_batch_end(
        self,
        trainer: Trainer,
        pl_module: AnomalyModule,
        outputs: STEP_OUTPUT | None,
        batch: Any,  # noqa: ANN401
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        del batch, batch_idx, dataloader_idx  # Unused arguments.

        if outputs is not None:
            self.post_process(trainer, pl_module, outputs)

    def on_predict_batch_end(
        self,
        trainer: Trainer,
        pl_module: AnomalyModule,
        outputs: Any,  # noqa: ANN401
        batch: Any,  # noqa: ANN401
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        del batch, batch_idx, dataloader_idx  # Unused arguments.

        if outputs is not None:
            self.post_process(trainer, pl_module, outputs)


        from datetime import datetime

        # Get the current timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # Create the file name with the timestamp
        file_name = f'output_{timestamp}.pkl'


        # Define the path and file name
        full_path = self.predict_path+f'\\{file_name}'

        # Create the directory if it doesn't exist
        if not os.path.exists(self.predict_path):
            os.makedirs(self.predict_path)
            print(f"Directory '{self.predict_path}' created.")

        # Open the file and write data to it
        with open(full_path, 'wb') as file:
            pickle.dump(outputs, file)
            print(f"Data written to '{full_path}'.")

        # with open(full_path, 'rb') as file:
        #     data = pickle.load(file)

    def post_process(self, trainer: Trainer, pl_module: AnomalyModule, outputs: STEP_OUTPUT) -> None:
        if isinstance(outputs, dict):
            self._post_process(outputs)
            if trainer.predicting or trainer.testing:
                self._compute_scores_and_labels(pl_module, outputs)

    @staticmethod
    def _compute_scores_and_labels(
        pl_module: AnomalyModule,
        outputs: dict[str, Any],
    ) -> None:
        if "pred_scores" in outputs:
            outputs["pred_labels"] = outputs["pred_scores"] >= pl_module.image_threshold.value
        if "anomaly_maps" in outputs:
            outputs["pred_masks"] = outputs["anomaly_maps"] >= pl_module.pixel_threshold.value
            if "pred_boxes" not in outputs:
                outputs["pred_boxes"], outputs["box_scores"] = masks_to_boxes(
                    outputs["pred_masks"],
                    outputs["anomaly_maps"],
                )
                outputs["box_labels"] = [torch.ones(boxes.shape[0]) for boxes in outputs["pred_boxes"]]
        # apply thresholding to boxes
        if "box_scores" in outputs and "box_labels" not in outputs:
            # apply threshold to assign normal/anomalous label to boxes
            is_anomalous = [scores > pl_module.pixel_threshold.value for scores in outputs["box_scores"]]
            outputs["box_labels"] = [labels.int() for labels in is_anomalous]

    @staticmethod
    def _post_process(outputs: STEP_OUTPUT) -> None:
        """Compute labels based on model predictions."""
        if isinstance(outputs, dict):
            if "pred_scores" not in outputs and "anomaly_maps" in outputs:
                # infer image scores from anomaly maps
                outputs["pred_scores"] = (
                    outputs["anomaly_maps"]  # noqa: PD011
                    .reshape(outputs["anomaly_maps"].shape[0], -1)
                    .max(dim=1)
                    .values
                )
            elif "pred_scores" not in outputs and "box_scores" in outputs and "label" in outputs:
                # infer image score from bbox confidence scores
                outputs["pred_scores"] = torch.zeros_like(outputs["label"]).float()
                for idx, (boxes, scores) in enumerate(zip(outputs["pred_boxes"], outputs["box_scores"], strict=True)):
                    if boxes.numel():
                        outputs["pred_scores"][idx] = scores.max().item()

            if "pred_boxes" in outputs and "anomaly_maps" not in outputs:
                # create anomaly maps from bbox predictions for thresholding and evaluation
                image_size: tuple[int, int] = outputs["image"].shape[-2:]
                pred_boxes: torch.Tensor = outputs["pred_boxes"]
                box_scores: torch.Tensor = outputs["box_scores"]

                outputs["anomaly_maps"] = boxes_to_anomaly_maps(pred_boxes, box_scores, image_size)

                if "boxes" in outputs:
                    true_boxes: list[torch.Tensor] = outputs["boxes"]
                    outputs["mask"] = boxes_to_masks(true_boxes, image_size)
