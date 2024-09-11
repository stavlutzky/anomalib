from typing import Any
from lightning import Callback
from lightning.pytorch import Trainer
from pathlib import Path
from anomalib.models import AnomalyModule
import os
import pickle
from datetime import datetime


class SaveResults(Callback):

    def __init__(self, results_path, dataset_name, category_name, experiment_name) -> None:
        super().__init__()
        self.results_path = results_path
        self.dataset_name = dataset_name
        self.category_name = category_name
        self.experiment_name = experiment_name
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

        # Get the current timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # Create the file name with the timestamp
        file_name = f'output_{timestamp}.pkl'

        # Define the path and file name
        full_path = Path(self.results_path).joinpath(self.dataset_name).joinpath(self.category_name).joinpath(self.experiment_name).joinpath("classfication_pickles")
        # Create the directory if it doesn't exist
        if not os.path.exists(full_path):
            os.makedirs(full_path)

        # Open the file and write data to it
        with open(full_path.joinpath(f"{file_name}"), 'wb') as file:
            pickle.dump(outputs, file)