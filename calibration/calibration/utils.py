import os
from typing import Dict
from pydantic import BaseModel
from scipy.spatial.transform import Rotation
import numpy as np
import pandas as pd

workspace_directory = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))


def get_dataset_directory():
    return os.path.join(workspace_directory, "datasets")


class ObjectDict(BaseModel):
    pass


class Solution(BaseModel):
    class SensorDict(BaseModel):
        class ExtrinsicsDict(BaseModel):
            class TranslationDict(BaseModel):
                x: float
                y: float
                z: float

                def to_numpy(self) -> np.ndarray:
                    return np.array([self.x, self.y, self.z])

            translation: TranslationDict

            class RotationDict(BaseModel):
                i: float
                j: float
                k: float
                w: float

                def to_scipy(self) -> Rotation:
                    return Rotation.from_quat([self.i, self.j, self.k, self.w])

            rotation: RotationDict

        extrinsics: ExtrinsicsDict

    devices: Dict[str, SensorDict]
    objects: Dict[str, ObjectDict]


def write_obc(dataframe: pd.DataFrame, output_filepath: str):
    required_columns = ["id", "x", "y", "z"]
    missing = [column for column in required_columns if column not in dataframe.columns]
    if missing:
        print(f"Can't export DataFrame to OBC - Missing columns: {missing}")
        return
    dataframe[required_columns].to_csv(path_or_buf=output_filepath, sep="\t", index=False)


def read_obc(filepath: str) -> pd.DataFrame:
    return pd.read_csv(filepath_or_buffer=filepath, sep="\t")


def skew(v: np.ndarray):
    assert v.size == 3
    assert v.shape == (3,) or v.shape == (3, 1)
    return np.array(
        [
            [0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0],
        ]
    )
