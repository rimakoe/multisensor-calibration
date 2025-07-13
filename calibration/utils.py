from pydantic import BaseModel
from typing import List, Dict
from pydantic import BaseModel, field_validator
import numpy as np
from scipy.spatial.transform import Rotation
import pandas as pd
from pypcd4 import PointCloud
from PIL import Image


class ObjectDict(BaseModel):
    pass


class Solution(BaseModel):
    class SensorDict(BaseModel):
        class ExtrinsicsDict(BaseModel):
            class TranslationDict(BaseModel):
                x: float
                y: float
                z: float

                def as_array(self) -> np.ndarray:
                    return np.array([self.x, self.y, self.z])

            translation: TranslationDict

            class RotationDict(BaseModel):
                i: float
                j: float
                k: float
                w: float

                def as_transform(self) -> Rotation:
                    return Rotation.from_quat([self.i, self.j, self.k, self.w])

            rotation: RotationDict

        extrinsics: ExtrinsicsDict

    devices: Dict[str, SensorDict]
    objects: Dict[str, ObjectDict]


def read_pcd(filepath: str) -> PointCloud:
    # TODO
    return PointCloud


def read_obc(filepath: str) -> pd.DataFrame:
    # TODO
    return pd.DataFrame()


def read_bmp(filepath: str) -> Image:
    return Image.open(filepath)
