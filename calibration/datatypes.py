import json, os
import numpy as np
import pandas as pd
import scipy as sp
from scipy.spatial.transform import Rotation
from utils import *
from PIL import Image
from pypcd4 import PointCloud
from typing import Tuple
import g2opy.g2opy as g2o


class SE3:
    """
    TODO
    """

    def __init__(
        self,
        rotation: Rotation = Rotation.from_euler("xyz", [0, 0, 0], degrees=True),
        translation: np.ndarray = np.array([0.0, 0.0, 0.0]),
    ):
        self.rotation = rotation
        self.translation = translation

    @classmethod
    def from_matrix(cls, matrix: np.ndarray):
        assert matrix.shape == (4, 4)
        rotation = Rotation.from_matrix(matrix[:3, :3])
        translation = matrix[:3, 3]
        return cls(rotation, translation)

    @classmethod
    def from_dict(self, extrinsics: Solution.SensorDict.ExtrinsicsDict) -> "SE3":
        return SE3(
            translation=Solution.SensorDict.ExtrinsicsDict.TranslationDict.as_array(extrinsics.translation),
            rotation=Solution.SensorDict.ExtrinsicsDict.RotationDict.as_transform(extrinsics.rotation),
        )

    def as_matrix(self) -> np.ndarray:
        matrix = np.eye(4)
        matrix[:3, :3] = self.rotation.as_matrix()
        matrix[:3, 3] = self.translation
        return matrix

    def as_tuple(self):
        return self.rotation, self.translation

    def as_dict(self, degrees: bool = True):
        r = self.rotation.as_euler("xyz", degrees=degrees)
        t = self.translation
        return {"x": t[0], "y": t[1], "z": t[2], "roll": r[0], "pitch": r[1], "yaw": r[2]}

    def as_eigen(self) -> g2o.SE3Quat:
        """
        The g2o.SE3Quat is a python binding to the standard C++ Eigen library.
        I like it more to store the rotation as scipy Rotation object since this provides many nice features and is easier to use.
        Therefore we can just export our type to that binding here.
        """
        return g2o.SE3Quat(self.rotation.as_matrix(), self.translation)

    def inverse(self):
        return SE3(self.rotation.inv(), -self.rotation.inv().apply(self.translation))

    def __matmul__(self, other: "SE3") -> "SE3":
        r_new = Rotation.from_matrix(self.rotation.as_matrix() @ other.rotation.as_matrix())
        t_new = self.rotation.apply(other.translation) + self.translation
        return SE3(rotation=r_new, translation=t_new)

    def __repr__(self):
        t: np.ndarray = np.round(self.translation, 3)
        r: np.ndarray = np.round(self.rotation.as_euler("xyz", degrees=True), 3)
        return f"SE3(t={t.tolist()},\tr={r.tolist()})\n"


class Frame:
    """
    TODO
    """

    def __init__(self, name: str, parent: "Frame" = None, transform: SE3 = SE3()):
        self.name: str = name
        self.parent: "Frame" = parent
        self.children: list["Frame"] = []
        self.transform: SE3 = transform

    def add_child(self, child: "Frame"):
        if not issubclass(type(child), Frame):
            raise TypeError("Can't add a child that is not at least inheriting from Tree")
        child.parent = self
        self.children.append(child)

    def remove_child(self, child: "Frame"):
        try:
            self.children.remove(child)
        except:
            parent_name = "root"
            if self.parent is not None:
                parent_name = self.parent.name
            print(UserWarning(f"Cant remove child {self.name} from {parent_name}"))

    def flatten(
        self,
        frame: "Frame" = None,
        path: List["Frame"] = None,
        depth: int = 0,
        only_leafs: bool = False,
        relative_coordinates: bool = True,
        degrees: bool = True,
    ):
        if frame is None:
            frame = self
        if path is None:
            path = []
        current_path: List[Frame] = path + [frame]

        if not relative_coordinates:
            transform = SE3()
            for frame in current_path:
                transform = transform @ frame.transform
        else:
            transform = frame.transform

        rows = []
        d = transform.as_dict()
        d.update(
            {
                "name": frame.name,
                "depth": depth,
                "path": "/".join([f.name for f in current_path]),
            }
        )
        if issubclass(type(frame), Marker):
            d.update({"id": frame.id})

        row = [d]
        if only_leafs:
            if not frame.children:
                rows.extend(row)
        else:
            rows.extend(row)

        for child in frame.children:
            rows.extend(
                self.flatten(
                    frame=child,
                    path=current_path,
                    depth=depth + 1,
                    only_leafs=only_leafs,
                    relative_coordinates=relative_coordinates,
                    degrees=degrees,
                )
            )
        return rows

    def as_dataframe(self, only_leafs: bool = False, relative_coordinates: bool = True, degrees: bool = True) -> pd.DataFrame:
        return pd.DataFrame(self.flatten(only_leafs=only_leafs, relative_coordinates=relative_coordinates, degrees=degrees))


class Vehicle(Frame):
    """
    TODO
    """

    def __init__(self, name: str, system_configuration: str, parent: Frame = None):
        super().__init__(name=name, parent=parent)
        self.system_configuration = system_configuration

    def __repr__(self):
        s = ""
        for child in self.children:
            s = s + child.__repr__()
        return s

    def as_dict(self):
        d = super().as_dict()
        d["system_configuration"] = self.system_configuration
        return d


class Sensor(Frame):
    """
    TODO
    """

    def __init__(
        self,
        name: str,
        data: np.ndarray = None,
        parent: Frame = None,
        transform: SE3 = SE3(),
    ):
        super().__init__(name, parent, transform)
        self.data = data

    @classmethod
    def from_dict(self, name: str, data: Solution.SensorDict) -> "Sensor":
        return Sensor(
            name=name,
            translation=Solution.SensorDict.ExtrinsicsDict.TranslationDict.as_array(data.extrinsics.translation),
            rotation=Solution.SensorDict.ExtrinsicsDict.RotationDict.as_transform(data.extrinsics.rotation),
        )

    def __repr__(self):
        return f"{self.name}\t|\t({self.translation} / {self.rotation.as_euler("xyz", degrees=True)})\n"


class Lidar(Sensor):
    """
    TODO
    """

    def __init__(
        self,
        name: str,
        data: PointCloud = None,
        features: pd.DataFrame = None,
        parent: Frame = None,
        transform: SE3 = SE3(),
    ):
        super().__init__(name, data, parent, transform)
        self.features = features


class Camera(Sensor):
    """
    TODO
    """

    class Intrinsics:
        def __init__(self, focal_length: float, principal_point: Tuple[float, float], distortion: np.ndarray = None):
            self.focal_length = focal_length
            self.principal_point = principal_point
            self.distortion = distortion

    def __init__(
        self,
        name: str,
        data: Image = None,
        features: pd.DataFrame = None,
        parent: Frame = None,
        transform: SE3 = SE3(),
        intrinsics: "Intrinsics" = None,
    ):
        super().__init__(name, data, parent, transform)
        self.features = features
        if not intrinsics:
            self.intrinsics = Camera.Intrinsics(focal_length=1.0, principal_point=(0.0, 0.0))
        else:
            self.intrinsics = intrinsics


class Marker(Frame):
    def __init__(
        self,
        id: int,
        name: str,
        parent: Frame = None,
        transform: SE3 = SE3(),
    ):
        super().__init__(name, parent, transform)
        self.id = id


class Plane(Frame):
    def __init__(
        self,
        name: str,
        parent: Frame = None,
        transform: SE3 = SE3(),
    ):
        super().__init__(name, parent, transform)

    def get_normal(self):
        return self.transform.rotation.apply(np.array([0, 0, 1]))


if __name__ == "__main__":
    directory_to_solution = os.path.join(os.getcwd(), "sensor_data", "solution.json")
    solution = Solution(**json.load(open(directory_to_solution)))
    vehicle = Vehicle(name="Vehicle", system_configuration="CL-0.5")

    for name, data in solution.devices.items():
        initial_guess = Frame(name="InitialGuess", transform=SE3.from_dict(data.extrinsics))
        if "lidar" in name.lower():
            initial_guess.add_child(Lidar(name=name))
            vehicle.add_child(initial_guess)
        if "camera" in name.lower():
            initial_guess.add_child(Camera(name=name))
            vehicle.add_child(initial_guess)

    print(vehicle.as_dataframe(only_leafs=True, relative_coordinates=False))
