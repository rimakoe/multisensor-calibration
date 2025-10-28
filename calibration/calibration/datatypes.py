import json, os
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation
from PIL import Image
from pypcd4 import PointCloud
from typing import List
import g2opy.g2opy as g2o
from calibration.utils import *


class Transform:
    """This is an object that stores the rotation and translation of a rigid body.
    It is intended to be used inside a Frame and stores a scipy rotation object that can be represented as euler, matrix, quaternion and rotation vector and a numpy array as translational vector.
    """

    def __init__(
        self,
        rotation: Rotation = Rotation.from_euler("xyz", [0, 0, 0], degrees=True),
        translation: np.ndarray = np.array([0.0, 0.0, 0.0]),
    ):
        """
        Parameters
        -----------
            rotation : Rotation
                Scipy datatype to represent rotations in every possible way (matrix, euler, quaternion, rotation vector)
            translation : np.ndarray
                Numpy array that stores the translational offset in euclidean coordinates
        """
        assert type(rotation) is Rotation, "rotation is expected to be scipy Rotation object"
        assert type(translation) is np.ndarray, "translation is expected to be a numpy array"
        assert translation.shape == (3, 1) or translation.shape == (3,), "translation is expected to be a vector with 3 entries"
        self.rotation = rotation
        self.translation = translation

    @classmethod
    def from_matrix(cls, matrix: np.ndarray):
        assert matrix.shape == (4, 4)
        rotation = Rotation.from_matrix(matrix[:3, :3])
        translation = matrix[:3, 3]
        return cls(rotation, translation)

    @classmethod
    def from_dict(self, extrinsics: Solution.SensorDict.ExtrinsicsDict) -> "Transform":
        return Transform(
            translation=Solution.SensorDict.ExtrinsicsDict.TranslationDict.to_numpy(extrinsics.translation),
            rotation=Solution.SensorDict.ExtrinsicsDict.RotationDict.to_scipy(extrinsics.rotation),
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
        return Transform(rotation=self.rotation.inv(), translation=-self.rotation.inv().apply(self.translation))

    def apply(self, other: np.ndarray):
        required_columns = ["x", "y", "z"]
        if type(other) is pd.DataFrame:
            missing = [required_column for required_column in required_columns if required_column not in other.columns]
            assert len(missing) == 0, f"missing column(s): {missing}"
            other = other[required_columns]
        if other.shape == (3,):
            return self.rotation.apply(other) + self.translation
        applied = self.rotation.apply(other) + self.translation
        if type(other) is pd.DataFrame:
            return pd.DataFrame(applied, columns=required_columns)
        return applied

    def __matmul__(self, other: "Transform") -> "Transform":
        r_new = Rotation.from_matrix(self.rotation.as_matrix() @ other.rotation.as_matrix())
        t_new = self.rotation.apply(other.translation) + self.translation
        return Transform(rotation=r_new, translation=t_new)

    def __repr__(self):
        t: np.ndarray = np.round(self.translation, 3)
        r: np.ndarray = np.round(self.rotation.as_euler("xyz", degrees=True), 3)
        return f"SE3(t={t.tolist()},\tr={r.tolist()})\n"

    def __str__(self):
        t: np.ndarray = np.round(self.translation, 3)
        r: np.ndarray = np.round(self.rotation.as_euler("xyz", degrees=True), 3)
        return f"{t.tolist()} | {r.tolist()}\n"

    def adjoint(self):
        assert self.rotation is not None
        assert self.translation is not None
        R = self.rotation.as_matrix()
        adj = np.zeros(shape=(6, 6))
        adj[:3, :3] = R
        adj[3:, 3:] = R
        adj[3:, :3] = skew(self.translation) @ R
        return adj


class Frame:
    def __init__(self, name: str, parent: "Frame" = None, transform: Transform = Transform(), covariance: np.ndarray = np.zeros((6, 6))):
        assert covariance.shape == (6, 6)
        self.name: str = name
        self.parent: "Frame" = parent
        self.children: list["Frame"] = []
        self.covariance: np.ndarray = covariance
        self.transform: Transform = transform

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
            transform = Transform()
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

    def get_frame(self, name: str, frame: "Frame" = None) -> "Frame":
        if frame is None:
            frame = self
        if name == frame.name:
            return frame
        for child in frame.children:
            result = self.get_frame(name, child)
            if result:
                return result
        return None

    def as_dataframe(self, only_leafs: bool = False, relative_coordinates: bool = True, degrees: bool = True) -> pd.DataFrame:
        return pd.DataFrame(self.flatten(only_leafs=only_leafs, relative_coordinates=relative_coordinates, degrees=degrees)).round(4)

    def __repr__(self):
        r: np.ndarray = self.transform.rotation.as_euler("xyz", degrees=True)
        sigmas = np.sqrt(np.diag(self.covariance))
        sigma_t: np.ndarray = sigmas[3:]
        sigma_r: np.ndarray = np.rad2deg(Rotation.from_rotvec(sigmas[:3]).as_euler("xyz"))
        return pd.DataFrame(
            {
                "x": [self.transform.translation[0]],
                "y": [self.transform.translation[1]],
                "z": [self.transform.translation[2]],
                "roll": [r[0]],
                "pitch": [r[1]],
                "yaw": [r[2]],
                "sx": [sigma_t[0]],
                "sy": [sigma_t[1]],
                "sz": [sigma_t[2]],
                "sroll": [sigma_r[0]],
                "spitch": [sigma_r[1]],
                "syaw": [sigma_r[2]],
            }
        )


class Vehicle(Frame):
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
    def __init__(
        self,
        name: str,
        data: np.ndarray = None,
        parent: Frame = None,
        transform: Transform = Transform(),
    ):
        super().__init__(name, parent, transform)
        self.data = data

    @classmethod
    def from_dict(self, name: str, data: Solution.SensorDict) -> "Sensor":
        return Sensor(
            name=name,
            translation=Solution.SensorDict.ExtrinsicsDict.TranslationDict.to_numpy(data.extrinsics.translation),
            rotation=Solution.SensorDict.ExtrinsicsDict.RotationDict.to_scipy(data.extrinsics.rotation),
        )

    def __repr__(self):
        r: np.ndarray = self.transform.rotation.as_euler("xyz", degrees=True)
        sigmas = np.sqrt(np.diag(self.covariance))
        sigma_t: np.ndarray = sigmas[3:]
        sigma_r: np.ndarray = np.rad2deg(Rotation.from_rotvec(sigmas[:3]).as_euler("xyz"))
        return pd.DataFrame(
            {
                "x": [self.transform.translation[0]],
                "y": [self.transform.translation[1]],
                "z": [self.transform.translation[2]],
                "roll": [r[0]],
                "pitch": [r[1]],
                "yaw": [r[2]],
                "sx": [sigma_t[0]],
                "sy": [sigma_t[1]],
                "sz": [sigma_t[2]],
                "sroll": [sigma_r[0]],
                "spitch": [sigma_r[1]],
                "syaw": [sigma_r[2]],
            }
        ).to_string()


class Marker(Frame):
    def __init__(
        self,
        id: int,
        name: str,
        parent: Frame = None,
        transform: Transform = Transform(),
    ):
        super().__init__(name, parent, transform)
        self.id = id


class Plane(Frame):
    def __init__(
        self,
        name: str,
        parent: Frame = None,
        transform: Transform = Transform(),
    ):
        super().__init__(name, parent, transform)

    @classmethod
    def from_normal_offset(self, name: str, normal: np.ndarray, offset: np.ndarray, parent: Frame = None):
        normal /= np.linalg.norm(normal, 2)
        arbitrary_vector = np.array([0.0, 1.0, 0.0])
        if np.allclose(normal, arbitrary_vector):
            arbitrary_vector = np.array([1.0, 0.0, 0.0])
        z = normal
        x = np.cross(arbitrary_vector, z)
        x /= np.linalg.norm(x, 2)
        y = np.cross(z, x)
        y /= np.linalg.norm(y, 2)
        return Plane(name, parent, transform=Transform(rotation=Rotation.from_matrix(np.stack([x, y, z], axis=1)), translation=offset))

    def get_offset(self):
        return self.transform.translation

    def get_normal(self):
        return self.transform.rotation.apply(np.array([0, 0, 1]))

    def as_cartesian(self):
        raise NotImplementedError("maybe nice to have that as well")

    def get_distance(self, point: np.ndarray):
        return np.dot(self.get_normal(), point) + self.get_offset()


class Lidar(Sensor):
    class Intrinsics:
        def __init__(self, scale: float = 1.0, offset: np.ndarray = np.zeros((3, 1))):
            self.scale = scale
            self.offset = offset

    def __init__(
        self,
        name: str,
        id: int,
        data: PointCloud = None,
        features: List[Plane] = None,
        parent: Frame = None,
        transform: Transform = Transform(),
        intrinsics: Intrinsics = None,
    ):
        super().__init__(name, data, parent, transform)
        assert id >= 1000, "Expecting ID of at least 1000 for a camera for safety when creating g2o graph."
        self.id = id
        self.features = features


class Camera(Sensor):
    class Intrinsics:
        def __init__(self, width: float, height: float, focal_length: np.ndarray, principal_point: np.ndarray, distortion: np.ndarray = None):
            assert focal_length.shape == (2, 1) or focal_length.shape == (2,)
            assert principal_point.shape == (2, 1) or principal_point.shape == (2,)
            assert height > 0
            assert width > 0
            self.width = width
            self.height = height
            self.focal_length = focal_length
            self.principal_point = principal_point
            self.distortion = distortion

        @classmethod
        def from_json(self, filepath: str):
            config = {}
            with open(filepath, "r") as f:
                config = json.load(f)
            focal_length = np.array([config["K"][0], config["K"][4]])
            principal_point = np.array([config["K"][2], config["K"][5]])
            distortion = np.array(config["d"])
            width = config["width"]
            height = config["height"]
            return Camera.Intrinsics(
                width=width,
                height=height,
                focal_length=focal_length,
                principal_point=principal_point,
                distortion=distortion,
            )

        def as_matrix(self):
            return np.array(
                [
                    [self.focal_length[0], 0.0, self.principal_point[0]],
                    [0.0, self.focal_length[1], self.principal_point[1]],
                    [0.0, 0.0, 1.0],
                ]
            )

    def __init__(
        self,
        name: str,
        id: int,
        data: Image = None,
        features: pd.DataFrame = None,
        parent: Frame = None,
        transform: Transform = Transform(),
        intrinsics: "Intrinsics" = None,
    ):
        super().__init__(name, data, parent, transform)
        assert id >= 1000, "Expecting ID of at least 1000 for a camera for safety when creating g2o graph."
        self.features = features
        if not intrinsics:
            self.intrinsics = Camera.Intrinsics(width=100, height=100, focal_length=np.array([1.0, 1.0]), principal_point=np.array([0.0, 0.0]))
        else:
            self.intrinsics = intrinsics
        self.id: int = id
        self.vertex: g2o.VertexSE3Expmap = None


if __name__ == "__main__":
    directory_to_solution = os.path.join(get_dataset_directory(), "C4L5", "solution.json")
    solution = Solution(**json.load(open(directory_to_solution)))
    vehicle = Vehicle(name="Vehicle", system_configuration="CL-0.5")
    sensor_id = 1000
    for name, data in solution.devices.items():
        initial_guess = Frame(name="InitialGuess", transform=Transform.from_dict(data.extrinsics))
        if "lidar" in name.lower():
            initial_guess.add_child(Lidar(name=name, id=sensor_id))
            vehicle.add_child(initial_guess)
            sensor_id += 1
        if "camera" in name.lower():
            initial_guess.add_child(Camera(name=name, id=sensor_id))
            vehicle.add_child(initial_guess)
            sensor_id += 1

    print(vehicle.as_dataframe(only_leafs=True, relative_coordinates=False))
