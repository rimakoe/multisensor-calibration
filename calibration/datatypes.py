import json, os
import numpy as np
import pandas as pd
import scipy as sp
from scipy.spatial.transform import Rotation
from utils import *
from PIL import Image
from pypcd4 import PointCloud


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
        return {
            "x": t[0],
            "y": t[1],
            "z": t[2],
            "roll": r[0],
            "pitch": r[1],
            "yaw": r[2],
        }

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
                "depth": depth,
                "path": "/".join([f.name for f in current_path]),
            }
        )
        if type(frame) is Marker:
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

    class Intrinsics:
        pass  # TODO clarify the camera intrinsics

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

    def __init__(
        self,
        name: str,
        data: Image = None,
        features: pd.DataFrame = None,
        parent: Frame = None,
        transform: SE3 = SE3(),
    ):
        super().__init__(name, data, parent, transform)
        self.features = features


class Marker(Frame):
    __template = "\n".join(
        [
            """    <model name="{marker_name}_marker_{marker_id}">""",
            """        <static>true</static>""",
            """        <pose>{x} {y} {z} {roll} {pitch} {yaw}</pose>""",
            """        <xacro:marker_macro name="marker" id="{marker_id}" parent_link="{marker_name}_link" x="{x_marker}" y="{y_marker}" z="{z_marker}" roll="{roll_marker}" pitch="{pitch_marker}" yaw="{yaw_marker}"/>""",
            """    </model>""",
            """""",
        ]
    )

    def __init__(
        self,
        id: int,
        name: str,
        parent: Frame = None,
        transform: SE3 = SE3(),
    ):
        super().__init__(name, parent, transform)
        self.id = id

    def as_xacro(self) -> str:
        r = self.parent.transform.rotation.as_euler("xyz")
        r_marker = self.transform.rotation.as_euler("xyz")
        return self.__template.format(
            marker_name=self.name,
            marker_id=self.id,
            x=self.parent.transform.translation[0],
            y=self.parent.transform.translation[1],
            z=self.parent.transform.translation[2],
            roll=r[0],
            pitch=r[1],
            yaw=r[2],
            x_marker=self.transform.translation[0],
            y_marker=self.transform.translation[1],
            z_marker=self.transform.translation[2],
            roll_marker=r_marker[0],
            pitch_marker=r_marker[1],
            yaw_marker=r_marker[2],
        )


class Plane(Frame):
    __template = "\n".join(
        [
            """    <model name="{name}">""",
            """         <static>true</static>""",
            """         <pose>{x} {y} {z} {roll} {pitch} {yaw}</pose>""",
            """         <link name="{name}_link">""",
            """             <visual name="{name}_visual">""",
            """             <pose>0.0 0.0 0.0 0.0 0.0 0.0</pose>""",
            """                 <geometry>""",
            """                     <box>""",
            """                         <size>1.0 1.0 0.02</size>""",
            """                     </box>""",
            """                 </geometry>""",
            """             <xacro:set_material color="0.0 0.0 1.0" />""",
            """             </visual>""",
            """         </link>""",
            """    </model>""",
            """""",
        ]
    )

    def __init__(
        self,
        name: str,
        parent: Frame = None,
        transform: SE3 = SE3(),
    ):
        super().__init__(name, parent, transform)

    def get_normal(self):
        return self.transform.rotation.apply(np.array([0, 0, 1]))

    def as_xacro(self):
        euler = self.transform.rotation.as_euler("xyz")
        return self.__template.format(
            name=self.name,
            x=self.transform.translation[0],
            y=self.transform.translation[1],
            z=self.transform.translation[2],
            roll=euler[0],
            pitch=euler[1],
            yaw=euler[2],
        )


class XACRO:
    __prefix = "\n".join(
        [
            """<?xml version="1.0"?>""",
            """    <robot xmlns:xacro="http://www.ros.org/wiki/xacro">""",
            """    <xacro:include filename="utils.xacro" />""",
            """""",
        ]
    )

    __suffix = """</robot>"""

    def __init__(self, origin: SE3):
        self.origin = origin
        with open(os.path.join(os.getcwd(), "ros2_ws", "src", "simulation", "worlds", "empty.urdf.xacro")) as f:
            self.empty_file = f.read()

    def export(self):
        output = ""
        # output += self.__prefix
        for child in self.origin.children:
            if type(child) is not Plane:
                continue
            output += child.as_xacro()
            for marker in child.children:
                if type(marker) is not Marker:
                    continue
                output += marker.as_xacro()
        # output += self.__suffix
        autogen = self.empty_file.replace("<!-- OBJECTS -->", output)
        with open(os.path.join(os.getcwd(), "ros2_ws", "src", "simulation", "worlds", "autogen.urdf.xacro"), "w") as f:
            f.write(autogen)


def to_obc(root: Frame):
    content = []
    content.append("\t".join(["x", "y", "z"]) + "\n")
    rows = []
    for frame1 in root.children:
        if type(frame1) is Plane:
            for frame2 in frame1.children:
                if type(frame2) is Marker:
                    world_transform = frame1.transform @ frame2.transform
                    r = world_transform.rotation.as_euler("xyz", degrees=True)
                    content.append(
                        "\t".join(
                            [
                                str(world_transform.translation[0]),
                                str(world_transform.translation[1]),
                                str(world_transform.translation[1]),
                            ]
                        )
                        + "\n"
                    )
                    rows.extend(
                        [
                            {
                                #                "name": node.name,
                                "path": "/".join([frame1.name, frame2.name]),
                                "id": frame2.id,
                                "x": world_transform.translation[0],
                                "y": world_transform.translation[1],
                                "z": world_transform.translation[2],
                                "roll": r[0],
                                "pitch": r[1],
                                "yaw": r[2],
                                "depth": 2,
                            }
                        ]
                    )
    with open(os.path.join(os.getcwd(), "ros2_ws", "src", "simulation", "worlds", "autogen.obc"), "w") as f:
        f.writelines(content)
    df = pd.DataFrame(rows)
    print(df)
    return content


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

    world = Frame(name="world")
    plane = Plane(
        name="p1",
        transform=SE3(translation=np.array([2.0, 0.0, 1.0]), rotation=Rotation.from_euler("xyz", [0.0, -90.0, 0.0], degrees=True)),
    )
    plane.add_child(
        Marker(
            name="0",
            id=0,
            transform=SE3(translation=np.array([-0.25, 0.0, 0.02]), rotation=Rotation.from_euler("xyz", [0.0, 0.0, 0.0], degrees=True)),
        )
    )
    plane.add_child(
        Marker(
            name="1",
            id=1,
            transform=SE3(translation=np.array([0.25, 0.0, 0.02]), rotation=Rotation.from_euler("xyz", [0.0, 0.0, 0.0], degrees=True)),
        )
    )
    plane.add_child(
        Marker(
            name="2",
            id=2,
            transform=SE3(translation=np.array([0.0, 0.0, 0.02]), rotation=Rotation.from_euler("xyz", [0.0, 0.0, 0.0], degrees=True)),
        )
    )
    plane.add_child(
        Marker(
            name="3",
            id=3,
            transform=SE3(translation=np.array([0.0, 0.25, 0.02]), rotation=Rotation.from_euler("xyz", [0.0, 0.0, 0.0], degrees=True)),
        )
    )
    plane.add_child(
        Marker(
            name="4",
            id=4,
            transform=SE3(translation=np.array([0.0, -0.25, 0.02]), rotation=Rotation.from_euler("xyz", [0.0, 0.0, 0.0], degrees=True)),
        )
    )
    plane.add_child(
        Marker(
            name="5",
            id=5,
            transform=SE3(translation=np.array([0.25, 0.25, 0.02]), rotation=Rotation.from_euler("xyz", [0.0, 0.0, 0.0], degrees=True)),
        )
    )
    world.add_child(plane)
    xacro = XACRO(world)
    xacro.export()
    to_obc(world)
    print(world.as_dataframe(only_leafs=False, relative_coordinates=True))
