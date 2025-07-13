import json, os
import numpy as np
import pandas as pd
import scipy as sp
from scipy.spatial.transform import Rotation
from utils import *
from PIL import Image
from pypcd4 import PointCloud


class Node:
    """
    1:N Tree Node Structure to handle the number of coordinate systems
    """

    def __init__(self, name: str, parent: "Node" = None):
        self.name: str = name
        self.parent: "Node" = parent
        self.children: list["Node"] = []

    def add_child(self, child: "Node"):
        if not issubclass(type(child), Node):
            raise TypeError("Can't add a child that is not at least inheriting from Tree")
        child.parent = self
        self.children.append(child)

    def remove_child(self, child: "Node"):
        try:
            self.children.remove(child)
        except:
            parent_name = "root"
            if self.parent is not None:
                parent_name = self.parent.name
            print(UserWarning(f"Cant remove child {self.name} from {parent_name}"))

    def remove(self):
        if self.parent is None:
            print(UserWarning("Current object is root node"))
            return
        self.parent.children.remove(self)


class SE3(Node):
    def __init__(
        self,
        name: str,
        parent: Node = None,
        rotation: Rotation = Rotation.from_euler("xyz", [0, 0, 0], degrees=True),
        translation: np.ndarray = np.array([0.0, 0.0, 0.0]),
    ):
        super().__init__(name=name, parent=parent)
        self.rotation = rotation
        self.translation = translation

    @classmethod
    def from_matrix(cls, mat: np.ndarray):
        assert mat.shape == (4, 4)
        rotation = Rotation.from_matrix(mat[:3, :3])
        translation = mat[:3, 3]
        return cls(rotation, translation)

    @classmethod
    def from_dict(self, name: str, extrinsics: Solution.SensorDict.ExtrinsicsDict) -> "SE3":
        return SE3(
            name=name,
            translation=Solution.SensorDict.ExtrinsicsDict.TranslationDict.as_array(extrinsics.translation),
            rotation=Solution.SensorDict.ExtrinsicsDict.RotationDict.as_transform(extrinsics.rotation),
        )

    def as_matrix(self) -> np.ndarray:
        mat = np.eye(4)
        mat[:3, :3] = self.rotation.as_matrix()
        mat[:3, 3] = self.translation
        return mat

    def as_tuple(self):
        return self.translation, self.rotation

    def inverse(self):
        return SE3(self.rotation.inv(), -self.rotation.inv().apply(self.translation))

    def __matmul__(self, other: "SE3") -> "SE3":
        r_new = Rotation.from_matrix(self.rotation.as_matrix() @ other.rotation.as_matrix())
        t_new = self.translation + self.rotation.apply(other.translation)
        return SE3(name=self.name + "@" + other.name, rotation=r_new, translation=t_new)

    def __repr__(self):
        t: np.ndarray = np.round(self.translation, 3)
        r: np.ndarray = np.round(self.rotation.as_euler("xyz", degrees=True), 3)
        return f"SE3(t={t.tolist()}, r={r.tolist()})\n"


class Vehicle(SE3):
    def __init__(self, name: str, system_configuration: str, parent=None):
        super().__init__(name=name, parent=parent)
        self.system_configuration = system_configuration
        self.recordings_directory = None

    def __repr__(self):
        s = ""
        for child in self.children:
            s = s + child.__repr__()
        return s

    def flatten(self, node=None, path=None, depth=0):
        if node is None:
            node = self
        if path is None:
            path = []
        current_path = path + [node.name]
        r = node.rotation.as_euler("xyz", degrees=True)
        rows = [
            {
                #                "name": node.name,
                "depth": depth,
                "x": node.translation[0],
                "y": node.translation[1],
                "z": node.translation[2],
                "roll": r[0],
                "pitch": r[1],
                "yaw": r[2],
                "path": "/".join(current_path),
            }
        ]
        for child in node.children:
            rows.extend(self.flatten(child, current_path, depth + 1))
        return rows

    def show(self):
        df = pd.DataFrame(vehicle.flatten())
        print(df.set_index("path"))


class Sensor(SE3):
    def __init__(
        self,
        name: str,
        data: np.ndarray = None,
        parent=None,
        rotation=Rotation.from_euler("xyz", [0, 0, 0], degrees=True),
        translation=np.array([0, 0, 0]),
    ):
        super().__init__(name, parent, rotation, translation)
        self.data = data

    class Intrinsics:
        pass  # TODO clarify the camera intrinsics here!

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
    def __init__(
        self,
        name: str,
        data: PointCloud = None,
        features: pd.DataFrame = None,
        parent=None,
        rotation=Rotation.from_euler("xyz", [0, 0, 0], degrees=True),
        translation=np.array([0, 0, 0]),
    ):
        super().__init__(name, data, parent, rotation, translation)


class Camera(Sensor):
    def __init__(
        self,
        name: str,
        data: Image = None,
        features: pd.DataFrame = None,
        parent=None,
        rotation=Rotation.from_euler("xyz", [0, 0, 0], degrees=True),
        translation=np.array([0, 0, 0]),
    ):
        super().__init__(name, data, parent, rotation, translation)


class Marker(SE3):
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
        parent=None,
        rotation=Rotation.from_euler("xyz", [0, 0, 0], degrees=True),
        translation=np.array([0, 0, 0]),
    ):
        super().__init__(name, parent, rotation, translation)
        self.id = id

    def as_xacro(self) -> str:
        r = self.parent.rotation.as_euler("xyz")
        r_marker = self.rotation.as_euler("xyz")
        return self.__template.format(
            marker_name=self.name,
            marker_id=self.id,
            x=self.parent.translation[0],
            y=self.parent.translation[1],
            z=self.parent.translation[2],
            roll=r[0],
            pitch=r[1],
            yaw=r[2],
            x_marker=self.translation[0],
            y_marker=self.translation[1],
            z_marker=self.translation[2],
            roll_marker=r_marker[0],
            pitch_marker=r_marker[1],
            yaw_marker=r_marker[2],
        )


class Plane(SE3):
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
        parent=None,
        rotation=Rotation.from_euler("xyz", [0, 0, 0], degrees=True),
        translation=np.array([0, 0, 0]),
    ):
        super().__init__(name, parent, rotation, translation)

    def get_normal(self):
        return self.rotation.apply(np.array([0, 0, 1]))

    def as_xacro(self):
        r = self.rotation.as_euler("xyz")
        return self.__template.format(
            name=self.name,
            x=self.translation[0],
            y=self.translation[1],
            z=self.translation[2],
            roll=r[0],
            pitch=r[1],
            yaw=r[2],
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
            if type(child) is Plane:
                output += child.as_xacro()
                for marker in child.children:
                    output += marker.as_xacro()
        # output += self.__suffix
        autogen = self.empty_file.replace("<!-- OBJECTS -->", output)
        with open(os.path.join(os.getcwd(), "ros2_ws", "src", "simulation", "worlds", "autogen.urdf.xacro"), "w") as f:
            f.write(autogen)
        return autogen


def to_obc(root: SE3):
    content = []
    content.append("\t".join(["x", "y", "z"]) + "\n")
    rows = []
    for depth1 in root.children:
        if type(depth1) is Plane:
            for depth2 in depth1.children:
                if type(depth2) is Marker:
                    world_coordinate = depth1 @ depth2
                    r = world_coordinate.rotation.as_euler("xyz", degrees=True)
                    content.append(
                        "\t".join(
                            [
                                str(world_coordinate.translation[0]),
                                str(world_coordinate.translation[1]),
                                str(world_coordinate.translation[1]),
                            ]
                        )
                        + "\n"
                    )
                    rows.extend(
                        [
                            {
                                #                "name": node.name,
                                "path": "/".join([depth1.name, depth2.name]),
                                "id": depth2.id,
                                "x": world_coordinate.translation[0],
                                "y": world_coordinate.translation[1],
                                "z": world_coordinate.translation[2],
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
        initial_guess = SE3.from_dict("InitialGuess" + name, data.extrinsics)
        if "lidar" in name.lower():
            initial_guess.add_child(Lidar(name=name))
            vehicle.add_child(initial_guess)
        if "camera" in name.lower():
            initial_guess.add_child(Camera(name=name))
            vehicle.add_child(initial_guess)

    vehicle.show()

    world = SE3(name="origin")
    plane = Plane(name="p1", translation=np.array([2.0, 0.0, 1.0]), rotation=Rotation.from_euler("xyz", [0.0, -90.0, 0.0], degrees=True))
    plane.add_child(
        Marker(
            name="0",
            id=0,
            translation=np.array([-0.25, 0.0, 0.02]),
            rotation=Rotation.from_euler("xyz", [0.0, 0.0, 0.0], degrees=True),
        )
    )
    plane.add_child(
        Marker(
            name="1",
            id=1,
            translation=np.array([0.25, 0.0, 0.02]),
            rotation=Rotation.from_euler("xyz", [0.0, 0.0, 0.0], degrees=True),
        )
    )
    plane.add_child(
        Marker(
            name="2",
            id=2,
            translation=np.array([0.0, 0.0, 0.02]),
            rotation=Rotation.from_euler("xyz", [0.0, 0.0, 0.0], degrees=True),
        )
    )
    plane.add_child(
        Marker(
            name="3",
            id=3,
            translation=np.array([0.0, 0.25, 0.02]),
            rotation=Rotation.from_euler("xyz", [0.0, 0.0, 0.0], degrees=True),
        )
    )
    plane.add_child(
        Marker(
            name="4",
            id=4,
            translation=np.array([0.0, -0.25, 0.02]),
            rotation=Rotation.from_euler("xyz", [0.0, 0.0, 0.0], degrees=True),
        )
    )
    plane.add_child(
        Marker(
            name="5",
            id=5,
            translation=np.array([0.25, 0.25, 0.02]),
            rotation=Rotation.from_euler("xyz", [0.0, 0.0, 0.0], degrees=True),
        )
    )
    world.add_child(plane)
    xacro = XACRO(world)
    print(xacro.export())
    to_obc(world)
