import os
from typing import Tuple
from calibration.datatypes import *


class XACROMarker(Marker):
    __template = "\n".join(
        [
            """    <model name="{parent_name}_{marker_name}">""",
            """        <static>true</static>""",
            """        <pose>{x} {y} {z} {roll} {pitch} {yaw}</pose>""",
            """        <xacro:marker_macro name="{marker_name}" id="{marker_id}" parent_link="{parent_name}_link" x="{x_marker}" y="{y_marker}" z="{z_marker}" roll="{roll_marker}" pitch="{pitch_marker}" yaw="{yaw_marker}"/>""",
            """    </model>""",
            """""",
        ]
    )

    def __init__(self, id: int, name: str, parent: Frame = None, transform: SE3 = SE3()):
        super().__init__(id, name, parent, transform)

    def as_xacro(self, path: List[Frame]) -> str:
        path.pop(-1)
        absolute_parent_transform = SE3()
        for frame in path:
            absolute_parent_transform = absolute_parent_transform @ frame.transform
        r_parent = absolute_parent_transform.rotation.as_euler("xyz")
        r = self.transform.rotation.as_euler("xyz")
        return self.__template.format(
            marker_name=self.name,
            marker_id=self.id,
            parent_name=self.parent.name,
            x=absolute_parent_transform.translation[0],
            y=absolute_parent_transform.translation[1],
            z=absolute_parent_transform.translation[2],
            roll=r_parent[0],
            pitch=r_parent[1],
            yaw=r_parent[2],
            x_marker=self.transform.translation[0],
            y_marker=self.transform.translation[1],
            z_marker=self.transform.translation[2],
            roll_marker=r[0],
            pitch_marker=r[1],
            yaw_marker=r[2],
        )


class XACROPlane(Plane):
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
            """                         <size>{size_x} {size_y} {size_z}</size>""",
            """                     </box>""",
            """                 </geometry>""",
            """             <xacro:set_material color="0.0 0.0 1.0" />""",
            """             </visual>""",
            """         </link>""",
            """    </model>""",
            """""",
        ]
    )

    def __init__(self, name: str, parent: Frame = None, transform: SE3 = SE3(), size: np.ndarray = np.array([1.0, 1.0, 0.02])):
        super().__init__(name, parent, transform)
        assert size.shape == (3,)
        self.size = size

    def get_normal(self):
        return self.transform.rotation.apply(np.array([0, 0, 1]))

    def as_xacro(self, path: List[Frame]) -> str:
        absolute_transform = SE3()
        for frame in path:
            absolute_transform = absolute_transform @ frame.transform
        r = absolute_transform.rotation.as_euler("xyz")
        t = absolute_transform.translation
        return self.__template.format(
            name=self.name,
            x=t[0],
            y=t[1],
            z=t[2],
            roll=r[0],
            pitch=r[1],
            yaw=r[2],
            size_x=self.size[0],
            size_y=self.size[1],
            size_z=self.size[2],
        )


def create_3x3_marker_descriptions(start_id: int = 0):
    return [
        (start_id + 0, SE3(translation=np.array([-0.25, 0.0, 0.001]))),
        (start_id + 1, SE3(translation=np.array([0.0, 0.0, 0.001]))),
        (start_id + 2, SE3(translation=np.array([0.25, 0.0, 0.001]))),
        (start_id + 3, SE3(translation=np.array([-0.25, 0.25, 0.001]))),
        (start_id + 4, SE3(translation=np.array([0.0, 0.25, 0.001]))),
        (start_id + 5, SE3(translation=np.array([0.25, 0.25, 0.001]))),
        (start_id + 6, SE3(translation=np.array([-0.25, -0.25, 0.001]))),
        (start_id + 7, SE3(translation=np.array([0.0, -0.25, 0.001]))),
        (start_id + 8, SE3(translation=np.array([0.25, -0.25, 0.001]))),
    ]


def create_321_marker_descriptions(start_id: int = 0):
    return [
        (start_id + 0, SE3(translation=np.array([0.25, -0.25, 0.001]))),
        (start_id + 1, SE3(translation=np.array([0.25, 0.0, 0.001]))),
        (start_id + 2, SE3(translation=np.array([0.25, 0.25, 0.001]))),
        (start_id + 3, SE3(translation=np.array([0.0, -0.125, 0.001]))),
        (start_id + 4, SE3(translation=np.array([0.0, 0.125, 0.001]))),
        (start_id + 5, SE3(translation=np.array([-0.25, 0.0, 0.001]))),
    ]


def create_nxm_marker_descriptions(n: int, m: int, start_id: int = 0):
    markers = []
    for row in range(n):
        x = row / (n - 1) - 0.5
        x *= 0.8
        for col in range(m):
            y = col / (m - 1) - 0.5
            y *= 0.8
            markers.append((start_id + (row * m) + (col + 1), SE3(translation=np.array([x, y, 0.001]))))
    return markers


def create_plane(
    name: str,
    size: np.ndarray = np.array([1, 1, 0.001]),
    transform: SE3 = SE3(),
    marker_descriptions: Tuple[int, SE3] = None,
):
    plane = XACROPlane(
        name=name,
        transform=transform,
        size=size,
    )
    if not marker_descriptions:
        marker_descriptions = create_3x3_marker_descriptions()
    for id, marker_transform in marker_descriptions:
        marker_transform.translation[0] *= size[0]
        marker_transform.translation[1] *= size[1]
        plane.add_child(
            XACROMarker(
                name=f"m{id}",
                id=id,
                transform=marker_transform,
            )
        )
    return plane


def create_module(id: int, transform: SE3 = SE3()) -> Frame:
    module = Frame(name=f"mod{id}", transform=transform)
    module.add_child(
        create_plane(
            name="p" + str(id * 3 + 0),
            size=np.array([1.0, 1.7, 0.001]),
            transform=SE3(translation=np.array([-0.1, 0.0, 1.75]), rotation=Rotation.from_euler("xyz", [0.0, -120.0, 0.0], degrees=True)),
            marker_descriptions=create_321_marker_descriptions(id * 18 + 0),
        )
    )
    module.add_child(
        create_plane(
            name="p" + str(id * 3 + 1),
            size=np.array([1.0, 2.0, 0.001]),
            transform=SE3(translation=np.array([0.0, 0.4, 1.0]), rotation=Rotation.from_euler("YXZ", [-90.0, 25.0, 90.0], degrees=True)),
            marker_descriptions=create_321_marker_descriptions(id * 18 + 6),
        )
    )
    module.add_child(
        create_plane(
            name="p" + str(id * 3 + 2),
            size=np.array([1.0, 2.0, 0.001]),
            transform=SE3(translation=np.array([0.0, -0.4, 1.0]), rotation=Rotation.from_euler("YXZ", [-90.0, -25.0, -90.0], degrees=True)),
            marker_descriptions=create_321_marker_descriptions(id * 18 + 12),
        )
    )
    return module


class XACRO:
    def __init__(self, world_name: str = "autogen"):
        self.world = Frame(name=world_name)
        with open(os.path.join("/home", "workspace", "ros2_ws", "src", "simulation", "worlds", "empty.urdf.xacro")) as f:
            self.empty_file = f.read()

    def flatten(self, frame: Frame = None, path: List[Frame] = None) -> str:
        if not frame:
            frame = self.world
        if not path:
            path = []
        current_path: List[Frame] = path + [frame]
        output = ""
        if type(frame) is XACROMarker or type(frame) is XACROPlane:
            output += frame.as_xacro(current_path)
        for child in frame.children:
            output += self.flatten(child, current_path)
        return output

    def export(self, output_directory: str = os.getcwd()):
        # CREATE THE autogen.urdf.xacro
        os.makedirs(output_directory, exist_ok=True)
        autogen = self.empty_file.replace("<!-- OBJECTS -->", self.flatten())
        with open(os.path.join(output_directory, f"{self.world.name}.urdf.xacro"), "w") as f:
            f.write(autogen)

        # CREATE THE OBC
        write_obc(
            dataframe=self.world.as_dataframe(only_leafs=True, relative_coordinates=False),
            output_filepath=os.path.join(output_directory, f"{self.world.name}.obc"),
        )
        df = read_obc(os.path.join(output_directory, f"{self.world.name}.obc"))
        print(df)
