from datatypes import *
from typing import Tuple
import os
import tqdm
from core import VehicleFactory


def projectXYZ2UV(data: pd.DataFrame, intrinsics: Camera.Intrinsics, filter_fov: bool = True):
    # directory_to_datasets = "/" + os.path.join("home", "workspace", "datasets")
    missing = [required_column for required_column in ["x", "y", "z"] if required_column not in data.columns]
    assert len(missing) == 0, f"missing column(s): {missing}"
    data = data[data["z"] > 0.0]
    u = intrinsics.focal_length[0] * data["x"] / data["z"] + intrinsics.principal_point[0]
    v = intrinsics.focal_length[1] * data["y"] / data["z"] + intrinsics.principal_point[1]
    df = pd.DataFrame({"id": data["id"], "u": u, "v": v})
    if filter_fov:
        df = df[df["u"] < intrinsics.width]
        df = df[df["u"] > 0]
        df = df[df["v"] < intrinsics.height]
        df = df[df["v"] > 0]
    return df


def generate_accurate_camera_solution(dataset_name="C1"):
    _convention_transform = SE3(rotation=Rotation.from_euler("YXZ", [90, 0, -90], degrees=True))
    camera: Camera = vehicle.get_frame("TopViewCameraFront")
    photogrammetry = read_obc(os.path.join(os.getcwd(), "datasets", dataset_name, "photogrammetry.obc"))
    photogrammetry[["x", "y", "z"]] = (camera.parent.transform @ _convention_transform).inverse().apply(photogrammetry)
    projected_data = projectXYZ2UV(data=photogrammetry, intrinsics=camera.intrinsics)
    projected_data.to_json(os.path.join(os.getcwd(), "datasets", dataset_name, "detections.json"))
    plt.scatter(x=projected_data["u"], y=projected_data["v"])
    plt.xlim([0.0, camera.intrinsics.width])
    plt.ylim([camera.intrinsics.height, 0.0])
    plt.show(block=True)


class XACRO:
    def __init__(self):
        self.world = Frame("autogen")
        with open(os.path.join(os.getcwd(), "ros2_ws", "src", "simulation", "worlds", "empty.urdf.xacro")) as f:
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

    def export(self, output_filepath: str):
        # CREATE THE autogen.urdf.xacro
        autogen = self.empty_file.replace("<!-- OBJECTS -->", self.flatten())
        with open(output_filepath, "w") as f:
            f.write(autogen)

        # CREATE THE OBC
        write_obc(
            dataframe=self.world.as_dataframe(only_leafs=True, relative_coordinates=False),
            output_filepath=os.path.join(os.getcwd(), "ros2_ws", "src", "simulation", "worlds", f"{self.world.name}.obc"),
        )
        df = read_obc(os.path.join(os.getcwd(), "ros2_ws", "src", "simulation", "worlds", "autogen.obc"))
        print(df)

    def __create_9x9_marker_descriptions(self, start_id: int = 0):
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

    def __create_321_marker_descriptions(self, start_id: int = 0):
        return [
            (start_id + 0, SE3(translation=np.array([0.25, -0.25, 0.001]))),
            (start_id + 1, SE3(translation=np.array([0.25, 0.0, 0.001]))),
            (start_id + 2, SE3(translation=np.array([0.25, 0.25, 0.001]))),
            (start_id + 3, SE3(translation=np.array([0.0, -0.125, 0.001]))),
            (start_id + 4, SE3(translation=np.array([0.0, 0.125, 0.001]))),
            (start_id + 5, SE3(translation=np.array([-0.25, 0.0, 0.001]))),
        ]

    def create_plane(
        self,
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
            marker_descriptions = self.__create_9x9_marker_descriptions()
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

    def create_module(self, id: int, transform: SE3 = SE3()) -> Frame:
        module = Frame(name=f"mod{id}", transform=transform)
        module.add_child(
            self.create_plane(
                name="p" + str(id * 3 + 0),
                size=np.array([1.0, 1.7, 0.001]),
                transform=SE3(translation=np.array([-0.1, 0.0, 1.75]), rotation=Rotation.from_euler("xyz", [0.0, -120.0, 0.0], degrees=True)),
                marker_descriptions=self.__create_321_marker_descriptions(id * 18 + 0),
            )
        )
        module.add_child(
            self.create_plane(
                name="p" + str(id * 3 + 1),
                size=np.array([1.0, 2.0, 0.001]),
                transform=SE3(translation=np.array([0.0, 0.4, 1.0]), rotation=Rotation.from_euler("YXZ", [-90.0, 25.0, 90.0], degrees=True)),
                marker_descriptions=self.__create_321_marker_descriptions(id * 18 + 6),
            )
        )
        module.add_child(
            self.create_plane(
                name="p" + str(id * 3 + 2),
                size=np.array([1.0, 2.0, 0.001]),
                transform=SE3(translation=np.array([0.0, -0.4, 1.0]), rotation=Rotation.from_euler("YXZ", [-90.0, -25.0, -90.0], degrees=True)),
                marker_descriptions=self.__create_321_marker_descriptions(id * 18 + 12),
            )
        )
        return module

    def create_world(self, name: str):
        self.world.name = name
        self.world.add_child(
            self.create_module(
                0,
                transform=SE3(
                    translation=np.array([3.5, 0.0, 0.0]),
                    rotation=Rotation.from_euler("xyz", [0, 0, 0], degrees=True),
                ),
            )
        )
        self.world.add_child(
            self.create_module(
                1,
                transform=SE3(
                    translation=np.array([2.5, 2.0, 0.0]),
                    rotation=Rotation.from_euler("xyz", [0, 0, 45], degrees=True),
                ),
            )
        )
        self.world.add_child(
            self.create_module(
                2,
                transform=SE3(
                    translation=np.array([0.0, 2.5, 0.0]),
                    rotation=Rotation.from_euler("xyz", [0, 0, 90], degrees=True),
                ),
            )
        )
        self.world.add_child(
            self.create_module(
                3,
                transform=SE3(
                    translation=np.array([-2.5, 2.0, 0.0]),
                    rotation=Rotation.from_euler("xyz", [0, 0, 135], degrees=True),
                ),
            )
        )
        self.world.add_child(
            self.create_module(
                4,
                transform=SE3(
                    translation=np.array([-3.5, 0.0, 0.0]),
                    rotation=Rotation.from_euler("xyz", [0, 0, 180], degrees=True),
                ),
            )
        )
        self.world.add_child(
            self.create_module(
                5,
                transform=SE3(
                    translation=np.array([-2.5, -2.0, 0.0]),
                    rotation=Rotation.from_euler("xyz", [0, 0, -135], degrees=True),
                ),
            )
        )
        self.world.add_child(
            self.create_module(
                6,
                transform=SE3(
                    translation=np.array([0.0, -2.5, 0.0]),
                    rotation=Rotation.from_euler("xyz", [0, 0, -90], degrees=True),
                ),
            )
        )
        self.world.add_child(
            self.create_module(
                7,
                transform=SE3(
                    translation=np.array([2.5, -2.0, 0.0]),
                    rotation=Rotation.from_euler("xyz", [0, 0, -45], degrees=True),
                ),
            )
        )

        print(self.world.as_dataframe(relative_coordinates=False))


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


if __name__ == "__main__":
    xacro = XACRO()
    xacro.create_world("autogen")
    xacro.export(output_filepath=os.path.join(os.getcwd(), "ros2_ws", "src", "simulation", "worlds", f"{xacro.world.name}.urdf.xacro"))
