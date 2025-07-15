from datatypes import *
from typing import Tuple


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

    def __init__(self):
        self.world = Frame("autogen")
        with open(os.path.join(os.getcwd(), "ros2_ws", "src", "simulation", "worlds", "empty.urdf.xacro")) as f:
            self.empty_file = f.read()

    def export(self):
        # CREATE THE autogen.urdf.xacro
        output = ""
        # output += self.__prefix
        for child in self.world.children:
            if type(child) is not XACROPlane:
                continue
            output += child.as_xacro()
            for marker in child.children:
                if type(marker) is not XACROMarker:
                    continue
                output += marker.as_xacro()
        # output += self.__suffix
        autogen = self.empty_file.replace("<!-- OBJECTS -->", output)
        with open(os.path.join(os.getcwd(), "ros2_ws", "src", "simulation", "worlds", f"{self.world.name}.urdf.xacro"), "w") as f:
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
            (start_id + 0, SE3(translation=np.array([-0.25, 0.0, 0.02]), rotation=Rotation.from_euler("xyz", [0.0, 0.0, 0.0], degrees=True))),
            (start_id + 1, SE3(translation=np.array([0.0, 0.0, 0.02]), rotation=Rotation.from_euler("xyz", [0.0, 0.0, 0.0], degrees=True))),
            (start_id + 2, SE3(translation=np.array([0.25, 0.0, 0.02]), rotation=Rotation.from_euler("xyz", [0.0, 0.0, 0.0], degrees=True))),
            (start_id + 3, SE3(translation=np.array([-0.25, 0.25, 0.02]), rotation=Rotation.from_euler("xyz", [0.0, 0.0, 0.0], degrees=True))),
            (start_id + 4, SE3(translation=np.array([0.0, 0.25, 0.02]), rotation=Rotation.from_euler("xyz", [0.0, 0.0, 0.0], degrees=True))),
            (start_id + 5, SE3(translation=np.array([0.25, 0.25, 0.02]), rotation=Rotation.from_euler("xyz", [0.0, 0.0, 0.0], degrees=True))),
            (start_id + 6, SE3(translation=np.array([-0.25, -0.25, 0.02]), rotation=Rotation.from_euler("xyz", [0.0, 0.0, 0.0], degrees=True))),
            (start_id + 7, SE3(translation=np.array([0.0, -0.25, 0.02]), rotation=Rotation.from_euler("xyz", [0.0, 0.0, 0.0], degrees=True))),
            (start_id + 8, SE3(translation=np.array([0.25, -0.25, 0.02]), rotation=Rotation.from_euler("xyz", [0.0, 0.0, 0.0], degrees=True))),
        ]

    def create_plane(
        self,
        name: str,
        size: np.ndarray = np.array([1, 1, 0.02]),
        transform: SE3 = SE3(),
        marker_descriptions: Tuple[int, SE3] = None,
        start_id: int = 0,
    ):
        plane = XACROPlane(
            name=name,
            transform=transform,
            size=size,
        )
        if not marker_descriptions:
            marker_descriptions = self.__create_9x9_marker_descriptions(start_id)
        for id, marker_transform in marker_descriptions:
            plane.add_child(
                XACROMarker(
                    name=f"m{id}",
                    id=id,
                    transform=marker_transform,
                )
            )
        return plane

    def create_world(self):
        self.world.add_child(
            self.create_plane(
                name="p1",
                size=np.array([1.0, 1.7, 0.02]),
                transform=SE3(translation=np.array([1.9, 0.0, 1.25]), rotation=Rotation.from_euler("xyz", [0.0, -120.0, 0.0], degrees=True)),
                start_id=0,
            )
        )
        self.world.add_child(
            self.create_plane(
                name="p2",
                transform=SE3(translation=np.array([2.0, 0.4, 1.0]), rotation=Rotation.from_euler("xyz", [0.0, -90.0, 25.0], degrees=True)),
                start_id=9,
            )
        )
        self.world.add_child(
            self.create_plane(
                name="p3",
                transform=SE3(translation=np.array([2.0, -0.4, 1.0]), rotation=Rotation.from_euler("xyz", [0.0, -90.0, -25.0], degrees=True)),
                start_id=18,
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

    def as_xacro(self) -> str:
        r = self.parent.transform.rotation.as_euler("xyz")
        r_marker = self.transform.rotation.as_euler("xyz")
        return self.__template.format(
            marker_name=self.name,
            marker_id=self.id,
            parent_name=self.parent.name,
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


if __name__ == "__main__":
    xacro = XACRO()
    xacro.create_world()
    xacro.export()
