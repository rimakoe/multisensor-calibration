import os
import numpy as np
from scipy.spatial.transform import Rotation
from calibration.generation.xacro_core import XACRO, create_module, create_plane, create_nxm_marker_descriptions
from calibration.datatypes import Transform


def fill_parallax(xacro: XACRO, distance: float):
    xacro.world.add_child(
        create_module(
            0,
            transform=Transform(
                translation=np.array([distance, 0.0, 0.0]),
                rotation=Rotation.from_euler("xyz", [0, 0, 0], degrees=True),
            ),
        )
    )
    xacro.world.add_child(
        create_module(
            1,
            transform=Transform(
                translation=np.array([7, 2.5, 0.0]),
                rotation=Rotation.from_euler("xyz", [0, 0, 30], degrees=True),
            ),
        )
    )
    xacro.world.add_child(
        create_module(
            2,
            transform=Transform(
                translation=np.array([7.0, -2.5, 0.0]),
                rotation=Rotation.from_euler("xyz", [0, 0, -30], degrees=True),
            ),
        )
    )


for distance in [10, 12, 14, 16, 18, 20]:
    xacro = XACRO(f"parallax_{distance}")
    fill_parallax(xacro, distance)
    xacro.export(output_directory=os.path.join(os.path.dirname(__file__), f"parallax_{distance}"))


def fill_parallax_front_plane(xacro: XACRO, distance: float):
    xacro.world.add_child(
        create_plane(
            name="p1",
            size=np.array([2.0, 2.0, 0.001]),
            transform=Transform(
                translation=np.array([distance, 0.0, 1.0]),
                rotation=Rotation.from_euler("XYZ", [0, -90, 0], degrees=True),
            ),
            marker_descriptions=create_nxm_marker_descriptions(5, 5, 0),
        )
    )
    xacro.world.add_child(
        create_plane(
            name="p2",
            size=np.array([2.0, 2.0, 0.001]),
            transform=Transform(
                translation=np.array([10.0, 4.0, 1.0]),
                rotation=Rotation.from_euler("XYZ", [0, -90, 0], degrees=True),
            ),
            marker_descriptions=create_nxm_marker_descriptions(5, 5, 25),
        )
    )
    xacro.world.add_child(
        create_plane(
            name="p3",
            size=np.array([2.0, 2.0, 0.001]),
            transform=Transform(
                translation=np.array([10.0, -4.0, 1.0]),
                rotation=Rotation.from_euler("XYZ", [0, -90, 0], degrees=True),
            ),
            marker_descriptions=create_nxm_marker_descriptions(5, 5, 50),
        )
    )


for distance in [10, 12, 14, 16, 18, 20]:
    xacro = XACRO(f"parallax_plane_front_{distance}")
    fill_parallax_front_plane(xacro, distance)
    xacro.export(output_directory=os.path.join(os.path.dirname(__file__), f"parallax_plane_front_{distance}"))
