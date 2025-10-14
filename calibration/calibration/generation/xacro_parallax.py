import os
import numpy as np
from scipy.spatial.transform import Rotation
from calibration.generation.xacro_core import XACRO, create_module
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
