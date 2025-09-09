import os
import numpy as np
from scipy.spatial.transform import Rotation
from calibration.generation.xacro_core import XACRO, create_module
from calibration.datatypes import SE3

xacro = XACRO("calib_room")
xacro.world.add_child(
    create_module(
        0,
        transform=SE3(
            translation=np.array([3.5, 0.0, 0.0]),
            rotation=Rotation.from_euler("xyz", [0, 0, 0], degrees=True),
        ),
    )
)
xacro.world.add_child(
    create_module(
        1,
        transform=SE3(
            translation=np.array([2.5, 2.0, 0.0]),
            rotation=Rotation.from_euler("xyz", [0, 0, 45], degrees=True),
        ),
    )
)
xacro.world.add_child(
    create_module(
        2,
        transform=SE3(
            translation=np.array([0.0, 2.5, 0.0]),
            rotation=Rotation.from_euler("xyz", [0, 0, 90], degrees=True),
        ),
    )
)
xacro.world.add_child(
    create_module(
        3,
        transform=SE3(
            translation=np.array([-2.5, 2.0, 0.0]),
            rotation=Rotation.from_euler("xyz", [0, 0, 135], degrees=True),
        ),
    )
)
xacro.world.add_child(
    create_module(
        4,
        transform=SE3(
            translation=np.array([-3.5, 0.0, 0.0]),
            rotation=Rotation.from_euler("xyz", [0, 0, 180], degrees=True),
        ),
    )
)
xacro.world.add_child(
    create_module(
        5,
        transform=SE3(
            translation=np.array([-2.5, -2.0, 0.0]),
            rotation=Rotation.from_euler("xyz", [0, 0, -135], degrees=True),
        ),
    )
)
xacro.world.add_child(
    create_module(
        6,
        transform=SE3(
            translation=np.array([0.0, -2.5, 0.0]),
            rotation=Rotation.from_euler("xyz", [0, 0, -90], degrees=True),
        ),
    )
)
xacro.world.add_child(
    create_module(
        7,
        transform=SE3(
            translation=np.array([2.5, -2.0, 0.0]),
            rotation=Rotation.from_euler("xyz", [0, 0, -45], degrees=True),
        ),
    )
)
xacro.export(output_directory=os.path.join(os.path.dirname(__file__), "calib_room"))
print(xacro.world.as_dataframe(relative_coordinates=False))
