import os
import numpy as np
from scipy.spatial.transform import Rotation
from calibration.datatypes import Transform
from calibration.generation.xacro_core import XACRO, create_nxm_marker_descriptions, create_plane

xacro_single_plane_bottom = XACRO("single_plane_bottom")
n = 10
m = 10
plane_id = 0
xacro_single_plane_bottom.world.add_child(
    create_plane(
        name="p" + str(plane_id),
        size=np.array([3.0, 3.0, 0.001]),
        transform=Transform(translation=np.array([6.0, 0.0, 0.1]), rotation=Rotation.from_euler("xyz", [0.0, 0.0, 0.0], degrees=True)),
        marker_descriptions=create_nxm_marker_descriptions(n, m),
    )
)
xacro_single_plane_bottom.export(output_directory=os.path.join(os.path.dirname(__file__), "single_plane_bottom"))

xacro_single_plane_front = XACRO("single_plane_front")
n = 10
m = 10
plane_id = 0
xacro_single_plane_front.world.add_child(
    create_plane(
        name="p" + str(plane_id),
        size=np.array([3.0, 3.0, 0.001]),
        transform=Transform(translation=np.array([5.0, 0.0, 1.5]), rotation=Rotation.from_euler("xyz", [0.0, -90.0, 0.0], degrees=True)),
        marker_descriptions=create_nxm_marker_descriptions(n, m),
    )
)
xacro_single_plane_front.export(output_directory=os.path.join(os.path.dirname(__file__), "single_plane_front"))

xacro_single_plane_left = XACRO("single_plane_left")
n = 10
m = 10
plane_id = 0
xacro_single_plane_left.world.add_child(
    create_plane(
        name="p" + str(plane_id),
        size=np.array([3.0, 3.0, 0.001]),
        transform=Transform(translation=np.array([8.0, 3.0, 1.5]), rotation=Rotation.from_euler("XYZ", [90.0, 0.0, 0.0], degrees=True)),
        marker_descriptions=create_nxm_marker_descriptions(n, m),
    )
)
xacro_single_plane_left.export(output_directory=os.path.join(os.path.dirname(__file__), "single_plane_left"))
