import os
import numpy as np
from scipy.spatial.transform import Rotation
from calibration.datatypes import Transform
from calibration.generation.xacro_core import XACRO, create_nxm_marker_descriptions, create_plane

xacro_dual_plane_bottom_left = XACRO("dual_plane_bottom_left")
n = 10
m = 10
plane_id = 0
xacro_dual_plane_bottom_left.world.add_child(
    create_plane(
        name="p" + str(plane_id),
        size=np.array([3.0, 3.0, 0.001]),
        transform=Transform(translation=np.array([6.0, 0.0, 0.1]), rotation=Rotation.from_euler("xyz", [0.0, 0.0, 0.0], degrees=True)),
        marker_descriptions=create_nxm_marker_descriptions(n, m, 0),
    )
)
plane_id += 1
xacro_dual_plane_bottom_left.world.add_child(
    create_plane(
        name="p" + str(plane_id),
        size=np.array([3.0, 3.0, 0.001]),
        transform=Transform(translation=np.array([6.0, 1.5, 1.5]), rotation=Rotation.from_euler("XYZ", [90.0, 0.0, 0.0], degrees=True)),
        marker_descriptions=create_nxm_marker_descriptions(n, m, n * m),
    )
)
xacro_dual_plane_bottom_left.export(output_directory=os.path.join(os.path.dirname(__file__), "dual_plane_bottom_left"))

xacro_dual_plane_bottom_front = XACRO("dual_plane_bottom_front")
n = 10
m = 10
plane_id = 0
xacro_dual_plane_bottom_front.world.add_child(
    create_plane(
        name="p" + str(plane_id),
        size=np.array([3.0, 3.0, 0.001]),
        transform=Transform(translation=np.array([7.5, 0.0, 1.5]), rotation=Rotation.from_euler("xyz", [0.0, -90.0, 0.0], degrees=True)),
        marker_descriptions=create_nxm_marker_descriptions(n, m, 0),
    )
)
plane_id += 1
xacro_dual_plane_bottom_front.world.add_child(
    create_plane(
        name="p" + str(plane_id),
        size=np.array([3.0, 3.0, 0.001]),
        transform=Transform(translation=np.array([6.0, 0.0, 0.1]), rotation=Rotation.from_euler("xyz", [0.0, 0.0, 0.0], degrees=True)),
        marker_descriptions=create_nxm_marker_descriptions(n, m, n * m),
    )
)
xacro_dual_plane_bottom_front.export(output_directory=os.path.join(os.path.dirname(__file__), "dual_plane_bottom_front"))

xacro_dual_plane_left_front = XACRO("dual_plane_left_front")
n = 10
m = 10
plane_id = 0
xacro_dual_plane_left_front.world.add_child(
    create_plane(
        name="p" + str(plane_id),
        size=np.array([3.0, 3.0, 0.001]),
        transform=Transform(translation=np.array([8.0, 3.0, 1.5]), rotation=Rotation.from_euler("XYZ", [90.0, 0.0, 0.0], degrees=True)),
        marker_descriptions=create_nxm_marker_descriptions(n, m, 0),
    )
)
plane_id += 1
xacro_dual_plane_left_front.world.add_child(
    create_plane(
        name="p" + str(plane_id),
        size=np.array([3.0, 3.0, 0.001]),
        transform=Transform(translation=np.array([9.5, 0.0, 1.5]), rotation=Rotation.from_euler("xyz", [0.0, -90.0, 0.0], degrees=True)),
        marker_descriptions=create_nxm_marker_descriptions(n, m, n * m),
    )
)
xacro_dual_plane_left_front.export(output_directory=os.path.join(os.path.dirname(__file__), "dual_plane_left_front"))
