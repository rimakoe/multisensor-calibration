import os
import numpy as np
from scipy.spatial.transform import Rotation
from calibration.generation.xacro_core import XACRO, create_plane, create_nxm_marker_descriptions, create_box
from calibration.datatypes import Transform


xacro = XACRO("camera_special")
plane_rotation = Rotation.from_euler("XYZ", [0, -90.0, 0.0], degrees=True)
face_rotation = Rotation.from_euler("XYZ", [0, -45.0, -45.0], degrees=True)

xacro.world.add_child(
    create_box(
        0,
        transform=Transform(
            translation=np.array([5, -1.0, 1.0]),
            rotation=Rotation.from_matrix(Rotation.from_euler("XYZ", [0.0, 0.0, -5.0], degrees=True).as_matrix() @ face_rotation.as_matrix()),
        ),
    )
)
xacro.world.add_child(
    create_box(
        1,
        transform=Transform(
            translation=np.array([4, 1.0, 1.5]),
            rotation=Rotation.from_matrix(Rotation.from_euler("XYZ", [0.0, 0.0, 20.0], degrees=True).as_matrix() @ face_rotation.as_matrix()),
        ),
    )
)
xacro.world.add_child(
    create_plane(
        "p7",
        size=np.array([0.1, 1.0, 0.001]),
        transform=Transform(
            translation=np.array([2.5, 0.0, 1.1]),
            rotation=Rotation.from_matrix(Rotation.from_euler("xyz", [0.0, 0.0, 0.0], degrees=True).as_matrix() @ plane_rotation.as_matrix()),
        ),
        marker_descriptions=create_nxm_marker_descriptions(2, 5, 2 * 27),
    )
)
xacro.world.add_child(
    create_plane(
        "p8",
        size=np.array([10.0, 10.0, 0.001]),
        transform=Transform(
            translation=np.array([100, 0.0, 5.0]),
            rotation=Rotation.from_matrix(Rotation.from_euler("xyz", [0.0, 0.0, 0.0], degrees=True).as_matrix() @ plane_rotation.as_matrix()),
        ),
        marker_descriptions=create_nxm_marker_descriptions(5, 5, 2 * 27 + 2 * 5),
    )
)
xacro.export(output_directory=os.path.join(os.path.dirname(__file__), "camera_special"))
print(xacro.world.as_dataframe(relative_coordinates=False))
