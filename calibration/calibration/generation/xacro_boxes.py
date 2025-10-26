import os
import numpy as np
from scipy.spatial.transform import Rotation
from calibration.generation.xacro_core import XACRO, create_box
from calibration.datatypes import Transform

rng = np.random.default_rng(0)  # one reproducible stream

xacro = XACRO("boxes")
face_rotation = Rotation.from_euler("XYZ", [0, -45.0, -45.0], degrees=True)
xacro.world.add_child(
    create_box(
        0,
        transform=Transform(
            translation=np.array([6, 0.0, 1.0]),
            rotation=Rotation.from_matrix(Rotation.from_euler("xyz", [0.0, 0.0, 0.0], degrees=True).as_matrix() @ face_rotation.as_matrix()),
        ),
    )
)
xacro.world.add_child(
    create_box(
        1,
        transform=Transform(
            translation=np.array([5, 3.0, 1.0]),
            rotation=Rotation.from_matrix(Rotation.from_euler("XYZ", [0.0, 0.0, 30.0], degrees=True).as_matrix() @ face_rotation.as_matrix()),
        ),
    )
)
xacro.world.add_child(
    create_box(
        2,
        transform=Transform(
            translation=np.array([5, -3.0, 1.0]),
            rotation=Rotation.from_matrix(Rotation.from_euler("XYZ", [0.0, 0.0, -30.0], degrees=True).as_matrix() @ face_rotation.as_matrix()),
        ),
    )
)
xacro.export(output_directory=os.path.join(os.path.dirname(__file__), "boxes"))
print(xacro.world.as_dataframe(relative_coordinates=False))

xacro = XACRO("boxes_circle")
face_rotation = Rotation.from_euler("XYZ", [0, -45.0, -45.0], degrees=True)
number_of_boxes = 8
radius = 6  # meter
for i in range(number_of_boxes):
    xacro.world.add_child(
        create_box(
            i,
            transform=Transform(
                translation=np.array([radius * np.cos(i / number_of_boxes * 2 * np.pi), radius * np.sin(i / number_of_boxes * 2 * np.pi), 1.0]),
                rotation=Rotation.from_matrix(
                    Rotation.from_euler("xyz", [0.0, 0.0, i / number_of_boxes * 360.0], degrees=True).as_matrix() @ face_rotation.as_matrix()
                ),
            ),
        )
    )
xacro.export(output_directory=os.path.join(os.path.dirname(__file__), "boxes_circle"))
print(xacro.world.as_dataframe(relative_coordinates=False))


xacro = XACRO("boxes_circle_noise")
face_rotation = Rotation.from_euler("XYZ", [0, -45.0, -45.0], degrees=True)
number_of_boxes = 8
radius = 6  # meter
for i in range(number_of_boxes):
    radius_noise = radius + rng.normal(0, 1.0)
    xacro.world.add_child(
        create_box(
            i,
            transform=Transform(
                translation=np.array(
                    [
                        radius_noise * np.cos(i / number_of_boxes * 2 * np.pi),
                        radius_noise * np.sin(i / number_of_boxes * 2 * np.pi),
                        rng.normal(0, 0.1) + 1.0,
                    ]
                ),
                rotation=Rotation.from_matrix(
                    Rotation.from_euler("xyz", rng.normal(0, 1.0, size=3) + [0.0, 0.0, i / number_of_boxes * 360.0], degrees=True).as_matrix()
                    @ face_rotation.as_matrix()
                ),
            ),
        )
    )
xacro.export(output_directory=os.path.join(os.path.dirname(__file__), "boxes_circle_noise"))
print(xacro.world.as_dataframe(relative_coordinates=False))

xacro = XACRO("boxes_everywhere")
face_rotation = Rotation.from_euler("XYZ", [0, -45.0, -45.0], degrees=True)
number_of_boxes = 8
radius = 8  # meter
for i in range(number_of_boxes):
    radius_noise = radius + rng.normal(0, 1.0)
    xacro.world.add_child(
        create_box(
            i,
            transform=Transform(
                translation=np.array(
                    [
                        radius_noise * np.cos(i / number_of_boxes * 2 * np.pi),
                        radius_noise * np.sin(i / number_of_boxes * 2 * np.pi),
                        rng.normal(0, 0.1) + 1.0,
                    ]
                ),
                rotation=Rotation.from_matrix(
                    Rotation.from_euler("xyz", rng.normal(0, 1.0, size=3) + [0.0, 0.0, i / number_of_boxes * 360.0], degrees=True).as_matrix()
                    @ face_rotation.as_matrix()
                ),
            ),
        )
    )
radius = 12  # meter
for i in range(number_of_boxes, 2 * number_of_boxes):
    radius_noise = radius + rng.normal(0, 1.0)
    xacro.world.add_child(
        create_box(
            i,
            transform=Transform(
                translation=np.array(
                    [
                        radius_noise * np.cos(i / number_of_boxes * 2 * np.pi + 2 * np.pi / (2 * number_of_boxes)),
                        radius_noise * np.sin(i / number_of_boxes * 2 * np.pi + 2 * np.pi / (2 * number_of_boxes)),
                        rng.normal(0, 0.1) + 1.0,
                    ]
                ),
                rotation=Rotation.from_matrix(
                    Rotation.from_euler("xyz", rng.normal(0, 1.0, size=3) + [0.0, 0.0, i / number_of_boxes * 360.0], degrees=True).as_matrix()
                    @ face_rotation.as_matrix()
                ),
            ),
        )
    )
xacro.export(output_directory=os.path.join(os.path.dirname(__file__), "boxes_everywhere"))
print(xacro.world.as_dataframe(relative_coordinates=False))
