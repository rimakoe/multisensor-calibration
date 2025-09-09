import os, json
import pandas as pd
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt
from calibration.datatypes import SE3, Camera
from calibration.core import Solution
from calibration.utils import read_obc, get_dataset_directory


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


def generate_accurate_camera_solution(dataset_name: str, camera_name: str):
    _convention_transform = SE3(rotation=Rotation.from_euler("YXZ", [90, 0, -90], degrees=True))
    solution_filepath = os.path.join(get_dataset_directory(), dataset_name, "solution.json")
    solution = Solution(**json.load(open(solution_filepath)))
    camera = solution.devices[camera_name]
    camera = Camera(
        name=camera_name,
        id=1000,
        transform=SE3(translation=camera.extrinsics.translation.to_numpy(), rotation=camera.extrinsics.rotation.to_scipy()),
        intrinsics=Camera.Intrinsics.from_json(os.path.join(get_dataset_directory(), dataset_name, camera_name.lower(), "camera_info.json")),
    )
    photogrammetry = read_obc(os.path.join("/home", "workspace", "datasets", dataset_name, "photogrammetry.obc"))
    photogrammetry[["x", "y", "z"]] = (camera.transform @ _convention_transform).inverse().apply(photogrammetry)
    projected_data = projectXYZ2UV(data=photogrammetry, intrinsics=camera.intrinsics)
    projected_data.to_json(os.path.join("/home", "workspace", "datasets", dataset_name, camera.name.lower(), "detections.json"))
    plt.scatter(x=projected_data["u"], y=projected_data["v"])
    plt.xlim([0.0, camera.intrinsics.width])
    plt.ylim([camera.intrinsics.height, 0.0])
    plt.show(block=True)


if __name__ == "__main__":
    generate_accurate_camera_solution(dataset_name="C1_p1bottom", camera_name="TopViewCameraFront")
