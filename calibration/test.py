import os
from datatypes import *
import core
import tqdm


def projectXYZ2UV(data: pd.DataFrame, intrinsics: Camera.Intrinsics, filter_fov: bool = True):
    missing = [required_column for required_column in ["x", "y", "z"] if required_column not in data.columns]
    assert len(missing) == 0, f"missing column(s): {missing}"
    # data = data[["x", "y", "z"]]
    data = data[data["z"] > 0.0]
    # K = intrinsics.as_matrix()
    # P = np.hstack([K, np.zeros((3, 1))])
    # xyz_hom = np.vstack([data.T, np.ones((1, data.shape[0]))])
    # uv_hom = P @ xyz_hom
    # uv_hom[0, :] /= xyz_hom[2, :]
    # uv_hom[1, :] /= xyz_hom[2, :]
    # return uv_hom[0:2, :]
    u = intrinsics.focal_length[0] * data["x"] / data["z"] + intrinsics.principal_point[0]
    v = intrinsics.focal_length[1] * data["y"] / data["z"] + intrinsics.principal_point[1]
    df = pd.DataFrame({"id": data["id"], "u": u, "v": v})
    if filter_fov:
        df = df[df["u"] < intrinsics.width]
        df = df[df["u"] > 0]
        df = df[df["v"] < intrinsics.height]
        df = df[df["v"] > 0]
    return df


if __name__ == "__main__":
    dataset_name = "C1"
    _convention_transform = SE3(rotation=Rotation.from_euler("YXZ", [90, 0, -90], degrees=True))
    p_values = []
    for i in tqdm.tqdm(range(10)):
        vehicle, report = core.main(dataset_name, silent=True)
        p_values.append(report["p-value"])
    print(np.mean(p_values))
    # camera: Camera = vehicle.get_frame("TopViewCameraFront")
    # photogrammetry = read_obc(os.path.join(os.getcwd(), "datasets", dataset_name, "photogrammetry.obc"))
    # photogrammetry[["x", "y", "z"]] = (camera.parent.transform @ _convention_transform).inverse().apply(photogrammetry)
    # projected_data = projectXYZ2UV(data=photogrammetry, intrinsics=camera.intrinsics)
    # projected_data.to_json(os.path.join(os.getcwd(), "datasets", dataset_name, "detections.json"))
    # plt.scatter(x=projected_data["u"], y=projected_data["v"])
    # plt.xlim([0.0, camera.intrinsics.width])
    # plt.ylim([camera.intrinsics.height, 0.0])
    # plt.show(block=True)
