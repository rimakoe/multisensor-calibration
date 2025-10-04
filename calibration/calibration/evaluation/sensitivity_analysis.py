import os
from calibration.datatypes import *
from calibration.core import VehicleFactory, Vehicle, ExtendedSparseOptimizer, evaluate
from calibration.core import main as aio_calibration
import tqdm
import seaborn as sns
import matplotlib.pyplot as plt


def feature_sensitivity():
    # Path configuration stuff
    directory_to_datasets = "/" + os.path.join("home", "workspace", "datasets")
    dataset_name = "C1_p1front"
    sensor_name = "FrontCameraWide"
    sensor_folder = os.path.join(directory_to_datasets, dataset_name, sensor_name.lower())
    solution_filepath = os.path.join(directory_to_datasets, dataset_name, "solution.json")
    solution = Solution(**json.load(open(solution_filepath)))

    # actual content of this investigation
    camera_feature_noise = 1.0  # px - normal distributed feature noise
    deviation = Transform(  # initial guess deviation from solution
        rotation=Rotation.from_euler("xyz", [3, 3, 3], degrees=True),
        translation=np.array([0.1, 0.1, 0.1]),
    )

    df = pd.DataFrame()
    for i in tqdm.tqdm(range(1000)):
        np.random.seed(i)
        features = pd.DataFrame.from_dict(
            json.load(open(os.path.join(sensor_folder, "detections.json"))),
        )
        features[["su", "sv"]] = np.ones(features[["u", "v"]].shape) * camera_feature_noise  # give really sigma here
        camera = Camera(
            name=sensor_name,
            id=1000,
            data=Image.open(os.path.join(sensor_folder, sensor_name.lower() + ".bmp")),
            intrinsics=Camera.Intrinsics.from_json(
                os.path.join(sensor_folder, "camera_info.json"),
            ),
            features=features,  # give the perfect features for now jsut to avoid None, this is modified during the loop in a normal distributed fashion
        )
        camera.features[["u", "v"]] = features[["u", "v"]].copy() + np.random.normal(0.0, camera_feature_noise, size=features[["u", "v"]].shape)
        initial_guess = Frame(name="InitialGuess", transform=Transform.from_dict(solution.devices[sensor_name].extrinsics) @ deviation)
        initial_guess.add_child(camera)
        vehicle = Vehicle(name="sensitivity analysis vehicle container", system_configuration=dataset_name)
        vehicle.add_child(initial_guess)
        optimizer = ExtendedSparseOptimizer(vehicle=vehicle)
        optimizer.add_photogrammetry(photogrammetry=read_obc(os.path.join(get_dataset_directory(), dataset_name, "photogrammetry.obc")))
        optimizer.add_reprojection_error_minimization(camera)
        optimizer.initialize_optimization()
        optimizer.set_verbose(False)
        optimizer.optimize(10000)
        report = evaluate(optimizer)
        delta = Transform.from_dict(solution.devices[sensor_name].extrinsics).inverse() @ initial_guess.transform @ camera.transform
        precision_rotation = Rotation.from_rotvec(np.sqrt(camera.covariance[:3, :3].diagonal().copy()))
        precision_translation = np.sqrt(camera.covariance[3:, 3:].diagonal().copy())
        precision_rotation_vector = precision_rotation.as_euler("xyz", degrees=True)  # in deg for slighlty better intuition
        delta_rotation_vector = delta.rotation.as_euler("xyz", degrees=True)
        report |= {
            "prec_tx": precision_translation[0],
            "prec_ty": precision_translation[1],
            "prec_tz": precision_translation[2],
            "prec_rx": precision_rotation_vector[0],
            "prec_ry": precision_rotation_vector[1],
            "prec_rz": precision_rotation_vector[2],
            "delta_tx": delta.translation[0],
            "delta_ty": delta.translation[1],
            "delta_tz": delta.translation[2],
            "delta_rx": delta_rotation_vector[0],
            "delta_ry": delta_rotation_vector[1],
            "delta_rz": delta_rotation_vector[2],
            "delta_t": np.linalg.norm(delta.translation, 2),
            "delta_r": np.linalg.norm(delta_rotation_vector, 2),
        }
        optimizer.clear()
        delta = Transform.from_dict(solution.devices[sensor_name].extrinsics).inverse() @ initial_guess.transform @ camera.transform
        df = pd.concat([df, pd.DataFrame(report, index=[i])])
    output = pd.DataFrame(
        {
            "mean": np.mean(
                df[
                    [
                        "prec_tx",
                        "prec_ty",
                        "prec_tz",
                        "prec_rx",
                        "prec_ry",
                        "prec_rz",
                        "delta_tx",
                        "delta_ty",
                        "delta_tz",
                        "delta_rx",
                        "delta_ry",
                        "delta_rz",
                    ]
                ],
                axis=0,
            ),
            "std": np.std(
                df[
                    [
                        "prec_tx",
                        "prec_ty",
                        "prec_tz",
                        "prec_rx",
                        "prec_ry",
                        "prec_rz",
                        "delta_tx",
                        "delta_ty",
                        "delta_tz",
                        "delta_rx",
                        "delta_ry",
                        "delta_rz",
                    ]
                ],
                axis=0,
            ),
        }
    )
    # output["normed"] = np.abs(output["mean"]) / output["std"]
    output = output.T
    output["noise"] = camera_feature_noise
    sns.boxplot(data=pd.melt(df), y="delta_tx")
    plt.show(block=True)
    print(output)


if __name__ == "__main__":
    feature_sensitivity()
