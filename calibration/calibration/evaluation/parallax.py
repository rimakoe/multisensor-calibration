from calibration.datatypes import *
from calibration.core import Vehicle, ExtendedSparseOptimizer, evaluate
import matplotlib.pyplot as plt

rng = np.random.default_rng(0)  # one reproducible stream


def run():
    # Path configuration stuff
    directory_to_datasets = "/" + os.path.join("home", "workspace", "datasets")
    dataset_name = "parallax"
    result = []
    for subset_name in sorted(os.listdir(os.path.join(directory_to_datasets, "parallax"))):
        sensor_name = "FrontCameraWide"
        sensor_folder = os.path.join(directory_to_datasets, dataset_name, subset_name, sensor_name.lower())
        solution_filepath = os.path.join(directory_to_datasets, dataset_name, subset_name, "solution.json")
        solution = Solution(**json.load(open(solution_filepath)))
        camera_feature_noise = 1.0  # px
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
        deviation = Transform(  # initial guess deviation from solution
            rotation=Rotation.from_euler("xyz", np.array([1.0, 2.0, 3.0]), degrees=True),
            translation=np.array([0.1, 0.1, 0.1]),
        )
        camera.features[["u", "v"]] = features[["u", "v"]].copy() + rng.normal(0.0, camera_feature_noise, size=features[["u", "v"]].shape)
        initial_guess = Frame(name="InitialGuess", transform=Transform.from_dict(solution.devices[sensor_name].extrinsics) @ deviation)
        initial_guess.add_child(camera)
        vehicle = Vehicle(name="sensitivity analysis vehicle container", system_configuration=dataset_name)
        vehicle.add_child(initial_guess)
        optimizer = ExtendedSparseOptimizer(vehicle=vehicle)
        optimizer.add_photogrammetry(photogrammetry=read_obc(os.path.join(get_dataset_directory(), dataset_name, subset_name, "photogrammetry.obc")))
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
        result.append(np.concatenate([precision_rotation.as_rotvec("xyz"), precision_translation]))  # to milli
    result = np.array(result) * 1e3  # to milli
    print(result)
    ax_left = plt.subplot(121)
    ax_left.grid(True)
    ax_left.plot(result[:, :3])
    ax_left.legend(["$\\omega_x$", "$\\omega_y$", "$\\omega_z$"])
    ax_left.yaxis.set_label_text("rotation $\\sigma$ in $mrad$")
    ax_left.xaxis.set_label_text("distance")
    ax_left.xaxis.set_ticks([0, 1, 2, 3, 4, 5])
    ax_left.xaxis.set_ticklabels([10, 12, 14, 16, 18, 20])
    ax_right = plt.subplot(122)
    ax_right.yaxis.tick_right()
    ax_right.yaxis.set_label_position("right")
    ax_right.grid(True)
    ax_right.plot(result[:, 3:])
    ax_right.legend(["$t_x$", "$t_y$", "$t_z$"])
    ax_right.yaxis.set_label_text("translation $\\sigma$ in $mm$")
    ax_right.xaxis.set_label_text("distance")
    ax_right.xaxis.set_ticks([0, 1, 2, 3, 4, 5])
    ax_right.xaxis.set_ticklabels([10, 12, 14, 16, 18, 20])

    plt.show(block=True)


if __name__ == "__main__":
    run()
