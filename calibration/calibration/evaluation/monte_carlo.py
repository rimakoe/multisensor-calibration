import pandas as pd
import numpy as np
import tqdm
from scipy.spatial.transform import Rotation
from calibration.core import VehicleFactory, main
from calibration.datatypes import Transform

rng = np.random.default_rng(0)  # one reproducible stream

if __name__ == "__main__":
    dataset_name = "C1L1_triple_module_front"
    reports = []
    for i in tqdm.tqdm(range(100)):
        factory = VehicleFactory()
        vehicle, solution = factory.create(
            dataset_name,
            sensor_whitelist=["FrontCameraWide"],
            deviation=Transform(
                rotation=Rotation.from_euler("xyz", np.array([0, 0, 0]) + rng.normal(0, 1, size=3), degrees=True),
                translation=np.array([0.0, 0.0, 0.0]) + rng.normal(0, 0.1, size=3),
            ),
            camera_feature_noise=1.0,
            lidar_feature_noise=0.01,
        )
        _, report = main(vehicle=vehicle, dataset_name=dataset_name, silent=True, plot=False)
        reports.append(report)
    df = pd.DataFrame(reports)
    mean = np.mean(df["p-value"])
    std = np.std(df["p-value"])
    print(f"p-values: mean {mean}\t std: {std}")
