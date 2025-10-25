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
    for i in tqdm.tqdm(range(500)):
        factory = VehicleFactory()
        vehicle, solution = factory.create(
            dataset_name,
            sensor_whitelist=["FrontCameraWide"],
            deviation=Transform(
                rotation=Rotation.from_euler("xyz", np.array([0, 0, 0]) + rng.normal(0, 1, size=3), degrees=True),
                translation=np.array([0.0, 0.0, 0.0]) + rng.normal(0.0, 0.05, size=3),
            ),
            camera_feature_noise=1.0,  # px
            lidar_feature_noise=0.01,  # m
        )
        _, report = main(vehicle=vehicle, dataset_name=dataset_name, silent=False, plot=False)
        if report is None:
            continue
        reports.append(report)
    df = pd.DataFrame(reports)
    mean = np.mean(df["chi2_score"])
    std = np.std(df["chi2_score"])
    print(f"chi2-score: mean {mean}\t std: {std}")
