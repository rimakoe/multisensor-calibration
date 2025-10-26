import pandas as pd
import numpy as np
import tqdm
from scipy.spatial.transform import Rotation
from calibration.core import *
from calibration.datatypes import Transform
from calibration.plots import *

ITERATIONS = 1000
INITIAL_GUESS_ROTATION_NOISE = 0.5  # sigma [degrees]
INITIAL_GUESS_TRANSLATION_NOISE = 0.05  # sigma [meter]

rng = np.random.default_rng(0)  # one reproducible stream

if __name__ == "__main__":
    dataset_name = "C1L1_box_everywhere"
    reports = []
    for i in tqdm.tqdm(range(ITERATIONS)):
        factory = VehicleFactory()
        vehicle, solution = factory.create(
            dataset_name=dataset_name,
            sensor_whitelist=["RefLidar"],
            deviation=Transform(
                rotation=Rotation.from_euler("xyz", np.array([0, 0, 0]) + rng.normal(0, INITIAL_GUESS_ROTATION_NOISE, size=3), degrees=True),
                translation=np.array([0.0, 0.0, 0.0]) + rng.normal(0.0, INITIAL_GUESS_TRANSLATION_NOISE, size=3),
            ),
            camera_feature_noise=1.0,  # px
            lidar_feature_noise=0.01,  # m
            use_ideal_features=True,
        )
        _, report = main(vehicle=vehicle, dataset_name=dataset_name, silent=False, plot=False)
        if report is None:
            continue
        reports.append(report)
    df = pd.DataFrame(reports)
    chi2_mean = np.mean(df["chi2_score"])
    chi2_std = np.std(df["chi2_score"])
    robust_median = np.median(df["robust_score"])
    plot_convergence(df=df, target="chi2_score", reference=report["chi2_score_center"], config="mean")
    plot_convergence(df=df, target="robust_score", reference=1.0, config="median")
    print(f"chi2_score: mean {chi2_mean}\t std: {chi2_std}")
    print(f"robust_score: median {robust_median}\t")
