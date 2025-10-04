import tqdm
from calibration.datatypes import *
from calibration.core import VehicleFactory, main

deviation = Transform(
    rotation=Rotation.from_euler("xyz", [3, 3, 3], degrees=True),
    translation=np.array([0.1, 0.1, 0.1]),
)
# deviation = SE3(
#     rotation=Rotation.from_euler("xyz", [0, 0, 0], degrees=True),
#     translation=np.array([0.0, 0.0, 0.0]),
# )

if __name__ == "__main__":
    dataset_name = "C1"
    df = pd.DataFrame()
    for i in tqdm.tqdm(range(100)):
        factory = VehicleFactory()
        vehicle, solution = factory.create(dataset_name, deviation=deviation, camera_feature_noise=1.0)
        vehicle, report = main(vehicle, dataset_name, silent=True, plot=False)
        df = pd.concat([df, pd.DataFrame(report, index=[i])])
    print(df)
