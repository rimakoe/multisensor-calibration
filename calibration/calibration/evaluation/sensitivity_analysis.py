import os
from calibration.datatypes import *
from calibration.core import VehicleFactory
from calibration.core import main as aio_calibration
import tqdm

deviation = SE3(
    rotation=Rotation.from_euler("xyz", [3, 3, 3], degrees=True),
    translation=np.array([0.1, 0.1, 0.1]),
)


def feature_sensitivity():
    directory_to_datasets = "/" + os.path.join("home", "workspace", "datasets")
    factory = VehicleFactory(directory_to_datasets)
    dataset_name = "C1"
    vehicle, solution = factory.create(
        dataset_name=dataset_name,
        deviation=deviation,
    )
    _convention_transform = SE3(rotation=Rotation.from_euler("YXZ", [90, 0, -90], degrees=True))
    df = pd.DataFrame()
    for i in tqdm.tqdm(range(10)):
        vehicle, report = aio_calibration(vehicle=vehicle, dataset_name=dataset_name, silent=True, plot=False)
        df = pd.concat([df, pd.DataFrame(report, index=[i])])
    print(df)


if __name__ == "__main__":
    feature_sensitivity()
