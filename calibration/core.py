from datatypes import *
from utils import *

# here the actual calibration happens

# TODO 1. feature detection (gather the marker xy coordinates in the image and the plane normals in 3D)

# TODO 2.1 reprojection error minimization

# TODO 2.2 plane normal fitting

# TODO 3. combined feature matching (extract edges somehow -> fit edges from image and edges in pointcloud together)

# TODO evaluate the result on accuracy, precision and reliablitiy (e.g. noise variations)


def get_detections(filepath: str) -> pd.DataFrame:

    return pd.DataFrame()


if __name__ == "__main__":
    dataset_directory = os.path.join([os.path.dirname(os.path.dirname(__file__)), "datasets"])
    dataset_name = "C1L1"  # ONE CAMERA ONE LIDAR
    marker_pointcloud = read_obc(os.path.join([dataset_directory, dataset_name, "photogrammetry", "markers.obc"]))
    camera = Camera(name="TopViewCameraFront", data=Image.open(image_filepath), features=get_detections(camera_folder))
