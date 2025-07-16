from datatypes import *
from utils import *
import g2opy.g2opy as g2o

# here the actual calibration happens

# TODO 1. feature detection (gather the marker xy coordinates in the image and the plane normals in 3D)

# TODO 2.1 reprojection error minimization

# TODO 2.2 plane normal fitting

# TODO 3. combined feature matching (extract edges somehow -> fit edges from image and edges in pointcloud together)

# TODO evaluate the result on accuracy, precision and reliablitiy (e.g. noise variations)


def view_graph(optimizer: g2o.SparseOptimizer):
    for id, vertex in optimizer.vertices().items():
        print(f"{id}: {vertex}")
    for edge in optimizer.edges():
        print(edge)


def init_cameras(optimizer: g2o.SparseOptimizer, frame: Frame, path: List[Frame] = None) -> Tuple[g2o.VertexSE3Expmap, g2o.CameraParameters]:
    if not path:
        path = []
    output = []
    current_path = path + [frame]
    if type(frame) is Camera:
        camera_parameter = g2o.CameraParameters(frame.intrinsics.focal_length, frame.intrinsics.principal_point, 0)
        camera_parameter.set_id(len(optimizer.vertices()))
        optimizer.add_parameter(camera_parameter)

        vertex = g2o.VertexSE3Expmap()
        vertex.set_id(len(optimizer.vertices()))
        vertex.set_fixed(False)
        initial_transform = SE3()
        for f in path:  # important to use the path here, since current path includes the camera transform itself
            initial_transform = initial_transform @ f.transform
        vertex.set_estimate(initial_transform.as_eigen())  # might has to be the inverse here for some reason. Investigate!
        optimizer.add_vertex(vertex)

        output.append((vertex, camera_parameter))

    for child in frame.children:
        output.extend(init_cameras(optimizer, child, current_path))
    return output


def calibrate(vehicle: Vehicle) -> dict:
    optimizer = g2o.SparseOptimizer()
    linear_solver = g2o.LinearSolverDenseSE3()
    block_solver = g2o.BlockSolverSE3(linear_solver)
    algorithm = g2o.OptimizationAlgorithmLevenberg(block_solver)
    optimizer.set_algorithm(algorithm)

    # recursively initializes the graph
    camera = init_cameras(optimizer, vehicle)
    # view_graph(optimizer)
    report = {}
    return report


if __name__ == "__main__":
    # marker_pointcloud = read_obc(os.path.join([dataset_directory, dataset_name, "photogrammetry", "markers.obc"]))
    directory_to_datasets = os.path.join(os.getcwd(), "datasets")
    dataset = "C4L5"
    solution_filepath = os.path.join(directory_to_datasets, dataset, "solution.json")
    solution = Solution(**json.load(open(solution_filepath)))
    vehicle = Vehicle(name="Vehicle", system_configuration="CL-0.5")

    deviation = SE3(rotation=Rotation.from_euler("xyz", [3, 2, 1], degrees=True), translation=np.array([0.1, 0.1, 0.1]))

    for name, data in solution.devices.items():
        initial_guess = Frame(name="InitialGuess", transform=SE3.from_dict(data.extrinsics) @ deviation)
        sensor_folder = os.path.join(directory_to_datasets, dataset, name.lower())
        if "lidar" in name.lower():
            initial_guess.add_child(
                Lidar(
                    name=name,
                    data=PointCloud.from_path(os.path.join(sensor_folder, name.lower() + ".pcd")),
                )
            )
            vehicle.add_child(initial_guess)
        if "camera" in name.lower():
            initial_guess.add_child(
                Camera(
                    name=name,
                    data=Image.open(os.path.join(sensor_folder, name.lower() + ".bmp")),
                    # intrinsics=Camera.Intrinsics.from_cfg(os.path.join(sensor_folder, "intrinsics.cfg")),
                    features=pd.DataFrame.from_dict(open(os.path.join(sensor_folder, "detections.json"))),  # Groundtruth detections from simulation
                )
            )
            vehicle.add_child(initial_guess)

    print("\nBefore Optimization")
    print(vehicle.as_dataframe(only_leafs=True))

    calibrate(vehicle)

    print("\nAfter Optimization")
    print(vehicle.as_dataframe(only_leafs=True))
