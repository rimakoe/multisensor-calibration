from datatypes import *
from utils import *
import g2opy.g2opy as g2o
import json
import matplotlib.pyplot as plt


class ExtendedSparseOptimizer(g2o.SparseOptimizer):
    """TODO"""

    def __init__(self, vehicle: Vehicle, algorithm: g2o.OptimizationAlgorithm):
        super().__init__()
        self.set_algorithm(algorithm)
        self.vehicle = vehicle
        self.__convention_transform = SE3(rotation=Rotation.from_euler("YXZ", [90, 0, -90], degrees=True))

    def add_frame(self, frame: Frame):
        raise NotImplementedError("Not implemented")

    def add_characteristic_points(self):
        raise NotImplementedError("Not implemented")

    def add_edge_minimization(self):
        raise NotImplementedError("Not implemented")

    def add_gicp(self):
        raise NotImplementedError("Not implemented")

    def add_photogrammetry(self, photogrammetry: pd.DataFrame):
        for _, row in photogrammetry.iterrows():
            marker_id: int = int(row["id"])
            marker_point: np.ndarray = row[["x", "y", "z"]].to_numpy()
            vertex_photogrammetry = g2o.VertexPointXYZ()
            vertex_photogrammetry.set_id(marker_id)
            vertex_photogrammetry.set_estimate(marker_point)
            vertex_photogrammetry.set_fixed(True)
            self.add_vertex(vertex_photogrammetry)

    def add_sensors(self, frame: Frame = None):
        if frame is None:
            frame = self.vehicle
        if type(frame) is Camera:
            self.add_reprojection_error_minimization(frame)
            return
        for child in frame.children:
            self.add_sensors(child)
        return

    def add_reprojection_error_minimization(self, camera: Camera):
        camera_parameter = g2o.CameraParameters(np.mean(camera.intrinsics.focal_length), camera.intrinsics.principal_point, 0)
        camera_parameter.set_id(camera.id)
        self.add_parameter(camera_parameter)

        vertex = g2o.VertexSE3Expmap()
        vertex.set_id(camera.id)
        vertex.set_fixed(False)
        vertex.set_estimate((camera.parent.transform @ self.__convention_transform).inverse().as_eigen())
        self.add_vertex(vertex)
        camera.vertex = vertex

        for _, row in camera.features.iterrows():
            marker_id: int = int(row["id"])
            sigma = 1.0  # TODO integrate uncertainty
            edge = g2o.EdgeProjectXYZ2UV()
            edge.set_vertex(0, self.vertex(marker_id))
            edge.set_vertex(1, self.vertex(camera.id))
            edge.set_measurement(row[["x", "y"]].to_numpy())
            edge.set_information(np.identity(2) / (sigma**2))  # no effect right now
            edge.set_parameter_id(0, camera.id)
            edge.set_id(camera.id + marker_id * 10)
            self.add_edge(edge)

    def __integrate_result(self, frame: Frame, id, result: SE3):
        if type(frame) is Camera and frame.id == id:
            frame.transform = frame.parent.transform.inverse() @ result @ self.__convention_transform.inverse()
            return
        for child in frame.children:
            self.__integrate_result(child, id, result)

    def optimize(self, iterations=1000):
        super().optimize(iterations)
        for id, vertex in self.vertices().items():
            if type(vertex) is not g2o.VertexSE3Expmap:
                continue
            if id >= 1000 and id < 2000:
                result = SE3(
                    translation=vertex.estimate().inverse().translation(),
                    rotation=Rotation.from_matrix(vertex.estimate().inverse().rotation().matrix()),
                )
                self.__integrate_result(self.vehicle, id, result)

    def __repr__(self):
        pass

    def view_graph(self):
        for id, vertex in self.vertices().items():
            print(f"{id}: {vertex}")
        for edge in self.edges():
            print(f"{edge}")


class VehicleFactory:
    def __init__(self, directory_to_datasets):
        self.directory_to_datasets = directory_to_datasets

    def create(self, dataset_name: str, deviation: SE3 = SE3()) -> Tuple[Vehicle, Solution]:
        solution_filepath = os.path.join(self.directory_to_datasets, dataset_name, "solution.json")
        solution = Solution(**json.load(open(solution_filepath)))
        vehicle = Vehicle(name="Vehicle", system_configuration=dataset_name)
        camera_id = 1000
        for name, data in solution.devices.items():
            initial_guess = Frame(name="InitialGuess", transform=SE3.from_dict(data.extrinsics) @ deviation)
            sensor_folder = os.path.join(self.directory_to_datasets, dataset_name, name.lower())
            if "lidar" in name.lower():
                initial_guess.add_child(
                    Lidar(
                        name=name,
                        data=PointCloud.from_path(os.path.join(sensor_folder, name.lower() + ".pcd")),
                    )
                )
                vehicle.add_child(initial_guess)
            if "camera" in name.lower():
                features = pd.DataFrame.from_dict(
                    json.load(open(os.path.join(sensor_folder, "detections.json"))),
                )
                features["x"] += 0.5
                features["y"] += 0.5
                initial_guess.add_child(
                    Camera(
                        name=name,
                        id=camera_id,
                        data=Image.open(os.path.join(sensor_folder, name.lower() + ".bmp")),
                        intrinsics=Camera.Intrinsics.from_json(
                            os.path.join(sensor_folder, "camera_info.json"),
                        ),
                        features=features,
                    )
                )
                vehicle.add_child(initial_guess)
                camera_id += 1
        return vehicle, solution


def fit_plane_ransac(name: str, points: pd.DataFrame, iterations: int = 1000) -> Plane:
    missing = [required_column for required_column in ["x", "y", "z"] if required_column not in points.columns]
    assert len(missing) == 0, f"missing column(s): {missing}"
    points = points[["x", "y", "z"]]
    best_normal = None
    best_offset = None
    best_inlier = None
    best_number_of_inliers = 0
    for i in range(iterations):
        samples = points.sample(3)
        v1 = samples.iloc[1] - samples.iloc[0]
        v2 = samples.iloc[2] - samples.iloc[0]
        offset = samples.iloc[0]
        normal = np.cross(v1, v2)
        normal /= np.linalg.norm(normal, 2)
        distances = np.abs(np.dot(points - offset, normal.reshape((3, 1))))
        inlier = distances < 0.01
        if len(distances[inlier]) > best_number_of_inliers:
            best_inlier = inlier
            best_number_of_inliers = len(distances[inlier])
            best_normal = normal
            best_offset = np.median(points[inlier], axis=0)
    return Plane.from_normal_offset(name, best_normal, best_offset), best_inlier


def fit_plane(name: str, points) -> Plane:
    if type(points) is pd.DataFrame:
        missing = [required_column for required_column in ["x", "y", "z"] if required_column not in points.columns]
        assert len(missing) == 0, f"missing column(s): {missing}"
        points = points[["x", "y", "z"]].to_numpy()
    if type(points) is not np.ndarray:
        raise TypeError("expected np.ndarray")
    centroid = np.mean(points, axis=0)
    pts_centered = points - centroid
    _, _, vh = np.linalg.svd(pts_centered)
    normal = vh[-1]
    normal /= np.linalg.norm(normal, 2)
    return Plane.from_normal_offset(name=name, normal=normal, offset=centroid)


def find_planes(photogrammetry: pd.DataFrame) -> List[Plane]:
    planes: List[Plane] = []
    counter = 0
    cache = []
    for _, row in photogrammetry.iterrows():
        counter += 1
        cache.append(row)
        if counter % 6 == 0:
            planes.append(fit_plane("plane", pd.concat(cache, axis=1).T))
            counter = 0
            cache = []
    return planes


def main(dataset_name: str = "C4L5"):
    directory_to_datasets = "/" + os.path.join("home", "workspace", "datasets")
    factory = VehicleFactory(directory_to_datasets)
    vehicle, solution = factory.create(
        dataset_name,
        deviation=SE3(
            rotation=Rotation.from_euler("xyz", [3, 2, 1], degrees=True),
            translation=np.array([0.5, 0.5, 1.0]),
        ),
    )
    optimizer = ExtendedSparseOptimizer(
        vehicle=vehicle,
        algorithm=g2o.OptimizationAlgorithmLevenberg(g2o.BlockSolverSE3(g2o.LinearSolverDenseSE3())),
    )
    optimizer.add_photogrammetry(photogrammetry=read_obc("/" + os.path.join("home", "workspace", "datasets", "C4L5", "photogrammetry.obc")))
    optimizer.add_sensors()
    optimizer.initialize_optimization()
    optimizer.set_verbose(True)
    optimizer.optimize(10000)

    print(optimizer.vehicle.as_dataframe(only_leafs=True, relative_coordinates=False))

    return vehicle, {}


from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors


def lidar_calibration():
    directory_to_datasets = "/" + os.path.join("home", "workspace", "datasets")
    factory = VehicleFactory(directory_to_datasets)
    deviation = SE3(
        rotation=Rotation.from_euler("xyz", [3, 2, 1], degrees=True),
        translation=np.array([0.1, 0.1, 0.1]),
    )
    vehicle, solution = factory.create(
        "C4L5",
        deviation=deviation,
    )
    lidar: Lidar = vehicle.get_frame("RefLidar")
    if not lidar:
        raise NameError("lidar not found via given name")
    pointcloud = pd.DataFrame(lidar.data.pc_data)
    pointcloud = pointcloud[np.isfinite(pointcloud).all(1)]
    pointcloud.dropna()
    pointcloud = lidar.parent.transform.apply(pointcloud)
    pointcloud = pointcloud[pointcloud["z"] > 0.5]
    pointcloud = pointcloud[np.linalg.norm(pointcloud[["x", "y", "z"]], 2, axis=1) < 5.0]
    pointcloud = pointcloud[np.linalg.norm(pointcloud[["x", "y", "z"]], 2, axis=1) > 2.0]

    kmeans = KMeans(n_clusters=8, random_state=0, n_init="auto").fit(pointcloud[["x", "y", "z"]].to_numpy())
    pointcloud["label"] = kmeans.labels_
    new_pointcloud = pointcloud
    photogrammetry_planes = find_planes(photogrammetry=read_obc("/" + os.path.join("home", "workspace", "datasets", "C4L5", "photogrammetry.obc")))
    photogrammetry_centroids = [plane.get_offset() for plane in photogrammetry_planes]

    sensor_planes: List[Plane] = []
    for label in np.unique(new_pointcloud["label"]):
        module_pointcloud = new_pointcloud[new_pointcloud["label"] == label]
        plane, inlier = fit_plane_ransac("plane", module_pointcloud)
        sensor_planes.append(plane)
        module_pointcloud = module_pointcloud[~inlier]
        plane, inlier = fit_plane_ransac("plane", module_pointcloud)
        sensor_planes.append(plane)
        module_pointcloud = module_pointcloud[~inlier]
        plane, inlier = fit_plane_ransac("plane", module_pointcloud)
        sensor_planes.append(plane)
    sensor_centroids = [plane.get_offset() for plane in sensor_planes]

    optimizer = g2o.SparseOptimizer()
    optimizer.set_algorithm(g2o.OptimizationAlgorithmLevenberg(g2o.BlockSolverSE3(g2o.LinearSolverDenseSE3())))
    # terminate_action = g2o.SparseOptimizerTerminateAction()
    # terminate_action.set_gain_threshold(10e-15)
    # optimizer.add_post_iteration_action(terminate_action)

    vehicle_vertex = g2o.VertexSE3()
    vehicle_vertex.set_id(0)
    vehicle_vertex.set_estimate(g2o.Isometry3d(lidar.parent.transform.rotation.as_matrix(), lidar.parent.transform.translation))
    vehicle_vertex.set_fixed(True)
    optimizer.add_vertex(vehicle_vertex)

    lidar_vertex = g2o.VertexSE3()
    lidar_vertex.set_id(1)
    lidar_vertex.set_estimate(g2o.Isometry3d())
    lidar_vertex.set_fixed(False)
    optimizer.add_vertex(lidar_vertex)

    neighbor = NearestNeighbors(n_neighbors=2, radius=0.4, n_jobs=4)
    neighbor.fit(photogrammetry_centroids)
    for sensor_idx, photogrammetry_idxs in enumerate(neighbor.kneighbors(sensor_centroids, 2)[1]):
        for photogrammetry_idx in photogrammetry_idxs:
            pplane: Plane = photogrammetry_planes[photogrammetry_idx]
            splane: Plane = sensor_planes[sensor_idx]

            # splane.transform = deviation @ pplane.transform  # HACK for absolute groundtruth matching here to check if solver converges

            normal0 = pplane.get_normal()
            if np.linalg.norm(pplane.get_offset() + normal0 * 0.1, 2) > np.linalg.norm(pplane.get_offset(), 2):
                normal0 *= -1.0

            normal1 = splane.get_normal()
            if np.linalg.norm(splane.get_offset() + normal1 * 0.1, 2) > np.linalg.norm(splane.get_offset(), 2):
                normal1 *= -1.0

            measurement = g2o.EdgeGICP()
            measurement.normal0 = normal0
            measurement.pos0 = pplane.get_offset()
            measurement.normal1 = normal1
            measurement.pos1 = splane.get_offset()
            measurement.prec0(0.01)
            measurement.prec1(0.01)

            edge = g2o.EdgeVVGicp()
            edge.set_vertex(0, vehicle_vertex)
            edge.set_vertex(1, lidar_vertex)
            edge.set_measurement(measurement)
            edge.set_information(np.linalg.inv(np.eye(3) / 0.01))
            edge.set_robust_kernel(g2o.RobustKernelHuber(0.01))
            optimizer.add_edge(edge)
            break  # for now just take the first one here. That is the closest and will most of the time be true. If not it is a seperate problem...

    optimizer.initialize_optimization()
    optimizer.set_verbose(True)
    optimizer.optimize(10000)

    # The lidar optimization happens in vehicle coordinates for the clustering and ground removal to be easier
    global_tranform = SE3(
        rotation=Rotation.from_matrix(optimizer.vertex(1).estimate().rotation().matrix()),
        translation=optimizer.vertex(1).estimate().translation(),
    )
    # Calculate the relative pose from the initial guess from that global result
    lidar.transform = lidar.parent.transform.inverse() @ global_tranform
    print("estimated improvement: " + str(lidar.transform))
    print("estimated deviation (inv improvement): " + str(lidar.transform.inverse()))

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # draw photogrammetry normals
    for plane in photogrammetry_planes:
        normal = plane.get_normal()
        offset = plane.get_offset()
        if np.linalg.norm(offset + normal * 0.1, 2) > np.linalg.norm(offset, 2):
            normal *= -1.0
        ax.quiver(offset[0], offset[1], offset[2], normal[0], normal[1], normal[2], length=0.5)

    # draw sensor normals
    for plane in sensor_planes:
        plane.transform = lidar.transform @ plane.transform
        offset = plane.get_offset()
        normal = plane.get_normal()
        if np.linalg.norm(offset + normal * 0.1, 2) > np.linalg.norm(offset, 2):
            normal *= -1.0
        ax.quiver(offset[0], offset[1], offset[2], normal[0], normal[1], normal[2], length=0.5, color="red")

    # draw scatter
    for label in np.unique(new_pointcloud["label"]):
        ax.scatter(
            xs=new_pointcloud[new_pointcloud["label"] == label]["x"][0:-1:10],
            ys=new_pointcloud[new_pointcloud["label"] == label]["y"][0:-1:10],
            zs=new_pointcloud[new_pointcloud["label"] == label]["z"][0:-1:10],
        )

    ax.set_aspect("equal")
    plt.show(block=True)


if __name__ == "__main__":
    lidar_calibration()
    # vehicle, report = main()
