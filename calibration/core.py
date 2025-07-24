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
            sigma = 1.0  # TODO
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


def fit_plane_ransac(name: str, points: pd.DataFrame) -> Plane:
    missing = [required_column for required_column in ["x", "y", "z"] if required_column not in points.columns]
    assert len(missing) == 0, f"missing column(s): {missing}"
    points = points[["x", "y", "z"]]
    best_normal = None
    best_offset = None
    best_number_of_inliers = 0
    for i in range(1000):
        samples = points.sample(3)
        v1 = samples.iloc[1] - samples.iloc[0]
        v2 = samples.iloc[2] - samples.iloc[0]
        offset = samples.iloc[0]
        normal = np.cross(v1, v2)
        normal /= np.linalg.norm(normal, 2)
        distances = np.abs(np.dot(points - offset, normal.reshape((3, 1))))
        inlier = distances < 0.05
        if len(distances[inlier]) > best_number_of_inliers:
            best_number_of_inliers = len(distances[inlier])
            best_normal = normal
            best_offset = np.median(points[inlier], axis=0)
    return Plane.from_normal_offset(name, best_normal, best_offset)


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

    print(optimizer.vehicle.as_dataframe(only_leafs=True, relative_coordinates=True))

    return vehicle, {}


from sklearn.cluster import KMeans


def seperate_modules(pointcloud: pd.DataFrame):
    kmeans = KMeans(n_clusters=8, random_state=0, n_init="auto").fit(pointcloud[["x", "y", "z"]].to_numpy())
    pointcloud["label"] = kmeans.labels_
    return kmeans.cluster_centers_, pointcloud


def lidar_calibration():
    directory_to_datasets = "/" + os.path.join("home", "workspace", "datasets")
    factory = VehicleFactory(directory_to_datasets)
    vehicle, solution = factory.create(
        "C4L5",
        deviation=SE3(
            rotation=Rotation.from_euler("xyz", [0, 0, 0], degrees=True),
            translation=np.array([0.0, 0.0, 0.0]),
        ),
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

    clusters, new_pointcloud = seperate_modules(pointcloud)

    planes = find_planes(photogrammetry=read_obc("/" + os.path.join("home", "workspace", "datasets", "C4L5", "photogrammetry.obc")))

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    for plane in planes:
        normal = plane.get_normal()
        offset = plane.get_offset()
        if np.linalg.norm(offset + normal * 0.1, 2) > np.linalg.norm(offset, 2):
            normal *= -1.0
        ax.quiver(offset[0], offset[1], offset[2], normal[0], normal[1], normal[2], length=0.5)
    # for cluster in clusters:
    #    ax.scatter(xs=cluster[0], ys=cluster[1], zs=cluster[2], s=50, marker="x")
    for label in np.unique(new_pointcloud["label"]):
        plane = fit_plane_ransac("plane", new_pointcloud[new_pointcloud["label"] == label])
        offset = plane.get_offset()
        normal = plane.get_normal()
        if np.linalg.norm(offset + normal * 0.1, 2) > np.linalg.norm(offset, 2):
            normal *= -1.0
        ax.quiver(offset[0], offset[1], offset[2], normal[0], normal[1], normal[2], length=0.5, color="red")
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
