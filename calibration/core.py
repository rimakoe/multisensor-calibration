from datatypes import *
from utils import *
import g2opy.g2opy as g2o
import json
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN
from sklearn.neighbors import NearestNeighbors
import tqdm

# only for groundtruth hack
deviation = SE3(
    rotation=Rotation.from_euler("xyz", [3, 3, 3], degrees=True),
    translation=np.array([0.1, 0.1, 0.1]),
)


class ExtendedSparseOptimizer(g2o.SparseOptimizer):
    """TODO"""

    def __init__(self, vehicle: Vehicle, algorithm: g2o.OptimizationAlgorithm):
        super().__init__()
        self.set_algorithm(algorithm)
        self.vehicle = vehicle
        self.__convention_transform = SE3(rotation=Rotation.from_euler("YXZ", [90, 0, -90], degrees=True))

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
            # self.add_reprojection_error_minimization(frame)
            return
        if type(frame) is Lidar:
            self.add_point_to_plane_error_minimization(frame)
            return
        for child in frame.children:
            self.add_sensors(child)
        return

    def add_reprojection_error_minimization(self, camera: Camera) -> None:
        camera_parameter = g2o.CameraParameters(np.mean(camera.intrinsics.focal_length), camera.intrinsics.principal_point, 0)
        camera_parameter.set_id(camera.id)
        self.add_parameter(camera_parameter)

        vertex = g2o.VertexSE3Expmap()
        vertex.set_id(camera.id)
        vertex.set_fixed(False)
        vertex.set_estimate((camera.parent.transform @ self.__convention_transform).inverse().as_eigen())
        self.add_vertex(vertex)
        camera.vertex = vertex

        # Try to make the result relative, but the edge does not care. For this to work the photogrammetrty vertices need to be transformed in sensor coordinates. Since this makes not too much sense we leave it like this.
        # init_vertex = g2o.VertexSE3Expmap()
        # init_vertex.set_id(camera.id + 100)
        # init_vertex.set_fixed(True)
        # init_vertex.set_estimate((camera.parent.transform @ self.__convention_transform).inverse().as_eigen())
        # self.add_vertex(init_vertex)

        # edge = g2o.EdgeSE3Expmap()
        # edge.set_vertex(0, init_vertex)
        # edge.set_vertex(1, vertex)
        # edge.set_measurement(g2o.SE3Quat())
        # edge.set_id(camera.id * 1000 + 900)
        # self.add_edge(edge)

        for _, row in camera.features.iterrows():
            marker_id: int = int(row["id"])
            sigma = 1.0  # TODO integrate uncertainty
            edge = g2o.EdgeProjectXYZ2UV()
            edge.set_vertex(0, self.vertex(marker_id))
            edge.set_vertex(1, self.vertex(camera.id))
            edge.set_measurement(row[["x", "y"]].to_numpy())
            edge.set_information(np.identity(2) / (sigma**2))  # no effect right now
            edge.set_parameter_id(0, camera.id)
            edge.set_id(camera.id * 1000 + marker_id * 10)
            self.add_edge(edge)

    def add_point_to_plane_error_minimization(self, lidar: Lidar) -> None:
        # TODO this is calculated for every lidar, make this only once outside somewhere
        photogrammetry = read_obc("/" + os.path.join("home", "workspace", "datasets", "C4L5", "photogrammetry.obc"))
        photogrammetry_planes: List[Plane] = []
        counter = 0
        cache = []
        for _, row in photogrammetry.iterrows():
            counter += 1
            cache.append(row)
            if counter % 6 == 0:
                photogrammetry_planes.append(fit_plane("plane", pd.concat(cache, axis=1).T))
                counter = 0
                cache = []
        photogrammetry_centroids = [plane.get_offset() for plane in photogrammetry_planes]

        offset = g2o.ParameterSE3Offset()
        offset.set_id(0)
        self.add_parameter(offset)

        initial_guess_vertex = g2o.VertexSE3()
        initial_guess_vertex.set_id(lidar.id + 100)
        initial_guess_vertex.set_estimate(g2o.Isometry3d())
        initial_guess_vertex.set_fixed(True)
        self.add_vertex(initial_guess_vertex)

        lidar_vertex = g2o.VertexSE3()
        lidar_vertex.set_id(lidar.id)
        lidar_vertex.set_estimate(g2o.Isometry3d(lidar.parent.transform.rotation.as_matrix(), lidar.parent.transform.translation))
        lidar_vertex.set_fixed(False)
        self.add_vertex(lidar_vertex)

        neighbor = NearestNeighbors(n_neighbors=2, radius=0.4, n_jobs=4)
        neighbor.fit(photogrammetry_centroids)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

        # draw photogrammetry normals
        for plane in photogrammetry_planes:
            normal = plane.get_normal()
            offset = plane.get_offset()
            if np.linalg.norm(offset + normal * 0.1, 2) < np.linalg.norm(offset, 2):
                normal *= -1.0
            ax.quiver(offset[0], offset[1], offset[2], normal[0], normal[1], normal[2], length=0.5)

        # draw sensor normals
        for plane in lidar.features:
            plane.transform = lidar.transform @ plane.transform
            offset = plane.get_offset()
            normal = plane.get_normal()
            if np.linalg.norm(offset + normal * 0.1, 2) < np.linalg.norm(offset, 2):
                normal *= -1.0
            ax.quiver(offset[0], offset[1], offset[2], normal[0], normal[1], normal[2], length=0.5, color="red")

        ax.set_aspect("equal")
        plt.show(block=True)

        for sensor_idx, photogrammetry_idxs in enumerate(neighbor.kneighbors([plane.get_offset() for plane in lidar.features], 2)[1]):
            for photogrammetry_idx in photogrammetry_idxs:
                pplane: Plane = photogrammetry_planes[photogrammetry_idx]
                splane: Plane = lidar.features[sensor_idx]

                splane.transform.translation = (
                    deviation @ pplane.transform
                ).translation  # HACK for absolute groundtruth matching here to check if solver converges
                splane.transform.rotation = (
                    deviation @ pplane.transform
                ).rotation  # HACK for absolute groundtruth matching here to check if solver converges

                normal0 = pplane.get_normal()
                if np.linalg.norm(pplane.get_offset() + normal0 * 0.1, 2) < np.linalg.norm(pplane.get_offset(), 2):
                    normal0 *= -1.0

                normal1 = splane.get_normal()
                if np.linalg.norm(splane.get_offset() + normal1 * 0.1, 2) < np.linalg.norm(splane.get_offset(), 2):
                    normal1 *= -1.0

                measurement = g2o.EdgeGICP()
                measurement.normal0 = normal0
                measurement.pos0 = np.dot(pplane.get_offset(), normal0) * normal0
                measurement.normal1 = normal1
                measurement.pos1 = np.dot(splane.get_offset(), normal1) * normal1
                # measurement.prec0(0.000001)
                # measurement.prec1(0.000001)

                edge = g2o.EdgeVVGicp()
                edge.set_id(lidar.id * 1000 + photogrammetry_idx)
                edge.set_vertex(0, initial_guess_vertex)
                edge.set_vertex(1, lidar_vertex)
                edge.set_measurement(measurement)
                edge.set_information(np.linalg.inv(np.eye(3) / (0.01**2)))
                # edge.set_robust_kernel(g2o.RobustKernelHuber(0.01))
                # self.add_edge(edge)

                vertex_point = g2o.VertexPointXYZ()
                vertex_point.set_estimate(np.dot(pplane.get_offset(), normal0) * normal0)
                vertex_point.set_id(lidar.id * 1000 + photogrammetry_idx + 300)
                vertex_point.set_fixed(True)
                self.add_vertex(vertex_point)

                edge_xyz = g2o.EdgeSE3PointXYZ()
                edge_xyz.set_id(lidar.id * 1000 + photogrammetry_idx + 200)
                edge_xyz.set_vertex(0, lidar_vertex)
                edge_xyz.set_vertex(1, vertex_point)
                edge_xyz.set_measurement(np.dot(splane.get_offset(), normal1) * normal1)
                edge_xyz.set_parameter_id(0, 0)
                edge_xyz.set_information(np.outer(normal1, normal1))
                # edge_xyz.set_robust_kernel(g2o.RobustKernelHuber(0.01))
                self.add_edge(edge_xyz)

                break  # for now just take the first one here. That is the closest and will most of the time be true. If not it is a seperate problem...

    def __integrate_result(self, frame: Frame, id, result: SE3):
        if type(frame) is Camera and frame.id == id:
            frame.transform = frame.parent.transform.inverse() @ (result.inverse() @ self.__convention_transform.inverse())
            return
        if type(frame) is Lidar and frame.id == id:
            frame.transform = result
            return
        for child in frame.children:
            self.__integrate_result(child, id, result)

    def optimize(self, iterations=1000):
        super().optimize(iterations)
        for id, vertex in self.vertices().items():
            if type(vertex) is not g2o.VertexSE3Expmap and type(vertex) is not g2o.VertexSE3:
                continue
            if id >= 1000 and id < 2000:
                result = SE3(
                    translation=vertex.estimate().translation(),
                    rotation=Rotation.from_matrix(vertex.estimate().rotation().matrix()),
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
        sensor_id = 1000
        for name, data in solution.devices.items():
            initial_guess = Frame(name="InitialGuess", transform=SE3.from_dict(data.extrinsics) @ deviation)
            sensor_folder = os.path.join(self.directory_to_datasets, dataset_name, name.lower())
            if "lidar" in name.lower() and not "front" in name.lower():
                # read the pcd
                data = PointCloud.from_path(os.path.join(sensor_folder, name.lower() + ".pcd"))
                # crop the pointcloud roughly
                pointcloud = pd.DataFrame(data.pc_data)
                pointcloud = pointcloud[np.isfinite(pointcloud).all(1)]
                pointcloud.dropna()
                pointcloud = initial_guess.transform.apply(pointcloud)  # transform data using the initial guess
                pointcloud = pointcloud[pointcloud["z"] > 0.5]
                pointcloud = pointcloud[np.linalg.norm(pointcloud[["x", "y", "z"]], 2, axis=1) < 5.0]
                pointcloud = pointcloud[np.linalg.norm(pointcloud[["x", "y", "z"]], 2, axis=1) > 2.0]
                # cluster the modules
                # kmeans = KMeans(n_clusters=8, random_state=0, n_init="auto").fit(pointcloud[["x", "y", "z"]].to_numpy())
                # pointcloud["label"] = kmeans.labels_
                db = DBSCAN(eps=0.3, min_samples=20).fit(pointcloud)
                pointcloud["label"] = db.labels_

                fig = plt.figure()
                ax = fig.add_subplot(111, projection="3d")

                # draw scatter
                for label in np.unique(pointcloud["label"]):
                    ax.scatter(
                        xs=pointcloud[pointcloud["label"] == label]["x"][0:-1:10],
                        ys=pointcloud[pointcloud["label"] == label]["y"][0:-1:10],
                        zs=pointcloud[pointcloud["label"] == label]["z"][0:-1:10],
                    )

                ax.set_aspect("equal")
                plt.show(block=True)
                # find the planes
                features: List[Plane] = []
                for label in tqdm.tqdm(np.unique(pointcloud["label"])):
                    module_pointcloud = pointcloud[pointcloud["label"] == label]
                    if len(module_pointcloud["x"]) < 100:
                        continue
                    plane, inlier = fit_plane_ransac("plane", module_pointcloud)
                    features.append(plane)
                    module_pointcloud = module_pointcloud[~inlier]
                    if len(module_pointcloud["x"]) < 100:
                        continue
                    plane, inlier = fit_plane_ransac("plane", module_pointcloud)
                    features.append(plane)
                    module_pointcloud = module_pointcloud[~inlier]
                    if len(module_pointcloud["x"]) < 100:
                        continue
                    plane, inlier = fit_plane_ransac("plane", module_pointcloud)
                    features.append(plane)

                initial_guess.add_child(
                    Lidar(
                        name=name,
                        id=sensor_id,
                        data=data,
                        features=features,
                    )
                )
                vehicle.add_child(initial_guess)
                sensor_id += 1
            if "camera" in name.lower():
                features = pd.DataFrame.from_dict(
                    json.load(open(os.path.join(sensor_folder, "detections.json"))),
                )
                features["x"] += 0.5
                features["y"] += 0.5
                initial_guess.add_child(
                    Camera(
                        name=name,
                        id=sensor_id,
                        data=Image.open(os.path.join(sensor_folder, name.lower() + ".bmp")),
                        intrinsics=Camera.Intrinsics.from_json(
                            os.path.join(sensor_folder, "camera_info.json"),
                        ),
                        features=features,
                    )
                )
                vehicle.add_child(initial_guess)
                sensor_id += 1
        return vehicle, solution


def fit_plane_ransac(name: str, points: pd.DataFrame, iterations: int = 100, expected_noise=0.005, linear_refinement=True) -> Plane:
    missing = [required_column for required_column in ["x", "y", "z"] if required_column not in points.columns]
    assert len(missing) == 0, f"missing column(s): {missing}"
    points = points[["x", "y", "z"]]
    best_normal = None
    best_offset = None
    best_inlier = None
    best_number_of_inliers = 0
    for i in range(iterations):
        samples = points.sample(n=3, random_state=i)  # using random_state for reproducible sequence and thus a reproducible algorithm behavior
        v1 = samples.iloc[1] - samples.iloc[0]
        v2 = samples.iloc[2] - samples.iloc[0]
        offset = samples.iloc[0]
        normal = np.cross(v1, v2)
        normal /= np.linalg.norm(normal, 2)
        distances = np.abs(np.dot(points - offset, normal.reshape((3, 1))))
        inlier = distances < expected_noise
        if len(distances[inlier]) > best_number_of_inliers:
            best_inlier = inlier
            best_number_of_inliers = len(distances[inlier])
            best_normal = normal
            best_offset = np.mean(points[inlier], axis=0)
    if linear_refinement:
        plane = fit_plane(name, points[best_inlier])
    else:
        plane = Plane.from_normal_offset(name, best_normal, best_offset.to_numpy())
    return plane, best_inlier


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


def main(dataset_name: str = "C4L5"):
    directory_to_datasets = "/" + os.path.join("home", "workspace", "datasets")
    factory = VehicleFactory(directory_to_datasets)
    vehicle, solution = factory.create(
        dataset_name,
        deviation=deviation,
    )
    optimizer = ExtendedSparseOptimizer(
        vehicle=vehicle,
        algorithm=g2o.OptimizationAlgorithmLevenberg(g2o.BlockSolverSE3(g2o.LinearSolverDenseSE3())),
    )
    optimizer.add_photogrammetry(photogrammetry=read_obc("/" + os.path.join(directory_to_datasets, dataset_name, "photogrammetry.obc")))
    optimizer.add_sensors()
    optimizer.initialize_optimization()
    optimizer.set_verbose(True)
    optimizer.optimize(100000)

    print(optimizer.vehicle.as_dataframe(only_leafs=True, relative_coordinates=False))

    return vehicle, {}


def lidar_calibration():
    directory_to_datasets = "/" + os.path.join("home", "workspace", "datasets")
    factory = VehicleFactory(directory_to_datasets)
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
    photogrammetry = read_obc("/" + os.path.join("home", "workspace", "datasets", "C4L5", "photogrammetry.obc"))
    photogrammetry_planes: List[Plane] = []
    counter = 0
    cache = []
    for _, row in photogrammetry.iterrows():
        counter += 1
        cache.append(row)
        if counter % 6 == 0:
            photogrammetry_planes.append(fit_plane("plane", pd.concat(cache, axis=1).T))
            counter = 0
            cache = []
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
            edge.set_information(np.linalg.inv(np.eye(3) / 0.0001))
            # edge.set_robust_kernel(g2o.RobustKernelHuber(0.01))
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
    vehicle, report = main()
