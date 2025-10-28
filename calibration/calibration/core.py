import json, argparse
from typing import Tuple
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from scipy.stats import chi2, norm
import tqdm
import seaborn as sns
import g2opy.g2opy as g2o
from calibration.datatypes import *
from calibration.utils import *
from calibration.plots import *

np.set_printoptions(edgeitems=30, linewidth=1000000)

DEBUG = False

# percentiles for the chi2 distribution for 3 and 2 dof

CHI2_SIGMA2_DOF2 = np.sqrt(9.21)
CHI2_SIGMA3_DOF2 = np.sqrt(13.815)

CHI2_SIGMA2_DOF3 = np.sqrt(11.345)
CHI2_SIGMA3_DOF3 = np.sqrt(16.266)

rng = np.random.default_rng(0)  # one reproducible stream


class ExtendedSparseOptimizer(g2o.SparseOptimizer):
    """Extension of the g2o.SparseOptimizer class that intends to be a help on creating generic parts of the problem such as adding the photogrammetry data of the room or the reprojection error minimization given just the camera as a data container."""

    def __init__(
        self,
        vehicle: Vehicle,
        algorithm: g2o.OptimizationAlgorithm = None,
        photogrammetry: pd.DataFrame = None,
        plot: bool = False,
        silent: bool = True,
    ):
        super().__init__()
        if algorithm is None:
            algorithm = g2o.OptimizationAlgorithmLevenberg(g2o.BlockSolverSE3(g2o.LinearSolverDenseSE3()))
        self.set_algorithm(algorithm)
        self.vehicle = vehicle
        self.photogrammetry = photogrammetry
        self.plot = plot
        self.silent = silent
        self._added_photogrammetry = False
        self._convention_transform = Transform(rotation=Rotation.from_euler("YXZ", [90, 0, -90], degrees=True))

    def set_photogrammetry(self, photogrammetry: pd.DataFrame):
        self.photogrammetry = photogrammetry

    def add_photogrammetry(self, photogrammetry: pd.DataFrame):
        if self._added_photogrammetry:
            return  # already added
        for _, row in photogrammetry.iterrows():
            marker_id: int = int(row["id"])
            marker_point: np.ndarray = row[["x", "y", "z"]].to_numpy()
            vertex_photogrammetry = g2o.VertexPointXYZ()
            vertex_photogrammetry.set_id(marker_id)
            vertex_photogrammetry.set_estimate(marker_point)
            vertex_photogrammetry.set_fixed(True)
            self.add_vertex(vertex_photogrammetry)
        self._added_photogrammetry = True

    def add_sensors(self, frame: Frame = None):
        if frame is None:
            frame = self.vehicle
        if type(frame) is Camera:
            if not self._added_photogrammetry:
                if self.photogrammetry is None:
                    raise ValueError("You need to initialize the photogrammetry first or add it to this object.")
                self.add_photogrammetry(self.photogrammetry)
                self._added_photogrammetry = True
            self.add_reprojection_error_minimization(frame)
            return
        if type(frame) is Lidar:
            self.add_point_to_plane_error_minimization(frame)
            return
        for child in frame.children:
            self.add_sensors(child)
        return

    def add_reprojection_error_minimization(self, camera: Camera) -> None:
        """
        Abstract function to add all the needed vertices and edges for a reprojection error minimization using a camera object.

        Parameters
        ----------
            camera : Camera
                Camera object that contains the camera informations, the detected features and the initial guess usually as the parent transform.
        Returns
        -------
            None
        """
        camera_parameter = g2o.CameraParameters(np.mean(camera.intrinsics.focal_length), camera.intrinsics.principal_point, 0)
        camera_parameter.set_id(camera.id)
        self.add_parameter(camera_parameter)

        vertex = g2o.VertexSE3Expmap()
        vertex.set_id(camera.id)
        vertex.set_fixed(False)
        vertex.set_estimate((camera.parent.transform @ self._convention_transform).inverse().as_eigen())
        self.add_vertex(vertex)
        camera.vertex = vertex

        for _, row in camera.features.iterrows():
            marker_id: int = int(row["id"])
            sigma = row[["su", "sv"]]
            edge = g2o.EdgeProjectXYZ2UV()
            edge.set_vertex(0, self.vertex(marker_id))
            edge.set_vertex(1, self.vertex(camera.id))
            edge.set_measurement(row[["u", "v"]].to_numpy())
            edge.set_information(np.diag(1.0 / sigma**2))  # no effect right now
            edge.set_parameter_id(0, camera.id)
            edge.set_id(camera.id * 1000 + marker_id * 10)
            edge.set_robust_kernel(g2o.RobustKernelHuber(CHI2_SIGMA3_DOF2))
            self.add_edge(edge)

    def add_point_to_plane_error_minimization(self, lidar: Lidar) -> None:
        feature_noise = 0.01  # this should be integrated into the actual feature vector of the lidar object
        photogrammetry_planes: List[Plane] = []
        counter = 0
        cache = []
        for _, row in self.photogrammetry.iterrows():
            counter += 1
            cache.append(row)
            if counter % 9 == 0:
                photogrammetry_planes.append(fit_plane("plane", pd.concat(cache, axis=1).T))
                counter = 0
                cache = []
        photogrammetry_centroids = [plane.get_offset() for plane in photogrammetry_planes]

        # Has to be set in order to work
        offset = g2o.ParameterSE3Offset()
        offset.set_id(0)
        self.add_parameter(offset)

        # First transform which is fixed but still contributes to the optimization DOF
        vehicle_vertex = g2o.VertexSE3()
        vehicle_vertex.set_id(lidar.id + 1000)
        vehicle_vertex.set_estimate(g2o.Isometry3d())
        vehicle_vertex.set_fixed(True)
        self.add_vertex(vehicle_vertex)

        # Second transform that is subject of optimization
        lidar_vertex = g2o.VertexSE3()
        lidar_vertex.set_id(lidar.id)
        lidar_vertex.set_estimate(g2o.Isometry3d(lidar.parent.transform.rotation.as_matrix(), lidar.parent.transform.translation))
        lidar_vertex.set_fixed(False)
        self.add_vertex(lidar_vertex)

        # Match the features
        neighbor = NearestNeighbors(n_neighbors=2, radius=0.4, n_jobs=4)
        neighbor.fit(photogrammetry_centroids)

        if self.plot:
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
                # Get corresponding planes and transform them into sensor coordinates using the initial guess to optimize relative to the initial guess
                pplane: Plane = photogrammetry_planes[photogrammetry_idx]
                splane: Plane = lidar.features[sensor_idx]

                # transform into sensor coordinate system
                splane.transform = lidar.parent.transform.inverse() @ splane.transform
                pplane.transform = lidar.parent.transform.inverse() @ pplane.transform

                # unify directions
                normal0 = pplane.get_normal()
                if np.linalg.norm(pplane.get_offset() + normal0 * 0.1, 2) < np.linalg.norm(pplane.get_offset(), 2):
                    normal0 *= -1.0
                normal1 = splane.get_normal()
                if np.linalg.norm(splane.get_offset() + normal1 * 0.1, 2) < np.linalg.norm(splane.get_offset(), 2):
                    normal1 *= -1.0

                # create measurement
                measurement = g2o.EdgeGICP()
                measurement.normal0 = normal0
                measurement.pos0 = np.dot(pplane.get_offset(), normal0) * normal0  # as hessian normal form
                measurement.normal1 = normal1
                measurement.pos1 = np.dot(splane.get_offset(), normal1) * normal1  # as hessian normal form

                # create edge
                edge = g2o.EdgeVVGicp()
                edge.set_id(lidar.id * 1000 + photogrammetry_idx)
                edge.set_vertex(0, vehicle_vertex)
                edge.set_vertex(1, lidar_vertex)
                edge.set_measurement(measurement)
                edge.set_information(np.eye(3) / (feature_noise**2))
                edge.set_robust_kernel(g2o.RobustKernelHuber(CHI2_SIGMA3_DOF3))
                self.add_edge(edge)

                break  # Currently only the most likely match from DBSCAN is used. This must be done more clever in real applications.

    def _integrate_result(self, frame: Frame, id, result: Tuple[Transform, np.ndarray]):
        if type(frame) is Camera and frame.id == id:
            transform, covariance = result
            frame.transform = frame.parent.transform.inverse() @ transform.inverse() @ self._convention_transform.inverse()
            frame.covariance = self._convention_transform.inverse().adjoint().T @ covariance @ self._convention_transform.inverse().adjoint()
            # frame.covariance = covariance # Only if we do not want the covariance to be rotated to fit to the convention and thus given in camera coords
            if self.plot:
                pearson = compute_pearson(covariance=covariance)
                precision = np.sqrt(np.diag(covariance))
                srot = Rotation.from_rotvec(precision[:3]).as_euler("xyz", degrees=True)
                print(
                    pd.DataFrame(
                        data={
                            "st_x": precision[3],
                            "st_y": precision[4],
                            "st_z": precision[5],
                            "seuler_x": srot[0],
                            "seuler_y": srot[1],
                            "seuler_z": srot[2],
                        },
                        index=[0],
                    )
                )
                plot_heatmap_compact(title=f"{frame.name} pearson matrix", map=pearson)
                xs = []
                ys = []
                camera_ids = []
                marker_ids = []
                for edge in self.edges():
                    if type(edge) is g2o.EdgeProjectXYZ2UV:
                        error = edge.error()
                        information = edge.information()
                        normed_error = error @ np.sqrt(information)
                        xs.append(normed_error[0])
                        ys.append(normed_error[1])
                        marker_ids.append(edge.vertex(0).id())
                        camera_ids.append(edge.vertex(1).id())
                df = pd.DataFrame(
                    {
                        "camera id": camera_ids,
                        "marker id": marker_ids,
                        "u": xs,
                        "v": ys,
                    }
                )
                plot_residual(df, frame)
            return
        if type(frame) is Lidar and frame.id == id:
            transform, covariance = result
            frame.covariance = covariance
            frame.transform = transform
            if self.plot:
                pearson = compute_pearson(covariance)
                precision = np.sqrt(np.diag(covariance))
                srot = Rotation.from_rotvec(precision[:3]).as_euler("xyz", degrees=True)
                print(
                    pd.DataFrame(
                        data={
                            "st_x": precision[3],
                            "st_y": precision[4],
                            "st_z": precision[5],
                            "seuler_x": srot[0],
                            "seuler_y": srot[1],
                            "seuler_z": srot[2],
                        },
                        index=[0],
                    )
                )
                # print(
                #    pd.DataFrame(
                #        data={
                #            "euler_x": frame.transform.rotation.as_euler("xyz", degrees=True)[0],
                #            "euler_y": frame.transform.rotation.as_euler("xyz", degrees=True)[1],
                #            "euler_z": frame.transform.rotation.as_euler("xyz", degrees=True)[2],
                #            "t_x": frame.transform.translation[0],
                #            "t_y": frame.transform.translation[1],
                #            "t_z": frame.transform.translation[2],
                #        },
                #        index=[0],
                #    )
                # )
                plot_heatmap_compact(title=f"{frame.name} pearson matrix", map=pearson)
            return
        for child in frame.children:
            self._integrate_result(child, id, result)

    def cut(self, percent: float = 0.1):
        chi2s = []
        edges = []
        for edge in self.edges():
            chi2s.append(edge.chi2())
            edges.append(edge)
        df = pd.DataFrame(data={"chi2": chi2s, "edge": edges})
        df.sort_values("chi2", inplace=True)
        for edge in df.iloc[int((1 - percent) * len(edges)) :]["edge"]:
            self.remove_edge(edge)

    def optimize(self, iterations=1000):
        super().optimize(iterations)
        for id, vertex in self.vertices().items():
            if type(vertex) is not g2o.VertexSE3Expmap and type(vertex) is not g2o.VertexSE3:
                continue
            if id >= 1000 and id < 2000:
                if not vertex.fixed():
                    covariance = np.linalg.inv(vertex.hessian())
                else:
                    covariance = np.zeros((6, 6))
                result = (
                    Transform(
                        translation=vertex.estimate().translation(),
                        rotation=Rotation.from_matrix(vertex.estimate().rotation().matrix()),
                    ),
                    covariance,
                )
                self._integrate_result(self.vehicle, id, result)

    def get_dof(self) -> int:
        dof: int = 0
        for edge in self.edges():
            if type(edge) is g2o.EdgeSE3PointXYZ:
                dof += 1
                continue
            if type(edge) is g2o.EdgeProjectXYZ2UV:
                dof += 2
                continue
            if type(edge) is g2o.EdgeVVGicp:
                dof += 3
                continue
        for id, vertex in self.vertices().items():
            if type(vertex) is g2o.VertexSE3Expmap or type(vertex) is g2o.VertexSE3:  # subtract the fixed one from ICP on purpose here as well
                dof -= 6
                continue
        return dof


class VehicleFactory:
    def __init__(self, directory_to_datasets=get_dataset_directory()):
        self.directory_to_datasets = directory_to_datasets

    def create(
        self,
        dataset_name: str,
        sensor_whitelist: List = None,
        deviation: Transform = Transform(),
        use_ideal_features: bool = True,
        camera_feature_noise: float = None,  # homoscedastic feature noise here for simulation
        lidar_feature_noise: float = None,  # homoscedastic feature noise here for simulation
    ) -> Tuple[Vehicle, Solution]:
        solution_filepath = os.path.join(self.directory_to_datasets, dataset_name, "solution.json")
        solution = Solution(**json.load(open(solution_filepath)))
        vehicle = Vehicle(name="Vehicle", system_configuration=dataset_name)
        sensor_id = 1000
        for name, data in solution.devices.items():
            if sensor_whitelist:
                if name.lower() not in list(map(str.lower, sensor_whitelist)):
                    continue
            initial_guess = Frame(name="InitialGuess", transform=Transform.from_dict(data.extrinsics) @ deviation)
            sensor_folder = os.path.join(self.directory_to_datasets, dataset_name, name.lower())
            if "lidar" in name.lower():
                data = PointCloud.from_path(os.path.join(sensor_folder, name.lower() + ".pcd"))  # read the pcd
                # crop the pointcloud roughly
                pointcloud = pd.DataFrame(data.pc_data)
                pointcloud = pointcloud[np.isfinite(pointcloud).all(1)]
                pointcloud.dropna()
                pointcloud = initial_guess.transform.apply(pointcloud)  # transform data using the initial guess
                pointcloud = pointcloud[pointcloud["z"] > 0.5]
                pointcloud = pointcloud[np.linalg.norm(pointcloud[["x", "y", "z"]], 2, axis=1) < 12.0]
                pointcloud = pointcloud[np.linalg.norm(pointcloud[["x", "y", "z"]], 2, axis=1) > 2.0]
                db = DBSCAN(eps=0.4, min_samples=20).fit(pointcloud)  # cluster the modules
                pointcloud["label"] = db.labels_  # add the labels to our data table

                if DEBUG:
                    fig = plt.figure()
                    ax = fig.add_subplot(111, projection="3d")
                    # draw scatter
                    for label in np.unique(pointcloud["label"]):
                        ax.scatter(
                            xs=pointcloud[pointcloud["label"] == label]["x"],
                            ys=pointcloud[pointcloud["label"] == label]["y"],
                            zs=pointcloud[pointcloud["label"] == label]["z"],
                        )
                    ax.set_aspect("equal")
                    plt.show(block=True)

                features: List[Plane] = []
                if use_ideal_features:
                    photogrammetry = read_obc(os.path.join(self.directory_to_datasets, dataset_name, "photogrammetry.obc"))
                    counter = 0
                    cache = []
                    for _, row in photogrammetry.iterrows():
                        counter += 1
                        cache.append(row)
                        if counter % 9 == 0:
                            plane = fit_plane("plane", pd.concat(cache, axis=1).T)
                            plane.transform.translation += plane.get_normal() * rng.normal(0, lidar_feature_noise)
                            plane.transform = deviation @ plane.transform
                            features.append(plane)
                            counter = 0
                            cache = []
                else:
                    features = detect_planes(pointcloud)  # find the planes
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
                features = pd.DataFrame.from_dict(json.load(open(os.path.join(sensor_folder, "detections.json"))))  # get ideal features
                features[["su", "sv"]] = np.ones(features[["u", "v"]].shape) * camera_feature_noise
                if camera_feature_noise:
                    features[["u", "v"]] += rng.normal(0.0, camera_feature_noise, size=features[["u", "v"]].shape)
                # features["u"] += 0.5  # gazebo has an offset here somehow
                # features["v"] += 0.5  # gazebo has an offset here somehow
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


def detect_planes(pointcloud: pd.DataFrame):
    planes: List[Plane] = []
    # The pointclouds are labeled from the clustering algorithm. We use the single clusters here and assume three planes to be present in each cluster.
    for label in tqdm.tqdm(np.unique(pointcloud["label"])):
        module_pointcloud = pointcloud[pointcloud["label"] == label]
        if len(module_pointcloud["x"]) < 100:
            continue
        plane, inlier = fit_plane_ransac("plane", module_pointcloud)
        planes.append(plane)
        module_pointcloud = module_pointcloud[~inlier]
        if len(module_pointcloud["x"]) < 100:
            continue
        plane, inlier = fit_plane_ransac("plane", module_pointcloud)
        planes.append(plane)
        module_pointcloud = module_pointcloud[~inlier]
        if len(module_pointcloud["x"]) < 100:
            continue
        plane, inlier = fit_plane_ransac("plane", module_pointcloud)
        planes.append(plane)
    return planes


def fit_plane_ransac(name: str, points: pd.DataFrame, iterations: int = 200, expected_noise=0.02, linear_refinement=True) -> Plane:
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


def evaluate(optimizer: ExtendedSparseOptimizer, silent: bool = True, plot: bool = False):
    if len(optimizer.edges()) == 0:
        raise ValueError("No Edges!")
    if not silent:
        print(optimizer.vehicle.as_dataframe(only_leafs=True, relative_coordinates=False))

    dof = optimizer.get_dof()
    # assert dof >= 0, "There is not DOF for a unique result"

    errors = []
    for edge in optimizer.edges():
        errors.append(edge.error() * np.sqrt(np.diag(edge.information())))
    errors = np.array(errors).flatten()
    z = 3.0
    C = 1.4826 * (1 + 5 / dof)
    sigma_estimate = np.sqrt(np.median(np.square(errors)))
    threshold = z / np.sqrt(dof)
    robust_estimate_corrected = C * sigma_estimate
    robust_global_test_passed = True
    robust_global_test_message = "OK"
    if robust_estimate_corrected > 1 + threshold:
        robust_global_test_passed = False
        robust_global_test_message = "FAILED - TOO HIGH"
    if robust_estimate_corrected < 1 - threshold:
        robust_global_test_passed = False
        robust_global_test_message = "FAILED - TOO LOW"

    active_chi2 = optimizer.active_chi2()
    robust_chi2 = optimizer.active_robust_chi2()
    p_value_center = chi2.sf(dof, df=dof)
    p_value = chi2.sf(active_chi2, df=dof)  # is the chi2_score
    chi2_global_test_passed = True
    alpha = 0.01
    chi2_global_test_comment = "OK"
    if p_value < alpha:
        chi2_global_test_passed = False
        chi2_global_test_comment = "FAILED - TOO LOW"
    if p_value > 1 - alpha:
        chi2_global_test_passed = False
        chi2_global_test_comment = "FAILED - TOO HIGH"
    report = {
        "robust_chi2": robust_chi2,
        "active_chi2": active_chi2,
        "kernel_quotient": 1.0 - robust_chi2 / active_chi2,
        "number_of_edges": len(optimizer.edges()),
        "dof": dof,
        "reduced_chi2": active_chi2 / dof,  # you want this to be close to 1
        "reduced_sigma": np.sqrt(2 / dof),
        "alpha": alpha,
        "chi2_score_center": p_value_center,
        "chi2_score": p_value,
        "chi2_global_test_passed": chi2_global_test_passed,
        "chi2_global_test_message": chi2_global_test_comment,
        "robust_score": robust_estimate_corrected,
        "robust_global_test_passed": robust_global_test_passed,
        "robust_global_test_message": robust_global_test_message,
    }
    if not silent:
        print("")
        print("========================== REPORT ==========================")
        print("")
        [print("{0:25} {1}".format(key, value)) for key, value in report.items()]
        print("")
        print("============================================================")
        print("")

    return report


def main(vehicle: Vehicle, dataset_name: str, post_processing=False, silent=False, plot=False) -> Tuple[Vehicle, dict]:
    optimizer = ExtendedSparseOptimizer(
        vehicle=vehicle,
        algorithm=g2o.OptimizationAlgorithmLevenberg(g2o.BlockSolverSE3(g2o.LinearSolverDenseSE3())),
        photogrammetry=read_obc(os.path.join(get_dataset_directory(), dataset_name, "photogrammetry.obc")),
        plot=plot,
        silent=silent,
    )
    optimizer.add_sensors()
    optimizer.set_verbose(False)
    for i in range(2):
        optimizer.initialize_optimization()
        optimizer.optimize(1000)
        if (  # WARNING: post processing shifts the estimated noise of the observed data if actual points from the distribution are deleted instead of outliers.
            not post_processing
        ):
            break
        report = evaluate(optimizer=optimizer, silent=True, plot=False)
        if report["global_test_passed"]:
            break
        optimizer.cut(0.05)  # cut the badest 5 % of the edges
        for edge in optimizer.edges():
            threshold: float = None
            if type(edge) is g2o.EdgeVVGicp:
                threshold = CHI2_SIGMA3_DOF3
            if type(edge) is g2o.EdgeProjectXYZ2UV:
                threshold = CHI2_SIGMA3_DOF2
            if threshold is None:
                continue
            if edge.chi2() < threshold**2:  # inside 3 sigma for this plane on a chi2 distribution for N=3
                continue
            optimizer.remove_edge(edge)
        if optimizer.get_dof() < 0:  # unable to optimize
            return None, None
    report = evaluate(optimizer=optimizer, silent=silent, plot=plot)
    return vehicle, report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="calibration", description="automated calibration toolchain for camera and lidar")
    parser.add_argument("whitelist", nargs="*", default=None)  # on default the whole dataset is calibrated
    parser.add_argument("-d", "--dataset", required=True)
    parser.add_argument("--dataset-directory", default=get_dataset_directory())
    args = parser.parse_args()
    factory = VehicleFactory(args.dataset_directory)
    vehicle, solution = factory.create(
        args.dataset,
        sensor_whitelist=args.whitelist,
        deviation=Transform(
            rotation=Rotation.from_euler("xyz", np.array([0, 0, 0]) + rng.normal(0, 1, size=3), degrees=True),
            translation=np.array([0.0, 0.0, 0.0]) + rng.normal(0, 0.1, size=3),
        ),
        camera_feature_noise=1.0,
        lidar_feature_noise=0.01,
        use_ideal_features=True,
    )
    vehicle, report = main(vehicle=vehicle, dataset_name=args.dataset, silent=False, plot=True)
