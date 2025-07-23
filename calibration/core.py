from datatypes import *
from utils import *
import g2opy.g2opy as g2o
import json


class ExtendedSparseOptimizer(g2o.SparseOptimizer):
    """TODO"""

    def __init__(self, vehicle: Vehicle, algorithm: g2o.OptimizationAlgorithm):
        super().__init__()
        self.set_algorithm(algorithm)
        self.vehicle = vehicle
        self.__convention_transform = SE3(rotation=Rotation.from_euler("YXZ", [90, 0, -90], degrees=True))

    def add_frame(self, frame: Frame):
        pass

    def add_characteristic_points(self):
        pass

    def add_edge_minimization(self):
        pass

    def add_gicp(self):
        pass

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


if __name__ == "__main__":
    vehicle, report = main()
