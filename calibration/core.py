from datatypes import *
from utils import *
import g2opy.g2opy as g2o
import json

# here the actual calibration happens

# TODO 1. feature detection (gather the marker xy coordinates in the image and the plane normals in 3D)

# 2.1 reprojection error minimization -> Done

# TODO 2.2 plane normal fitting

# TODO 3. combined feature matching (extract edges somehow -> fit edges from image and edges in pointcloud together)

# TODO evaluate the result on accuracy, precision and reliablitiy (e.g. noise variations)


class OptimizatinProblem:
    camera_id = 0

    def __init__(self, vehicle: Vehicle, solution: Solution, photogrammetry: pd.DataFrame):
        self.vehicle = vehicle
        self.solution = solution
        self.photogrammetry = photogrammetry
        self.optimizer = g2o.SparseOptimizer()
        linear_solver = g2o.LinearSolverDenseSE3()
        block_solver = g2o.BlockSolverSE3(linear_solver)
        algorithm = g2o.OptimizationAlgorithmLevenberg(block_solver)
        self.optimizer.set_algorithm(algorithm)
        self.r_to_camera_coordinate = SE3(rotation=Rotation.from_euler("YXZ", [90, 0, -90], degrees=True))

    def init_photogrammetry(self):
        for index, row in self.photogrammetry.iterrows():
            marker_id: int = int(row["id"])
            vertex_photogrammetry = g2o.VertexPointXYZ()
            vertex_photogrammetry.set_id(marker_id)
            vertex_photogrammetry.set_estimate(row[["x", "y", "z"]].to_numpy())
            vertex_photogrammetry.set_fixed(True)
            self.optimizer.add_vertex(vertex_photogrammetry)

    def init_cameras(
        self,
        frame: Frame,
        path: List[Frame] = None,
    ) -> Tuple[g2o.VertexSE3Expmap, g2o.CameraParameters]:
        if not path:
            path = []
        output = []
        current_path = path + [frame]
        if type(frame) is Camera:
            camera_parameter = g2o.CameraParameters(np.mean(frame.intrinsics.focal_length), frame.intrinsics.principal_point, 0)
            camera_parameter.set_id(self.camera_id)
            self.optimizer.add_parameter(camera_parameter)
            frame.id = self.camera_id
            vertex = g2o.VertexSE3Expmap()
            vertex.set_id(1000 + self.camera_id)
            vertex.set_fixed(False)
            initial_transform = SE3()
            for f in path:  # important to use the path here, since current path includes the camera transform itself
                initial_transform = initial_transform @ f.transform
            vertex.set_estimate(
                (initial_transform @ self.r_to_camera_coordinate).inverse().as_eigen()
            )  # might has to be the inverse here for some reason. Investigate!
            self.optimizer.add_vertex(vertex)
            frame.vertex = vertex
            projected_points = np.array([-1, 320, 240])

            output.append((vertex, camera_parameter))
            for index, row in frame.features.iterrows():
                marker_id: int = int(row["id"])
                sigma = 1.0
                edge = g2o.EdgeProjectXYZ2UV()
                edge.set_vertex(0, self.optimizer.vertex(marker_id))
                edge.set_vertex(1, self.optimizer.vertex(1000 + self.camera_id))
                edge.set_measurement(row[["x", "y"]].to_numpy())
                edge.set_information(np.identity(2) / (sigma**2))  # no effect right now
                edge.set_parameter_id(0, self.camera_id)
                edge.set_id(self.camera_id * 10 + marker_id)
                self.optimizer.add_edge(edge)

                point = self.photogrammetry.loc[self.photogrammetry["id"] == marker_id][["x", "y", "z"]].to_numpy().flatten()
                point = (initial_transform @ self.r_to_camera_coordinate).inverse().apply(point)
                uv = camera_parameter.cam_map(point)  # Project to image plane
                projected_points = np.vstack([np.hstack([marker_id, uv]), projected_points])

            self.camera_id += 1
            debug = False
            if debug:
                img = np.asarray(frame.data)
                fig, ax = plt.subplots(1, 1)
                ax.imshow(img, origin="upper", extent=[0, img.shape[1], img.shape[0], 0])  # origin and extent specified to be pixel perfect here
                ax.scatter(projected_points[:, 1], projected_points[:, 2], c="red", marker="+", s=100, linewidths=2, label="Projected Points")
                for i, u, v in projected_points:
                    ax.text(u + 5, v - 5, f"P{i}", color="white", fontsize=10, bbox=dict(facecolor="black", alpha=0.6, pad=2))
                ax.set_xlim([0, img.shape[1]])
                ax.set_ylim([img.shape[0], 0])
                ax.set_title("3D Points Projected to Image Plane")
                ax.axis("off")
                plt.legend()
                plt.show(block=True)

        for child in frame.children:
            output.extend(self.init_cameras(child, current_path))
        return output

    def create(self) -> dict:
        self.init_photogrammetry()
        self.init_cameras(self.vehicle)
        return {}

    def integrate_result(self, frame: Frame, id, result: SE3):
        if type(frame) is Camera and frame.id + 1000 == id:
            frame.transform = frame.parent.transform.inverse() @ result @ self.r_to_camera_coordinate.inverse()
            return
        for child in frame.children:
            self.integrate_result(child, id, result)

    def solve(self):
        self.optimizer.initialize_optimization()
        self.optimizer.set_verbose(True)
        self.optimizer.optimize(10000)

        for id, vertex in self.optimizer.vertices().items():
            if id >= 1000 and id < 2000:
                result = SE3(
                    translation=vertex.estimate().inverse().translation(),
                    rotation=Rotation.from_matrix(vertex.estimate().inverse().rotation().matrix()),
                )
                self.integrate_result(self.vehicle, id, result)
        return {}

    def view_graph(self):
        for id, vertex in self.optimizer.vertices().items():
            print(f"{id}: {vertex}")
        for edge in self.optimizer.edges():
            print(f"{edge}")


def c4l5():
    directory_to_datasets = "/" + os.path.join("home", "workspace", "datasets")
    dataset = "C4L5"
    solution_filepath = os.path.join(directory_to_datasets, dataset, "solution.json")
    solution = Solution(**json.load(open(solution_filepath)))
    vehicle = Vehicle(name="Vehicle", system_configuration="CL-0.5")

    deviation = SE3(rotation=Rotation.from_euler("xyz", [3, 2, 1], degrees=True), translation=np.array([0.5, 0.5, 0.5]))
    # deviation = SE3()

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
            features = pd.DataFrame.from_dict(
                json.load(open(os.path.join(sensor_folder, "detections.json"))),
            )
            features["x"] += 0.5
            features["y"] += 0.5
            initial_guess.add_child(
                Camera(
                    name=name,
                    data=Image.open(os.path.join(sensor_folder, name.lower() + ".bmp")),
                    intrinsics=Camera.Intrinsics.from_json(
                        os.path.join(sensor_folder, "camera_info.json"),
                    ),
                    features=features,
                )
            )
            vehicle.add_child(initial_guess)

    op = OptimizatinProblem(
        vehicle=vehicle,
        solution=solution,
        photogrammetry=read_obc("/" + os.path.join("home", "workspace", "datasets", "C4L5", "photogrammetry.obc")),
    )

    op.create()
    op.solve()
    print(vehicle.as_dataframe(only_leafs=True, relative_coordinates=True))

    return vehicle, {}


if __name__ == "__main__":
    vehicle, report = c4l5()
