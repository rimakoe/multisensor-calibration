import os
from glob import glob
from setuptools import find_packages, setup

package_name = "simulation"

setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        # ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "config"), glob(os.path.join("config", "*.rviz"))),
        (os.path.join("share", package_name, "launch"), glob(os.path.join("launch", "*.launch.py"))),
        (os.path.join("share", package_name, "worlds"), glob(os.path.join("worlds", "*.xacro"))),
        (os.path.join("share", package_name, "worlds"), glob(os.path.join("worlds", "*.sdf"))),
        (os.path.join("share", package_name, "robots"), glob(os.path.join("robots", "*.xacro"))),
        (os.path.join("share", package_name, "sensors"), glob(os.path.join("sensors", "*.xacro"))),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="root",
    maintainer_email="root@todo.todo",
    description="TODO: Package description",
    license="TODO: License declaration",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [],
    },
)
