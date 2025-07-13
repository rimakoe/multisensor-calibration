FROM ros:rolling-ros-core-noble

# COPY --from=g2o:latest /usr/local/lib /usr/local/lib
# COPY --from=g2o:latest /usr/local/include /usr/local/include
# COPY --from=g2o:latest /usr/bin /usr/bin
# COPY --from=g2o:latest /usr/lib /usr/lib
# COPY --from=g2o:latest /usr/local/lib/python3.10/dist-packages /usr/local/lib/python3.10/dist-packages

#ENV LD_LIBRARY_PATH=/usr/local/lib

################
# ROS2

# RUN apt-get update && apt-get install -y wget lsb-release

# install ros2 packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    ros-rolling-ros-core \
    ros-rolling-rosbag2-storage-mcap \
    ros-rolling-twist-mux \
    ros-rolling-sensor-msgs-py \
    ros-rolling-cv-bridge \
    ros-rolling-vision-opencv \
    ros-rolling-ros-gz \
    python3-dev \
    python3-colcon-common-extensions \
    pip \
    git \
    cmake \
    build-essential \
    ros-rolling-ros2bag \
    python3-venv \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && python3 -m venv /opt/venv
    # && echo "source /opt/venv/bin/activate" >> ~/.bashrc

ENV PATH=/opt/venv/bin:$PATH

RUN pip install colcon-common-extensions pypcd4 opencv-python==3.4.18.65 numpy==1.26.4

#RUN  wget https://packages.osrfoundation.org/gazebo.gpg -O /usr/share/keyrings/pkgs-osrf-archive-keyring.gpg \
#    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/pkgs-osrf-archive-keyring.gpg] http://packages.osrfoundation.org/gazebo/ubuntu-stable $(lsb_release -cs) main" | tee /etc/apt/sources.list.d/gazebo-stable.list > /dev/null \
#    && apt-get update && apt-get install --no-install-recommends -y \    
#    ros-humble-ros-base \
#    gz-harmonic \
#    libgz-plugin2 \
#    libgz-transport13 \
#    libgz-sensors8 \
#    && rm -rf /var/lib/apt/lists/*

################

# RUN pip install setuptools==68

RUN echo "export GZ_VERSION=rolling" >> ~/.bashrc \
    && echo "source /opt/ros/rolling/setup.bash" >> ~/.bashrc \
    && echo "source /home/workspace/ros2_ws/install/local_setup.bash" >> ~/.bashrc \
    && echo "export GZ_SIM_RESOURCE_PATH=/home/workspace/ros2_ws/install/simulation/share/simulation/models" >> ~/.bashrc