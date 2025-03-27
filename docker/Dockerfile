FROM public.ecr.aws/lts/ubuntu:jammy

COPY --from=g2o:latest /usr/local/lib /usr/local/lib
COPY --from=g2o:latest /usr/local/include /usr/local/include
COPY --from=g2o:latest /usr/bin /usr/bin
COPY --from=g2o:latest /usr/lib /usr/lib
COPY --from=g2o:latest /usr/local/lib/python3.10/dist-packages /usr/local/lib/python3.10/dist-packages

ENV LD_LIBRARY_PATH=/usr/local/lib

################
# ROS2

RUN apt-get update && apt-get install -y wget lsb-release

# setup timezone
RUN echo 'Etc/UTC' > /etc/timezone && \
    ln -s /usr/share/zoneinfo/Etc/UTC /etc/localtime && \
    apt-get update && \
    apt-get install -q -y --no-install-recommends tzdata && \
    rm -rf /var/lib/apt/lists/*

# install packages
RUN apt-get update && apt-get install -q -y --no-install-recommends \
    dirmngr \
    gnupg2 \
    && rm -rf /var/lib/apt/lists/*

# setup keys
RUN set -eux; \
    key='4B63CF8FDE49746E98FA01DDAD19BAB3CBF125EA'; \
    export GNUPGHOME="$(mktemp -d)"; \
    gpg --batch --keyserver keyserver.ubuntu.com --recv-keys "$key"; \
    mkdir -p /usr/share/keyrings; \
    gpg --batch --export "$key" > /usr/share/keyrings/ros2-snapshots-archive-keyring.gpg; \
    gpgconf --kill all; \
    rm -rf "$GNUPGHOME"

# setup sources.list
RUN echo "deb [ signed-by=/usr/share/keyrings/ros2-snapshots-archive-keyring.gpg ] http://snapshots.ros.org/iron/final/ubuntu jammy main" > /etc/apt/sources.list.d/ros2-snapshots.list

# setup environment
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

ENV ROS_DISTRO=iron

# install ros2 packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    ros-iron-ros-core \
    ros-iron-rosbag2-storage-mcap \
    ros-iron-twist-mux \
    ros-iron-sensor-msgs-py \
    ros-iron-cv-bridge \
    ros-iron-vision-opencv \
    ros-iron-gz-ros2-control \
    ros-iron-ros-gz \
    ros-iron-gazebo-ros-pkgs \ 
    python3-colcon-common-extensions \
    && rm -rf /var/lib/apt/lists/*

RUN  wget https://packages.osrfoundation.org/gazebo.gpg -O /usr/share/keyrings/pkgs-osrf-archive-keyring.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/pkgs-osrf-archive-keyring.gpg] http://packages.osrfoundation.org/gazebo/ubuntu-stable $(lsb_release -cs) main" | tee /etc/apt/sources.list.d/gazebo-stable.list > /dev/null \
    && apt-get update && apt-get install --no-install-recommends -y \    
    ros-iron-ros-base \
    gz-harmonic \
    libgz-plugin2 \
    libgz-transport13 \
    libgz-sensors8 \
    && rm -rf /var/lib/apt/lists/*

################

#RUN apt-get update && apt-get install --no-install-recommends -y \
RUN pip install setuptools==68

RUN echo "export GZ_VERSION=harmonic" >> ~/.bashrc \
    && echo "source /opt/ros/iron/setup.bash" >> ~/.bashrc \
    && echo "source /home/workspace/ros2_ws/install/local_setup.bash" >> ~/.bashrc \
    && echo "export GZ_SIM_RESOURCE_PATH=/home/workspace/ros2_ws/install/simulation/share/simulation/models" >> ~/.bashrc