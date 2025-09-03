# Base image with ROS 2 ${ROS_DISTRO} Desktop
FROM cyberbotics/webots:R2025a-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV ROS_DISTRO=humble
ENV USER=bmstu
ARG UID=1000
ARG NVIDIA_GPU=1
ARG NVIDIA_DRIVER=570
ENV ROS_DOMAIN_ID=231
ARG DEBIAN_FRONTEND=noninteractive


RUN useradd -m -g users ubuntu && groupadd ubuntu
RUN usermod -l  bmstu ubuntu && \
    groupmod -n bmstu ubuntu && \
    usermod -d /bmstu -m  bmstu  && \
    usermod -c "BMSTU YRC" bmstu && \
    usermod -s /bin/bash bmstu
    # RUN usermod -G dialout
RUN echo "bmstu ALL=(ALL) NOPASSWD: ALL" >> /etc/ers

RUN apt-get update && apt-get install -y \
    git \
    nano \
    tmux \
    curl \
    wget \
    locales \
    && apt-get clean



RUN apt update && apt install locales -y \
    && locale-gen en_US en_US.UTF-8 \
    && update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8 \
    && export LANG=en_US.UTF-8

RUN apt install software-properties-common -y \
    && add-apt-repository universe
    
# Add ROS 2 apt repo
RUN apt-get update && apt-get install -y software-properties-common curl gnupg lsb-release \
    && curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
        | gpg --dearmor -o /usr/share/keyrings/ros-archive-keyring.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] \
        http://packages.ros.org/ros2/ubuntu $(lsb_release -cs) main" \
        | tee /etc/apt/sources.list.d/ros2.list > /dev/null

RUN apt-get update && apt-get install -y \
    python3-pip \
    build-essential \
    python3-colcon-common-extensions \
    ros-${ROS_DISTRO}-ros-base \
    ros-${ROS_DISTRO}-rqt \
    ros-${ROS_DISTRO}-rqt-common-plugins \
    # ros-${ROS_DISTRO}-realsense2-camera \
    # ros-${ROS_DISTRO}-gazebo-ros-pkgs \
    # ros-${ROS_DISTRO}-geometry2 \
    ros-${ROS_DISTRO}-webots-ros2 \
    ros-${ROS_DISTRO}-robot-localization \
    ros-${ROS_DISTRO}-pointcloud-to-laserscan \
    && apt-get clean

RUN apt-get update && apt-get install -y \
    ros-${ROS_DISTRO}-desktop \
    ros-dev-tools \
    && apt-get clean
# Set locale
# RUN locale-gen en_US en_US.UTF-8
# ENV LANG=en_US.UTF-8
# ENV LANGUAGE=en_US:en
# ENV LC_ALL=en_US.UTF-8

ENV DEBIAN_FRONTEND=noninteractive

RUN  apt-get update \
    &&  apt-get install -y ros-${ROS_DISTRO}-xacro ros-${ROS_DISTRO}-joint-state-publisher-gui \
    &&  apt-get install -y ros-${ROS_DISTRO}-tf-transformations  ros-${ROS_DISTRO}-rqt-tf-tree \
    &&  rm -rf /var/lib/apt/lists/*

RUN  apt update \
    &&  apt install -y ros-${ROS_DISTRO}-navigation2 \
    &&  apt install -y ros-${ROS_DISTRO}-nav2-bringup \
    &&  apt install -y ros-${ROS_DISTRO}-slam-toolbox \
    &&  rm -rf /var/lib/apt/lists/*

# Source ROS 2 on container start
RUN echo "source /opt/ros/${ROS_DISTRO}/setup.bash" >> /bmstu/.bashrc
# RUN echo "export PS1='${debian_chroot:+($debian_chroot)}\[\033[01;31m\]\u@\h\[\033[00m\]:\[\033[01;32m\]\w\[\033[00m\]\$ '" >> /bmstu/.bashrc
COPY ./bashrc  /tmp/bashrc
RUN cat /tmp/bashrc >> /bmstu/.bashrc &&\
    rm -f /tmp/bashrc &&\
    chown -R bmstu:bmstu /bmstu
    
# Install NVIDIA OpenGL libraries and tools for verification
# Install OpenGL libraries for both NVIDIA and Mesa fallback, plus tools
RUN  apt-get update &&  apt-get install -y --no-install-recommends \
    libnvidia-gl-${NVIDIA_DRIVER} \
    libgl1 \
    libglu1-mesa \
    mesa-utils \
    nvidia-utils-${NVIDIA_DRIVER} \
    &&  rm -rf /var/lib/apt/lists/*

# Create an entrypoint script to detect GPU and configure OpenGL
RUN echo '#!/bin/bash\n\
if ["${NVIDIA_GPU}"="1"]; then\n\
    echo "NVIDIA GPU detected, configuring for NVIDIA OpenGL"\n\
    export LIBGL_ALWAYS_INDIRECT=0\n\
    export __GLX_VENDOR_LIBRARY_NAME=nvidia\n\
else\n\
    echo "No NVIDIA GPU detected, falling back to Mesa"\n\
    export LIBGL_ALWAYS_INDIRECT=0\n\
    export __GLX_VENDOR_LIBRARY_NAME=mesa\n\'
#NVIDIA END


COPY ./ros_entrypoint.sh /ros_entrypoint.sh
RUN  chmod +x /ros_entrypoint.sh


USER bmstu
# Set working directory
WORKDIR /bmstu/ros2_ws

# Create entrypoint

ENTRYPOINT ["/ros_entrypoint.sh"]
CMD ["/bin/bash"]
