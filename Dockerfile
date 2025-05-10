# Use ROS Humble as the base image
FROM ros:humble

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-opencv \
    python3-tk \
    ros-humble-cv-bridge \
    ros-humble-sensor-msgs \
    ros-humble-geometry-msgs \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies using pip
RUN pip3 install --no-cache-dir \
    numpy \
    scipy \
    trimesh \
    ultralytics \
    pyquaternion \
    pillow
    #rospkg \
    #rclpy

# Copy project files into the container
COPY . /app

# Set entry point
#ENTRYPOINT ["/bin/bash"]
