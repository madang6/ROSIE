FROM ros:humble

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3-pip \
        python3-colcon-common-extensions \
    && rm -rf /var/lib/apt/lists/*

# Python deps
COPY requirements.txt /tmp/requirements.txt
RUN pip3 install --no-cache-dir -r /tmp/requirements.txt

# Application code
WORKDIR /app
COPY rosie/ ./rosie/
COPY openwebui/ ./openwebui/

# Source ROS 2 underlay on every command
SHELL ["/bin/bash", "-c"]
RUN echo "source /opt/ros/humble/setup.bash" >> /etc/bash.bashrc

ENV PYTHONPATH="/app:${PYTHONPATH}"
ENV ROS_DOMAIN_ID=0

EXPOSE 8000

# Default: run the FastAPI sidecar
CMD ["bash", "-c", "source /opt/ros/humble/setup.bash && uvicorn rosie.server:app --host 0.0.0.0 --port 8000"]
