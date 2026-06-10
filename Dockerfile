# ==============================
# Stage 1: Builder
# ==============================
FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Shanghai
ENV CONDA_DIR=/opt/conda
ENV PATH=$CONDA_DIR/bin:$PATH

# Setup Proxy (Optional)
ARG http_proxy
ARG https_proxy
ENV http_proxy=${http_proxy}
ENV https_proxy=${https_proxy}

# Use Aliyun Mirrors
RUN sed -i 's|http://archive.ubuntu.com/ubuntu/|http://mirrors.aliyun.com/ubuntu/|g' /etc/apt/sources.list && \
    sed -i 's|http://security.ubuntu.com/ubuntu/|http://mirrors.aliyun.com/ubuntu/|g' /etc/apt/sources.list

# Install Build Dependencies + CRITICAL X11/Rendering libs for Isaac Sim
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc-12 g++-12 cmake build-essential unzip git-lfs \
    libglu1-mesa-dev vulkan-tools wget \
    && update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-12 100 \
    && update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-12 100 \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh && \
    bash miniconda.sh -b -p $CONDA_DIR && \
    rm miniconda.sh && \
    $CONDA_DIR/bin/conda clean -afy

# Accept Conda TOS + Create Environment
RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r && \
    conda create -n unitree_sim_env python=3.11 -y && \
    conda clean -afy

# Switch to Conda Environment
SHELL ["conda", "run", "-n", "unitree_sim_env", "/bin/bash", "-c"]  

RUN conda install -y -c conda-forge "libgcc-ng>=12" "libstdcxx-ng>=12" && \
    apt-get update && apt-get install -y libvulkan1 vulkan-tools && rm -rf /var/lib/apt/lists/*

# Install PyTorch
RUN pip install --upgrade pip && \
    pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124

# Install Isaac Sim
RUN pip install "isaacsim[all,extscache]==5.1.0" --extra-index-url https://pypi.nvidia.com

# Create Workspace
RUN mkdir -p /home/code
WORKDIR /home/code

# Clone and install IsaacLab (Removed PIP_NO_DEPENDENCIES so flatdict and h5py install naturally)
RUN git clone https://github.com/isaac-sim/IsaacLab.git && \
    cd IsaacLab && \
    export ACCEPT_EULA=Y && \
    export ISAACSIM_ACCEPT_EULA=Y && \
    export OMNI_KIT_ACCEPT_EULA=Y && \
    ./isaaclab.sh --install
    
# Build CycloneDDS
RUN git clone https://github.com/eclipse-cyclonedds/cyclonedds -b releases/0.10.x /cyclonedds && \
    cd /cyclonedds && mkdir build install && cd build && \
    cmake .. -DCMAKE_INSTALL_PREFIX=../install && \
    cmake --build . --target install

# CycloneDDS Environment
ENV CYCLONEDDS_HOME=/cyclonedds/install

# Install unitree_sdk2_python
RUN git clone https://github.com/unitreerobotics/unitree_sdk2_python && \
    cd unitree_sdk2_python && pip install -e .

# Clone unitree_sim_isaaclab
RUN git clone https://github.com/unitreerobotics/unitree_sim_isaaclab.git /home/code/unitree_sim_isaaclab && \
    cd /home/code/unitree_sim_isaaclab && git submodule update --init --depth 1 && \
    cd teleimager && pip install -e . && \
    cd ../ && pip install -r requirements.txt

# ==============================
# Stage 2: Runtime
# ==============================
FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04 AS runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Shanghai
ENV CONDA_DIR=/opt/conda
ENV PATH=$CONDA_DIR/bin:$PATH

# Disable Isaac Sim Auto-Startup popups
ENV OMNI_KIT_ALLOW_ROOT=1
ENV OMNI_KIT_DISABLE_STARTUP=1

# Install Runtime Dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglu1-mesa git-lfs zenity unzip \
    && rm -rf /var/lib/apt/lists/*

# Copy Conda Environment and Code
COPY --from=builder /home/code/IsaacLab /home/code/IsaacLab
COPY --from=builder /home/code/unitree_sdk2_python /home/code/unitree_sdk2_python
COPY --from=builder /cyclonedds /cyclonedds
COPY --from=builder /opt/conda /opt/conda
COPY --from=builder /home/code/unitree_sim_isaaclab /home/code/unitree_sim_isaaclab

ENV CYCLONEDDS_HOME=/cyclonedds/install

# Bashrc Initialization
RUN echo 'source /opt/conda/etc/profile.d/conda.sh' >> ~/.bashrc && \
    echo 'conda activate unitree_sim_env' >> ~/.bashrc && \
    echo 'export OMNI_KIT_ALLOW_ROOT=1' >> ~/.bashrc && \
    echo 'export OMNI_KIT_DISABLE_STARTUP=1' >> ~/.bashrc

WORKDIR /home/code

# Default Command
CMD ["conda", "run", "-n", "unitree_sim_env", "/bin/bash"]