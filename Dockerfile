# --------------------------------------------
# Stage 1: CUDA-independent base with Python, tools
# --------------------------------------------
FROM ubuntu:24.04 as base-env

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Seoul

# Add PostgreSQL APT repository for version 18
RUN apt-get update && apt-get install -y curl gnupg && \
    mkdir -p /etc/apt/keyrings && \
    curl -fsSL https://www.postgresql.org/media/keys/ACCC4CF8.asc | gpg --dearmor -o /etc/apt/keyrings/postgresql.gpg && \
    echo "deb [signed-by=/etc/apt/keyrings/postgresql.gpg] http://apt.postgresql.org/pub/repos/apt noble-pgdg main" > /etc/apt/sources.list.d/pgdg.list

# Install core dependencies + PostgreSQL client 18
RUN apt-get update && apt-get install -y \
    tzdata gnupg git curl wget zip vim sudo tmux \
    make ninja-build g++ build-essential checkinstall \
    libssl-dev libsqlite3-dev libncursesw5-dev tk-dev \
    libgdbm-dev libc6-dev libbz2-dev libreadline-dev libffi-dev \
    liblzma-dev libgdm-dev zlib1g-dev \
    swig libblas-dev liblapack-dev libatlas-base-dev libgflags-dev \
    nodejs npm \
    language-pack-en postgresql-client-18 && \
    ln -fs /usr/share/zoneinfo/$TZ /etc/localtime && \
    echo $TZ > /etc/timezone && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Install SQLite
RUN apt-get update && \
    apt-get install -y sqlite3 libsqlite3-dev && \
    apt-get clean && rm -rf /var/lib/apt/lists/*


# Build Python 3.14 (latest version)
WORKDIR /usr/src
RUN PYTHON_VERSION=$(curl -s https://www.python.org/ftp/python/ | grep -oP '3\.14\.\d+' | sort -V | tail -1) && \
    curl -O https://www.python.org/ftp/python/${PYTHON_VERSION}/Python-${PYTHON_VERSION}.tgz && \
    tar -xvf Python-${PYTHON_VERSION}.tgz && \
    cd Python-${PYTHON_VERSION} && \
    ./configure --enable-optimizations && \
    make -j"$(nproc)" && make altinstall && \
    cd .. && rm -rf Python-${PYTHON_VERSION} Python-${PYTHON_VERSION}.tgz

# Set Python alternatives
RUN update-alternatives --install /usr/bin/python python /usr/local/bin/python3.14 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/local/bin/python3.14 1 && \
    python3.14 -m ensurepip && \
    python3.14 -m pip install --upgrade pip setuptools numpy

# Set pip to python3.14
RUN ln -s /usr/local/bin/pip3.14 /usr/bin/pip


# --------------------------------------------
# Stage 2: Final image with CUDA
# --------------------------------------------
FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04

# Copy base environment
COPY --from=base-env / /

# Set environment variables
ENV PATH="${PATH}:/usr/local/cuda/bin"
ENV LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/usr/local/cuda/lib64"
ENV CUDA_HOME=/usr/local/cuda
ENV PYTHONPATH=./

# Create a non-root user
RUN useradd -ms /bin/bash user && \
    echo "user:user" | chpasswd && \
    usermod -aG sudo user

# Not sure why, but these lines are necesary to use "user" as sudoer
RUN chown -R user:user /home/user
RUN chmod 4755 /usr/bin/sudo

# Set working directory and user
USER user
WORKDIR /home/user


