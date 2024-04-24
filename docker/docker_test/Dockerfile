FROM --platform=linux/amd64 pytorch/pytorch:2.2.1-cuda12.1-cudnn8-runtime
# Use a 'large' base container to show-case how to load tensorflow or pytorch and use the GPU (when enabled)

# Ensures that Python output to stdout/stderr is not buffered: prevents missing information when terminating
ENV PYTHONUNBUFFERED 1

RUN groupadd -r user && useradd -m --no-log-init -r -g user user
USER user

WORKDIR /opt/app

# Copy the environment.yml file into the Docker image
COPY --chown=user:user environment.yml /opt/app/environment.yml

# Create the conda environment
RUN conda env create -f /opt/app/environment.yml && \
    echo "source activate cellpainting" >> ~/.bashrc && \
    conda init bash

# Set the PATH to include the conda environment's bin directory
ENV PATH /home/user/.conda/envs/cellpainting/bin:$PATH

COPY --chown=user:user resources /opt/app/resources

#Copy your python scripts
COPY --chown=user:user inference.py /opt/app/
COPY --chown=user:user mynetwork.py /opt/app/
ENTRYPOINT ["python", "inference.py"]
