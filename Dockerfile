# Use Miniconda base image for Linux amd64
FROM continuumio/miniconda3:latest

# Set working directory
WORKDIR /app

# Copy entire project into image
COPY . .

# Set environment name (must match environment.yml)
ARG ENV_NAME=haar

# Create the conda environment
RUN conda env create -f environment.yml

# Install any additional pip requirements for API
RUN conda run -n ${ENV_NAME} pip install --no-cache-dir -r requirements_api.txt

# Initialize git submodules (if any)
RUN git submodule update --init --recursive

# Ensure subsequent RUN, CMD use the correct environment
SHELL ["conda", "run", "-n", "haar", "/bin/bash", "-c"]

# Set PATH for convenience
ENV PATH /opt/conda/envs/${ENV_NAME}/bin:$PATH

# Make sure your start script and download script are executable
RUN chmod +x start_api.sh && chmod +x scripts/download.sh

# Expose the default FastAPI port
EXPOSE 8000

# Run FastAPI API server as container entrypoint
ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "haar", "./start_api.sh"]
