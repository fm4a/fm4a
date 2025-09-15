#!/bin/bash
# AI Foundation Models for the Atmosphere
# This script sets up the required conda environment for the hands-on session of the
# AI Foundation Models for the Atmosphere workshop.

set +e

# Configuration variables
REPOSITORY_NAME="fm4a"
REPOSITORY_URL="https://github.com/fm4a/${REPOSITORY_NAME}.git"
REPOSITORY_PATH="."

PRITHVI_WX_WEIGHTS_DIR="data/weights/"
PRITHVI_WX_WEIGHTS_FILE="${PRITHVI_WX_WEIGHTS_DIR}/prithvi.wxc.rollout.600m.v1.pt"
PRITHVI_WX_WEIGHTS_URL="https://www.nsstc.uah.edu/data/sujit.roy/demo/consolidated.pth"

# Log messages with timestamps
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Clone repository if it doesn't exist already
clone_repository() {
    if [[ ! -d "$REPOSITORY_PATH" ]]; then
        log "Cloning repository from $REPOSITORY_URL"
        git -C clone "$REPOSITORY_URL"
        if [[ $? -ne 0 ]]; then
            log "ERROR: Failed to clone repository"
            return 1
        fi
        log "Repository cloned successfully"
    else
        log "Repository already exists at $REPOSITORY_PATH"
    fi
    return 0
}

# Download weather model weights
download_weather_model() {
    log "Downloading weather model weights"
    mkdir -p "$PRITHVI_WX_WEIGHTS_DIR"

    if [[ -f "$PRITHVI_WX_WEIGHTS_FILE" ]]; then
        log "Weather model weights already downloaded"
    else
        log "Downloading weather model weights from $PRITHVI_WX_WEIGHTS_URL"
        wget -O "$PRITHVI_WX_WEIGHTS_FILE" "$PRITHVI_WX_WEIGHTS_URL" --no-check-certificate
        if [[ $? -ne 0 ]]; then
            log "ERROR: Failed to download weather model weights"
            return 1
        fi
        log "Weather model weights downloaded successfully"
    fi
    return 0
}


# Create and setup conda environment
create_conda_env() {
    local env_name="$1"

    log "Setting up conda environment: $env_name"

    # Check if environment already exists
    if conda info --envs | grep -q "$env_name"; then
        log "Environment '$env_name' already exists"
    else
        log "Creating conda environment '$env_name'"
        conda create -n "$env_name" python=3.12 -y -q
	conda run -n $env_name pip install uv

        if [[ $? -ne 0 ]]; then
            log "ERROR: Failed to create conda environment '$env_name'"
            return 1
        fi

        # Activate the new environment and install packages
        log "Installing packages for '$env_name'"

        # Install requirements from the appropriate file
        local requirements_file="${REPOSITORY_PATH}/requirements.txt"
        if [[ -f "$requirements_file" ]]; then
            conda run -n $env_name uv pip install -r "$requirements_file"
            if [[ $? -ne 0 ]]; then
                log "WARNING: Some packages in requirements.txt may have failed to install"
            fi
        else
            log "WARNING: Requirements file not found at $requirements_file"
        fi

        # Special setup for weather environment
	download_weather_model

        # Register the kernel
        conda run -n fm4a python -m ipykernel install --user --name "$env_name" --display-name "$env_name"
    fi
    download_weather_model

    log "Setup complete for environment: $env_name"
    return 0
}

# Main function to orchestrate the setup
main() {
    log "Starting workshop environment setup"

    # Ensure conda is available
    if ! command_exists conda; then
        log "ERROR: conda is not installed or not in PATH"
        return 1
    fi

    # Activate base conda environment
    log "Activating base conda environment"
    source /opt/conda/bin/activate

    # Clone the repository
    clone_repository || return 1

    # Setup each environment
    log "Setting up all environments found in $environments_dir"
    create_conda_env fm4a

    log "Workshop environment setup completed successfully"
    return 0
}

# Run the main function
main

# Exit on error, but allow for unbound variables
set -e
exit $?
