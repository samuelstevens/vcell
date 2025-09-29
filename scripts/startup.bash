#!/bin/env bash
set -euo pipefail  # Exit on error, undefined vars, pipe failures

# Configuration
readonly LOG_FILE="/tmp/startup.log"

# Logging function
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

# Get metadata value from Google Cloud metadata server (required)
get_metadata() {
    local key="$1"

    if ! command -v curl >/dev/null 2>&1; then
        log "Error: curl not available to fetch metadata"
        exit 1
    fi

    local value=$(curl -s "http://metadata.google.internal/computeMetadata/v1/instance/attributes/$key" \
        -H "Metadata-Flavor: Google" 2>/dev/null || echo "")

    if [ -z "$value" ]; then
        log "Error: Required metadata key '$key' not found"
        exit 1
    fi

    echo "$value"
}

# Start logging - append to log file and show on terminal
exec > >(tee -a "$LOG_FILE")
exec 2>&1

# Error handling
trap 'log "Error on line $LINENO" >&2' ERR
trap 'log "Script interrupted" >&2; exit 130' INT TERM

# Get all required metadata upfront
log "Reading metadata configuration."
readonly COMMIT=$(get_metadata "git-commit")
readonly GCS_BUCKET=$(get_metadata "gcs-bucket")
readonly EXP_PATH=$(get_metadata "exp-path")
readonly EXP_ARGS=$(get_metadata "exp-args")
log "Metadata configuration loaded."

# Export Weights & Biases environment variables
export WANDB_API_KEY=$(get_metadata "wandb-api-key")
export WANDB_PROJECT=$(get_metadata "wandb-project")
export WANDB_ENTITY=$(get_metadata "wandb-entity")
log "Weights & Biases environment variables set."

# Install uv
log "Installing uv."
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.local/bin/env
log "Installed uv."

log "Cloning vcell repository."
git clone https://github.com/samuelstevens/vcell.git ~/vcell
log "Cloned vcell repository."
cd ~/vcell

log "Checking out commit: $COMMIT."
git checkout "$COMMIT"
log "Checked out commit: $COMMIT."

# Download dataset from GCS
export ROOT=~/data
log "Creating data directory '$ROOT'."
mkdir -p $ROOT
log "Created data directory '$ROOT'."

log "Downloading data from GCS: $GCS_BUCKET"
gcloud storage cp -r "$GCS_BUCKET" $ROOT
log "Downloaded data from GCS."

# Run experiment
log "Running experiment: $EXP_PATH"

log "Starting experiment."
# Use eval to properly expand the arguments as separate parameters
uv run $EXP_PATH $EXP_ARGS
log "Experiment completed."

log "Completed successfully."
