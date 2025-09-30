#!/bin/bash

set -euo pipefail  # Exit on error, undefined vars, pipe failures

# Configuration
readonly LOG_FILE="/tmp/startup.log"

# Start logging - append to log file and show on terminal
exec > >(tee -a "$LOG_FILE")
exec 2>&1

# Helper to log with timestamp
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

# Helper to read metadata
get_metadata() {
  curl -s -H "Metadata-Flavor: Google" "http://metadata.google.internal/computeMetadata/v1/instance/attributes/$1"
}

log "***TPU VM $(hostname) starting.***"

# Error handling
trap 'log "Error on line $LINENO" >&2' ERR
trap 'log "Script interrupted" >&2; exit 130' INT TERM

# Get all required metadata upfront
log "Reading metadata configuration."
COMMIT=$(get_metadata "git-commit")
readonly COMMIT
GCS_BUCKET=$(get_metadata "gcs-bucket")
readonly GCS_BUCKET
EXP_SCRIPT=$(get_metadata "exp-script")
readonly EXP_SCRIPT
EXP_ARGS=$(get_metadata "exp-args")
readonly EXP_ARGS
log "Metadata configuration loaded."

# Clone repo
readonly CODE_REPO="https://github.com/samuelstevens/vcell.git"
readonly CODE_DEST="/tmp/vcell"

log "1. Cloning repo $CODE_REPO into $CODE_DEST..." 
git clone "$CODE_REPO" "$CODE_DEST"
cd $CODE_DEST
git checkout "$COMMIT"
log "Cloned $CODE_REPO into $CODE_DEST successfully." 

# Download dataset from GCS
readonly DATA_DEST="/tmp/vcc_data"
mkdir -p "$DATA_DEST"
log "2. Downloading dataset from GCS bucket $GCS_BUCKET..." 
gcloud storage cp "$GCS_BUCKET" "$DATA_DEST"
log "Downloaded dataset from GCS bucket $GCS_BUCKET." 

# Install uv
log "3. Installing uv..." 
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.local/bin/env
log "Installed uv" 

log "4. Running $EXP_SCRIPT with arguments: $EXP_ARGS." 

# Run the experiment
uv run --extra tpu "$EXP_SCRIPT" $EXP_ARGS

log "***Startup script completed at $(date -Is)***" 
