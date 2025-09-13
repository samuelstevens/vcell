#!/bin/bash

# Helper to read metadata
get_metadata() {
  curl -s -H "Metadata-Flavor: Google" \
    "http://metadata.google.internal/computeMetadata/v1/instance/attributes/$1"
}

echo "***TPU VM $(hostname) starting at $(date -Is)***" | tee /tmp/startup.log

# Clone repo
REPO="https://github.com/samuelstevens/vcell.git"
DEST="/tmp/vcell"
echo "1. Cloning repo $REPO into $DEST..." | tee -a /tmp/startup.log
sudo git clone "$REPO" "$DEST" 2>&1 | tee -a /tmp/startup.log || {
  echo "ERROR: git clone failed!" | tee -a /tmp/startup.log
  exit 1
}
echo "Cloned $REPO into $DEST successfully." | tee -a /tmp/startup.log

# Download dataset from GCS
DATA_DEST="/tmp/vcc_data"
mkdir -p "$DATA_DEST"
echo "2. Downloading dataset from GCS bucket vcell-bucket..." | tee -a /tmp/startup.log
gsutil cp gs://vcell-bucket/vcc_data.zip "$DATA_DEST/" || {
  echo "ERROR: gsutil download failed!" | tee -a /tmp/startup.log
  exit 1
}
echo "Downloaded dataset from GCS bucket vcell-bucket." | tee -a /tmp/startup.log

# Unzip and flatten dataset
echo "3. Unzipping dataset..." | tee -a /tmp/startup.log
unzip "$DATA_DEST/vcc_data.zip" -d "$DATA_DEST" || {
  echo "ERROR: unzip failed!" | tee -a /tmp/startup.log
  exit 1
}

if [[ -d "$DATA_DEST/vcc_data" ]]; then
    echo "4. Flattening nested vcc_data folder..." | tee -a /tmp/startup.log
    mv "$DATA_DEST/vcc_data/"* "$DATA_DEST/"
    rm -rf "$DATA_DEST/vcc_data"
fi
echo "Dataset ready in $DATA_DEST." | tee -a /tmp/startup.log

# Change to the cloned repo directory
cd /tmp/vcell

# Install uv
echo "5. Installing uv..." | tee -a /tmp/startup.log
sudo pip install --upgrade pip
sudo pip install uv
echo "Installed uv" | tee -a /tmp/startup.log

# Configure Weights & Biases
echo "6. Configuring wandb..." | tee -a /tmp/startup.log
WANDB_API_KEY=$(get_metadata wandb-api-key)
WANDB_PROJECT=$(get_metadata wandb-project)
WANDB_ENTITY=$(get_metadata wandb-entity)
echo "Configured wandb with project: $WANDB_PROJECT and entity: $WANDB_ENTITY." | tee -a /tmp/startup.log

# Get the experiment script from metadata
EXP_SCRIPT=$(get_metadata experiment-script)
ARGS=$(get_metadata experiment-args)
echo "7. Running experiments/$EXP_SCRIPT with arguments: $ARGS. See experiment log for details..." | tee -a /tmp/startup.log

# Run the experiment
sudo uv run "experiments/$EXP_SCRIPT" $ARGS 2>&1 | tee -a /tmp/experiment.log

echo "***Startup script completed at $(date -Is)***" | tee -a /tmp/startup.log