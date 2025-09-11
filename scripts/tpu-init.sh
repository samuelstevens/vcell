#!/bin/bash
# TPU startup script — clones private GitHub repo and downloads dataset from GCS

set -e  # exit on any error

# Helper to read metadata
get_metadata() {
  curl -s -H "Metadata-Flavor: Google" \
    "http://metadata.google.internal/computeMetadata/v1/instance/attributes/$1"
}

# Log VM startup
echo "TPU VM $(hostname) starting at $(date -Is)" | tee /tmp/startup.log

# Install dependencies
sudo apt-get update -y
echo "Installed dependencies" | tee -a /tmp/startup.log

# Setup SSH directory
mkdir -p ~/.ssh
chmod 700 ~/.ssh
echo "Setup ssh dir" | tee -a /tmp/startup.log

# Get private SSH key from metadata
PRIVATE_KEY_B64=$(get_metadata github-ssh-key)
if [[ -z "$PRIVATE_KEY_B64" ]]; then
  echo "ERROR: github-ssh-key metadata not found!" | tee -a /tmp/startup.log
  exit 1
fi
echo "Got private key" | tee -a /tmp/startup.log

# Decode Base64 SSH key into ~/.ssh/id_ed25519
echo "$PRIVATE_KEY_B64" | base64 --decode > ~/.ssh/id_ed25519
chmod 600 ~/.ssh/id_ed25519
echo "SSH key installed" | tee -a /tmp/startup.log

# Configure SSH for GitHub
cat > ~/.ssh/config <<EOF
Host github.com
  IdentityFile ~/.ssh/id_ed25519
  IdentitiesOnly yes
  AddKeysToAgent yes
EOF
chmod 600 ~/.ssh/config
echo "Setup shh config" | tee -a /tmp/startup.log

# Add GitHub host key
ssh-keyscan github.com >> ~/.ssh/known_hosts
chmod 644 ~/.ssh/known_hosts

# Start ssh-agent and add key
eval "$(ssh-agent -s)" | tee -a /tmp/startup.log
ssh-add ~/.ssh/id_ed25519 | tee -a /tmp/startup.log

# Clone private repo
REPO="git@github.com:samuelstevens/vcell.git"
DEST="/tmp/vcell"
echo "Cloning repo $REPO into $DEST..." | tee -a /tmp/startup.log
git clone "$REPO" "$DEST" 2>&1 | tee -a /tmp/startup.log || {
  echo "ERROR: git clone failed!" | tee -a /tmp/startup.log
  exit 1
}
echo "Cloned $REPO into $DEST successfully" | tee -a /tmp/startup.log

# Download dataset from GCS
DATA_DEST="/tmp/vcc_data"
mkdir -p "$DATA_DEST"
echo "Downloading dataset from GCS bucket vcell-bucket..." | tee -a /tmp/startup.log
gsutil cp gs://vcell-bucket/vcc_data.zip "$DATA_DEST/" || {
  echo "ERROR: gsutil download failed!" | tee -a /tmp/startup.log
  exit 1
}

# Unzip dataset
echo "Unzipping dataset..." | tee -a /tmp/startup.log
unzip "$DATA_DEST/vcc_data.zip" -d "$DATA_DEST" || {
  echo "ERROR: unzip failed!" | tee -a /tmp/startup.log
  exit 1
}

if [[ -d "$DATA_DEST/vcc_data" ]]; then
    echo "Flattening nested vcc_data folder..." | tee -a /tmp/startup.log
    mv "$DATA_DEST/vcc_data/"* "$DATA_DEST/"
    rm -rf "$DATA_DEST/vcc_data"
fi

echo "Dataset ready in $DATA_DEST" | tee -a /tmp/startup.log

# Change to the cloned repo directory
cd /tmp/vcell

# Install uv
sudo pip install --upgrade pip
sudo pip install uv
echo "Installed uv" | tee -a /tmp/startup.log

# Run the experiment
echo "Running experiment" | tee -a /tmp/startup.log
sudo uv run experiments/05_training.py \
    --vcc /tmp/vcc_data/ \
    --data.h5ad-fpath /tmp/vcc_data/adata_Training.h5ad \
    --batch-size 32 \
    --n-train 100 2>&1 | tee -a /tmp/training.log

echo "Startup script completed at $(date -Is)" | tee -a /tmp/startup.log
