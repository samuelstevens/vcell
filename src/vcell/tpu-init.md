## Running TPU init

We can pass metadata to our TPU create command including an initilization script and a github ssh key. In short, this script clones the vcell repo, downloads/unzips the vcc_data from a GCS instance and runs an experiment.

TODO: Remove the ssh nonsense and clone a public github repo. Allow user to pass bucket name, experiment name, wandb creds.

To get started:

1. Create a Google Cloud Storage bucket and upload vcc_data.zip. The script assumes your vcell bucket name is vcell-bucket.

2. Grant TPU VM access to your GCS bucket:

```sh
gcloud projects add-iam-policy-binding PROJECT_ID \
  --member="serviceAccount:PROJECT_ID-compute@developer.gserviceaccount.com" \
  --role="roles/storage.objectViewer"
```

3. Create SSH key (local) and add the public key to GitHub.

4. Base64 encode the private key.

5. Create a spot tpu and pass the startup script and base64 encoded github ssh key.

```sh
gcloud compute tpus tpu-vm create SOME_NAME --zone=ZONE \
--accelerator-type=TYPE --version=VERSION \
--metadata "github-ssh-key=<BASE64-ED25519-KEY>" --metadata-from-file startup-script=tpu-init.sh --spot
```

Startup logs can be found /tmp/startup and training logs in /tmp/training
