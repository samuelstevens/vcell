## Running TPU init

We can pass metadata to our TPU create command that helps speed up the painful init process. 

To get started:

1. Create a Google Cloud Storage bucket and upload vcc_data.zip. The script assumes your vcell bucket name is vcell-bucket.

2. Grant TPU VM access to your GCS bucket:
* `$PROJECT_ID` is the id found on the Google UI thingy.

```sh
gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="serviceAccount:$PROJECT_ID-compute@developer.gserviceaccount.com" \
  --role="roles/storage.objectViewer"
```

5. Create a spot tpu and pass these new fields:
* `$SCRIPT` is the name of a file under /experiments. Ex: "05_training.py"
* `$ARGS` will be passed directly into $SCRIPT. Ex: "--vcc /tmp/vcc_data/ --data.h5ad-fpath /tmp/vcc_data/adata_Training.h5ad --wandb-key xxxxxx"

```sh
gcloud compute tpus tpu-vm create $NAME --zone=$ZONE \
--accelerator-type=$TYPE --version=e$VERSION \
--metadata "experiment-script=$SCRIPT,experiment-args=$ARGS" \ 
--metadata-from-file startup-script=tpu-init.sh --spot
```

Startup logs can be found at /tmp/startup.log and experiment logs at /tmp/experiment.log.
