## Running TPU init

We can pass metadata to our TPU create command including an initilization script and a github ssh key.

```sh
gcloud compute tpus tpu-vm create maciejtest21 --zone=us-central1-a --accelerator-type=v3-8 --version=tpu-vm-base --metadata "github-ssh-key=<BASE64-ED25519-KEY>" --metadata-from-file startup-script=tpu-init.sh --spot
```
