## Make a TPU VM

TL;DR:

```sh
gcloud compute tpus tpu-vm create $NAME --project $PROJECT_ID --zone $ZONE --accelerator-type $TPU_VERSION --version $IMAGE
```

![Image of my google cloud account, showing that my project trc-project has ID trc-project-466816](docs/assets/console.jpg)

![Image of my email from TRC, showing that I have access to `v2-8` TPU VMs in `us-central1-f`](docs/assets/tpu-email.jpg)
