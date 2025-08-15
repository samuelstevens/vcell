## Make a TPU VM

TL;DR:

```sh
gcloud compute tpus tpu-vm create $NAME \
  --project $PROJECT_ID \
  --zone $ZONE \
  --accelerator-type $TPU_VERSION \
  --version $IMAGE
```

* `$NAME` can be whatever you want. You will reference it when you want to connect to the TPU VM.
* `$PROJECT_ID` is the project ID you shared with the TPU Research Cloud program folks. It's in the email you get from them (see below).
* The `$TPU_VERSION` and `$ZONE` variables depend on what you get access to. That's in your email.
* The `$IMAGE` variable depends on the `$TPU_VERSION`, and you have to look it up here: https://cloud.google.com/tpu/docs/runtimes. v4 and older use tpu-ubuntu2204-base.

![Image of my email from TRC, showing that I have access to `v2-8` TPU VMs in `us-central1-f`](/docs/assets/tpu-email.jpg)

You will need to run this command multiple times until it succeeds.

Once you have a TPU VM, you can see them all with:

```
$ gcloud compute tpus tpu-vm list --project $PROJECT_ID --zone $ZONE
NAME   ZONE           ACCELERATOR_TYPE  TYPE  TOPOLOGY  NETWORK  RANGE          STATUS
tpu-2  us-central1-f  v2-8              V2    2x2       default  10.128.0.0/20  READY
```

Then you can connect via SSH:

```sh
gcloud compute tpus tpu-vm ssh $NAME \
  --project $PROJECT_ID \
  --zone $ZONE
```

You should have sudo access.
