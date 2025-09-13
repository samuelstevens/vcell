## TPU Tricks

`gcloud` can `scp` files and folders to and from TPU VMs.

For instance, to get the vcc.zip file onto your TPU VM:

```sh
gcloud compute tpus tpu-vm scp $VCC_ZIP $NAME:$PROJECT_ROOT/data/inputs \
  --project $PROJECT_ID \
  --zone $ZONE
```

For me, from my project root, I run:

```sh
gcloud compute tpus tpu-vm scp data/inputs/vcc.zip tpu-2:projects/vcell/data/inputs --project trc-project-466816 --zone us-central1-f
```

Then on my TPU VM, I just run `unzip vcc.zip`.

Or to get the predictions file onto my laptop so that I can submit it:

```sh
gcloud compute tpus tpu-vm scp $NAME:$PROJECT_ROOT/pred_raw.h5ad . \
  --project $PROJECT_ID \
  --zone $ZONE
```

For me, from my project root, I run:

```sh
gcloud compute tpus tpu-vm scp tpu-2:~/projects/vcell/pred_raw.h5ad . --project trc-project-466816 --zone us-central1-f
```

Note the `.` to signify where on your laptop it should go; `.` means this directory.

## Adding Persistent Disks

gcloud compute disks create tpu-data --size 200GB --type pd-standard --zone us-central1-f --project trc-project-466816

gcloud alpha compute tpus tpu-vm attach-disk tpu-2 --zone us-central1-f --disk tpu-data --project trc-project-466816 --mode read-write
