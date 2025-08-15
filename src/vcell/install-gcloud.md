## Install `gcloud`

A simple guide to installing the `gcloud` binary so that it can be easily uninstalled.

1. Download and extract the package.
2. Move it somewhere useful.
3. Add the directory to your $PATH.
4. Login to Google Cloud.

Download the right package from https://cloud.google.com/sdk/docs/install for your computer.

<details>
<summary>My Choice</summary>

I clicked the macOS tab and then chose the macOS 64-bit Apple Silicon option: https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/google-cloud-cli-darwin-arm.tar.gz

You can download and extract this to whatever directory you want. We will move it.

</details>

```
$ pwd
/Users/samstevens/Development/vcell
$ wget https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/google-cloud-cli-darwin-arm.tar.gz
--2025-08-15 09:51:22--  https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/google-cloud-cli-darwin-arm.tar.gz
Resolving dl.google.com (dl.google.com)... 74.125.21.91, 74.125.21.190, 74.125.21.136, ...
Connecting to dl.google.com (dl.google.com)|74.125.21.91|:443... connected.
HTTP request sent, awaiting response... 200 OK
Length: 56538937 (54M) [application/gzip]
Saving to: ‘google-cloud-cli-darwin-arm.tar.gz’

google-cloud-cli-darwin-arm.tar.gz              100%[====================================================================================================>]  53.92M  31.4MB/s    in 1.7s

2025-08-15 09:51:24 (31.4 MB/s) - ‘google-cloud-cli-darwin-arm.tar.gz’ saved [56538937/56538937]
$ ls
google-cloud-cli-darwin-arm.tar.gz
$ tar -xzf google-cloud-cli-darwin-arm.tar.gz  # -x extract -z use gzip -f filepath
$ ls
google-cloud-cli-darwin-arm.tar.gz  google-cloud-sdk/
```

I store all these crappy non-pip tools in `~/.local/pkg`. You can put it wherever you want because we will eventually add it to our `$PATH`.

```
$ mv google-cloud-sdk ~/.local/pkg
$ ls ~/.local/pkg
aws-cli/  google-cloud-sdk/
```

Then I add the `bin/` to my path.
I use fish, so it's just `fish_add_path ~/.local/pkg/google-cloud-sdk/bin`.
If you use a different shell, then do whatever you do to add `~/.local/pkg/google-cloud-sdk/bin` to your path.

Then you can run:

```
$ gcloud version
Google Cloud SDK 533.0.0
bq 2.1.22
core 2025.08.01
gcloud-crc32c 1.0.0
gsutil 5.35
Updates are available for some Google Cloud CLI components.  To install them,
please run:
  $ gcloud components update
```

Now you have gcloud installed in a way that is easy to delete!
You need to run `gcloud auth login` and login with your google account.
