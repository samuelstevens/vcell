# scripts/download_replogle.py
import os

import beartype
import requests
import tyro
import vcell.helpers

urls = (
    "https://zenodo.org/records/7041849/files/ReplogleWeissman2022_K562_essential.h5ad",
    "https://zenodo.org/records/7041849/files/ReplogleWeissman2022_K562_gwps.h5ad",
)


@beartype.beartype
def main(dump_to: str, chunk_size_kb: int = 10, download: bool = True):
    os.makedirs(dump_to, exist_ok=True)

    chunk_size = int(chunk_size_kb * 1024)

    if download:
        for url in urls:
            # Download images.
            r = requests.get(images_url, stream=True)
            r.raise_for_status()

            # Needs to be updated to vcell.helpers.progress.
            t = tqdm.tqdm(
                total=int(r.headers["content-length"]),
                unit="B",
                unit_scale=1,
                unit_divisor=1024,
                desc="Downloading images",
            )
            with open(fpath, "wb") as fd:
                for chunk in r.iter_content(chunk_size=chunk_size):
                    fd.write(chunk)
                    t.update(len(chunk))
            t.close()

        print(f"Downloaded images: {images_zip_path}.")


if __name__ == "__main__":
    tyro.cli(main)
