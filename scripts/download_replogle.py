# scripts/download_replogle.py
import logging
import pathlib
from typing import Literal

import beartype
import requests
import tyro

from vcell.helpers import progress

log_format = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=log_format)
logger = logging.getLogger("replogle")

url_map = {
    "essential": "https://zenodo.org/records/7041849/files/ReplogleWeissman2022_K562_essential.h5ad",
    "gwps": "https://zenodo.org/records/7041849/files/ReplogleWeissman2022_K562_gwps.h5ad",
}


@beartype.beartype
def main(
    dump_to: str,
    dataset: Literal["essential", "gwps", "both"] = "both",
    chunk_size_kb: int = 10,
):
    """Download Replogle et al. 2022 datasets.

    Args:
        dump_to: Directory to save downloaded files
        dataset: Which dataset(s) to download: "essential", "gwps", or "both"
        chunk_size_kb: Download chunk size in KB
    """

    dump_path = pathlib.Path(dump_to)
    dump_path.mkdir(parents=True, exist_ok=True)

    chunk_size = int(chunk_size_kb * 1024)

    # Determine which URLs to download
    if dataset == "both":
        urls_to_download = list(url_map.values())
        datasets_to_download = list(url_map.keys())
    else:
        urls_to_download = [url_map[dataset]]
        datasets_to_download = [dataset]

    logger.info(f"Downloading dataset(s): {', '.join(datasets_to_download)}")

    for url in urls_to_download:
        # Extract filename from URL
        filename = url.split("/")[-1]
        fpath = dump_path / filename

        logger.info(f"Downloading {filename}...")

        # Download file
        r = requests.get(url, stream=True)
        r.raise_for_status()

        # Get total size
        total_size = int(r.headers.get("content-length", 0))

        # Calculate total chunks for progress tracking
        total_chunks = (total_size + chunk_size - 1) // chunk_size

        # Download with progress
        with open(fpath, "wb") as fd:
            chunks = progress(
                r.iter_content(chunk_size=chunk_size),
                desc=filename,
                total=total_chunks,
                every=total_chunks // 50,
            )
            for chunk in chunks:
                if chunk:  # filter out keep-alive chunks
                    fd.write(chunk)

        logger.info(f"Downloaded: {fpath}")

    logger.info(f"All files downloaded to: {dump_path}")


if __name__ == "__main__":
    tyro.cli(main)
