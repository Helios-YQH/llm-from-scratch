from pathlib import Path, PurePosixPath

import modal

from lm_data.common import MODAL_SHARED_PATH

# Modal is not used here; this is a placeholder name (Modal object construction is
# lazy and never goes online). Modules such as scripts/train.py require it not to
# be "TODO" at import time.
RUN_USER = "selfstudy"

(DATA_PATH := Path("data")).mkdir(exist_ok=True)

app = modal.App(f"data-{RUN_USER}")
data_volume = modal.Volume.from_name(f"data-{RUN_USER}", create_if_missing=True, version=2)
shared_data_volume = modal.Volume.from_name(
    "shared-data", create_if_missing=True, version=2, environment_name="shared-data"
)


def build_image(*, include_tests: bool = False) -> modal.Image:
    image = modal.Image.debian_slim(python_version="3.12")
    image = image.uv_sync()
    image = image.add_local_python_source("lm_basics")
    image = image.add_local_python_source("lm_data")
    if include_tests:
        image = image.add_local_dir("tests", remote_path="/root/tests")
    return image


VOLUME_MOUNTS: dict[str | PurePosixPath, modal.Volume | modal.CloudBucketMount] = {
    "/root/data": data_volume,
    str(MODAL_SHARED_PATH): shared_data_volume.read_only(),
}

MODAL_SECRETS = []
