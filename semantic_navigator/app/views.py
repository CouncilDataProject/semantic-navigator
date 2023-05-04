#!/usr/bin/env python

import logging

import pandas as pd
from flask import (
    Blueprint,
    render_template,
)
from gcsfs import GCSFileSystem

from . import TEMPLATES_DIR

###############################################################################
# Logging

log = logging.getLogger(__name__)

###############################################################################
# Globals

# Store the current dataset version
GCP_PROJECT_ID = "sem-nav-eva-005"
DATASET_VERSION = "2023-05-02"

# Create a GCS File system with anon creds
FS = GCSFileSystem(GCP_PROJECT_ID, token="anon")

# Read the dataset from remote
DATASET = pd.read_parquet(
    f"gs://{GCP_PROJECT_ID}/{DATASET_VERSION}/dataset.parquet",
    storage_options={"token": "anon"},
)

###############################################################################
# Flask routing

views = Blueprint(
    "views",
    __name__,
    template_folder=TEMPLATES_DIR,
)

###############################################################################


@views.route("/", methods=["GET"])
def index() -> str:
    random_row_from_dataset = DATASET.sample(1).iloc[0]
    with FS.open(random_row_from_dataset.chunk_storage_path, "r") as open_f:
        example_random_text = open_f.read()

    return render_template(
        "index.html",
        example_random_text=example_random_text,
        example_random_text_path=random_row_from_dataset.chunk_storage_path,
    )
