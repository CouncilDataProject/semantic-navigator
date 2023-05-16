#!/usr/bin/env python

import logging

import pandas as pd
from flask import (
    Blueprint,
    render_template,
    request
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

@views.route("/", methods=["GET", "POST"])
def index() -> str:
    num_samples = 20
    random_rows_from_dataset = DATASET.sample(num_samples).iloc[:num_samples]

    texts_info = []
    for index, row in random_rows_from_dataset.iterrows():
        with FS.open(row.chunk_storage_path, "r") as open_f:
            example_random_text = open_f.read()

        
        link = 'https://councildataproject.org/seattle/#/events/%s?s=%s&t=%s' % (row.event_id, row.session_index, round(row.start_time))

        texts_info.append({
            'index': index,
            'text': example_random_text,
            'chunk_id': row.chunk_storage_path,
            'link': link,
        })
    

    if request.method == 'POST':
        output = request.get_json()
        # TODO: Update the dataframe

    return render_template(
        "index.html",
        texts_info=texts_info,
    )
