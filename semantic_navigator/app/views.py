#!/usr/bin/env python

import logging

import pandas as pd
from flask import (
    Blueprint,
    render_template,
    request
)
from gcsfs import GCSFileSystem
from semantic_navigator.train import sort_annotations, update_dataset
from semantic_navigator.query_embedding import query_results

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


@views.route("/", methods=["GET","POST"])
def index() -> str:
    random_row_from_dataset = DATASET.sample(1).iloc[0]
    with FS.open(random_row_from_dataset.chunk_storage_path, "r") as open_f:
        example_random_text = open_f.read()

    #if request.method == "POST":
         # TODO: parse search query

    return render_template(
        "index.html",
        example_random_text=example_random_text,
        example_random_text_path=random_row_from_dataset.chunk_storage_path,
        example_random_text_session=random_row_from_dataset.session_datetime
    )

@views.route("/", methods=["GET","POST"])
def train() -> str:
    # generate best guesses based on search query
    num_samples = 20

    # query will come from index (landing page)
    first_20_results = query_results(query, num_samples, DATASET)

    # formatting to get necessary attributes for displaying
    texts_info = []
    for index, row in first_20_results.iterrows():
        with FS.open(row.chunk_storage_path, "r") as open_f:
            example_random_text = open_f.read()
        
        link = 'https://councildataproject.org/seattle/#/events/%s?s=%s&t=%s' % (row.event_id, row.session_index, round(row.start_time))
        texts_info.append({
            'index': index,
            'text': example_random_text,
            'chunk_id': row.chunk_storage_path,
            'link': link,
        })
    
    # update results based on user's annotations
    if request.method == 'POST':
        output = request.get_json()

        # get positives and negatives from user's annotations
        positives, negatives = sort_annotations(output)

        # update dataframe
        next_20_best_samples = update_dataset(positives, negatives, DATASET)

        # matching format to texts_info
        texts_info = []
        for index, row in next_20_best_samples.iterrows():
            with FS.open(row.chunk_storage_path, "r") as open_f:
                transcript = open_f.read()
        
            link = 'https://councildataproject.org/seattle/#/events/%s?s=%s&t=%s' % (row.event_id, row.session_index, round(row.start_time))
            texts_info.append({
            'index': index,
            'text': transcript,
            'chunk_id': row.chunk_storage_path,
            'link': link,
            })
        
    return render_template(
        "index.html",
        texts_info=texts_info,
    )