#!/usr/bin/env python

import logging
from typing import Dict, List

import pandas as pd
from flask import Blueprint, redirect, render_template, request, url_for
from gcsfs import GCSFileSystem
from sentence_transformers import SentenceTransformer

from semantic_navigator.constants import EMBEDDING_MODEL_NAME
from semantic_navigator.query_embedding import query_results
from semantic_navigator.train import get_final_dataset, sort_annotations, update_dataset

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

# load the embedding model
EMBEDDING_MODEL = SentenceTransformer(EMBEDDING_MODEL_NAME)

# Store global positives and negatives history
POSITIVES: List[str] = []
NEGATIVES: List[str] = []

###############################################################################
# Flask routing

views = Blueprint(
    "views",
    __name__,
    template_folder=TEMPLATES_DIR,
)

###############################################################################


def _create_text_infos(dataset: pd.DataFrame, with_proba: bool = False) -> List[Dict]:
    texts_info = []
    for index, row in dataset.iterrows():
        with FS.open(row.chunk_storage_path, "r") as open_f:
            text_selection = open_f.read()

        link = (
            f"https://councildataproject.org/seattle/"
            f"#/events/{row.event_id}?s={row.session_index}&t={round(row.start_time)}"
        )
        info = {
            "index": index,
            "date": row.session_datetime.date(),
            "text": text_selection,
            "chunk_id": row.chunk_id,
            "link": link,
        }

        # Add optional probability
        if with_proba:
            info["proba"] = row.prob

        texts_info.append(info)

    return texts_info


@views.route("/", methods=["GET", "POST"])
def index() -> str:
    # Handle search
    if request.method == "POST":
        query = request.form.get("search")
        return redirect(
            url_for(
                "views.train",
                q=query,
            )
        )

    # Handle landing page
    return render_template("index.html")


@views.route("/train/<q>", methods=["GET", "POST"])
def train(q: str) -> str:
    # Import global storage
    global POSITIVES
    global NEGATIVES

    # If we were rerouted here from index
    # Load with query similarity
    if request.method == "GET":
        # query will come from index (landing page)
        first_20_results = query_results(q, 20, DATASET, EMBEDDING_MODEL)

        # formatting to get necessary attributes for displaying
        text_infos = _create_text_infos(first_20_results)

    # If we were routed here from clicking the train button / post
    # Get new annotations, update global annotations, train model, render new
    if request.method == "POST":
        # get positives and negatives from user's annotations
        batch_positives, batch_negatives = sort_annotations(request.form)

        # Update global storage
        POSITIVES = [
            *POSITIVES,
            *batch_positives,
        ]
        NEGATIVES = [
            *NEGATIVES,
            *batch_negatives,
        ]

        # Update dataset / train and update
        next_20_best_samples = update_dataset(POSITIVES, NEGATIVES, DATASET)[:20]
        text_infos = _create_text_infos(next_20_best_samples)

    return render_template("train.html", texts_info=text_infos)


@views.route("/results", methods=["GET"])
def results() -> str:
    # Train final model and get proba dataset
    top_50_similar = get_final_dataset(POSITIVES, NEGATIVES, DATASET)[:50]
    text_infos = _create_text_infos(top_50_similar, with_proba=True)

    return render_template("results.html", texts_info=text_infos)
