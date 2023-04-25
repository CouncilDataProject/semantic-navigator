#!/usr/bin/env python

import logging

from flask import (
    Blueprint,
    render_template,
)

from ..data import load_cdp_sea_small
from . import TEMPLATES_DIR

###############################################################################

log = logging.getLogger(__name__)

###############################################################################

views = Blueprint(
    "views",
    __name__,
    template_folder=TEMPLATES_DIR,
)

CDP_EXAMPLE_DATASET = load_cdp_sea_small()

###############################################################################


@views.route("/", methods=["GET"])
def index() -> str:
    random_row_from_dataset = CDP_EXAMPLE_DATASET.sample(1).iloc[0]
    with open(random_row_from_dataset.chunk_text_path) as open_f:
        example_random_text = open_f.read()

    return render_template(
        "index.html",
        example_random_text=example_random_text,
    )
