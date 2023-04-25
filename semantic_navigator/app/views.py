#!/usr/bin/env python

import logging

from flask import (
    Blueprint,
    render_template,
)

from . import TEMPLATES_DIR

###############################################################################

log = logging.getLogger(__name__)

###############################################################################

views = Blueprint(
    "views",
    __name__,
    template_folder=TEMPLATES_DIR,
)

###############################################################################


@views.route("/", methods=["GET"])
def index() -> str:
    return render_template("index.html")
