"""Web app support package for semantic_navigator."""

import shutil
from logging import getLogger
from pathlib import Path

import pandas as pd

###############################################################################

DATA_DIR = Path(__file__).resolve().parent
CDP_SEATTLE_SMALL_DATASET = DATA_DIR / "sem-nav-cdp-sea-small.parquet"
CDP_SEATTLE_SMALL_TEXT_FILES_ARCHIVE = (
    DATA_DIR / "sem-nav-cdp-sea-small-processed-chunks.tar.gz"
)

###############################################################################

log = getLogger(__name__)

###############################################################################


def load_cdp_sea_small() -> pd.DataFrame:
    # Load the dataframe at the very least
    dataset = pd.read_parquet(CDP_SEATTLE_SMALL_DATASET)

    # Unpack the archive
    shutil.unpack_archive(CDP_SEATTLE_SMALL_TEXT_FILES_ARCHIVE)

    return dataset
