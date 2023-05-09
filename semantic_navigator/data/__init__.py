"""Web app support package for semantic_navigator."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from datetime import datetime
from functools import partial
from itertools import chain
from logging import getLogger
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd
from cdp_backend.pipeline.transcript_model import Transcript
from dataclasses_json import DataClassJsonMixin
from gcsfs import GCSFileSystem
from sentence_transformers import SentenceTransformer

from .. import constants

###############################################################################

log = getLogger(__name__)

###############################################################################


@dataclass
class _TextChunkWithMeta(DataClassJsonMixin):
    chunk_id: str
    event_id: str
    start_time: float
    session_id: str
    session_index: int
    session_datetime: datetime
    text: str


@dataclass
class _TextChunkingError:
    event_id: str
    session_id: str
    session_index: int
    session_datetime: datetime
    error: str


def _get_text_chunks_from_transcript(
    session_details_row: pd.Series,
    char_count_thresh: int,
) -> list[_TextChunkWithMeta] | _TextChunkingError:
    try:
        # Read the transcript
        with open(session_details_row.transcript_path) as open_transcript:
            t = Transcript.from_json(open_transcript.read())

        # Store for all chunks
        completed_chunks = []

        # Storage for the current chunk
        current_chunk_sentences: list[str] = []
        current_chunk_len = 0
        current_chunk_start_time = 0
        for sentence in t.sentences:
            # If we have reached the max char count
            # store the current chunk then reset
            if current_chunk_len >= char_count_thresh:
                # Add chunk to dataset
                chunk_id = str(uuid4())

                # Add all metadata to dataset
                completed_chunks.append(
                    _TextChunkWithMeta(
                        chunk_id=chunk_id,
                        event_id=session_details_row.event_id,
                        start_time=current_chunk_start_time,
                        session_id=session_details_row.id,
                        session_index=session_details_row.session_index,
                        session_datetime=session_details_row.session_datetime,
                        text=" ".join(current_chunk_sentences),
                    )
                )

                # Reset the current chunk
                current_chunk_sentences = []
                current_chunk_len = 0

            # If no sentences in current chunk we are on a new chunk
            # Store the start time
            if current_chunk_len == 0:
                current_chunk_start_time = sentence.start_time

            # Append sentence to current list of chunk sentences
            current_chunk_sentences.append(sentence.text)
            current_chunk_len += len(sentence.text)

        return completed_chunks

    # If anything went wrong, raise
    except Exception as e:
        return _TextChunkingError(
            event_id=session_details_row.event.id,
            session_id=session_details_row.id,
            session_index=session_details_row.session_index,
            session_datetime=session_details_row.session_datetime,
            error=str(e),
        )


def _store_text_to_gcs(
    chunk_details: dict[str, str | float | int | datetime | np.ndarray],
    fs: GCSFileSystem,
    ds_isolation_path: str,
) -> dict[str, str | float | int | datetime | np.ndarray]:
    # Construct the complete store path
    chunk_store_path = (
        f"{ds_isolation_path}/processed-chunks/{chunk_details['chunk_id']}.txt"
    )

    # Upload
    with fs.open(chunk_store_path, "w") as open_text:
        open_text.write(chunk_details["text"])

    # Return completed object
    # Specifically removing the text data
    return {
        "chunk_id": chunk_details["chunk_id"],
        "event_id": chunk_details["event_id"],
        "start_time": chunk_details["start_time"],
        "session_id": chunk_details["session_id"],
        "session_index": chunk_details["session_index"],
        "session_datetime": chunk_details["session_datetime"],
        "chunk_storage_path": chunk_store_path,
        "embedding": chunk_details["embedding"],
    }


def _generate_cdp_sea_dataset(
    credentials_path: str | Path,
    n: int | None = None,
    char_count_thresh: int = 1024,
    embedding_model: str = constants.EMBEDDING_MODEL_NAME,
    random_seed: int = 12,
    debug: bool = False,
) -> str:
    from cdp_data import CDPInstances, datasets
    from tqdm.contrib.concurrent import process_map, thread_map

    ###############################################################

    # Set seeds
    random.seed(random_seed)
    np.random.seed(random_seed)

    ###############################################################

    # Load the embedding model
    model = SentenceTransformer(embedding_model)

    # Read credentials to get project
    with open(credentials_path) as open_f:
        creds = json.load(open_f)

    # Get project
    gcs_project_id = creds["project_id"]

    # Create filesystem connection
    fs = GCSFileSystem(gcs_project_id, token=str(credentials_path))

    # Get current datetime as a string
    # which we will isolate all parts of the dataset under
    dt_now = datetime.now().date().isoformat()

    # Get dataset isolated path
    ds_isolation_path = f"{gcs_project_id}/{dt_now}"

    # Get Seattle transcript data
    df = datasets.get_session_dataset(
        CDPInstances.Seattle,
        store_transcript=True,
        raise_on_error=False,
        replace_py_objects=True,
    )

    # Select sample
    if n is not None:
        df = df.sample(n)

    # Create partial of chunk creation function
    chunk_creation_func = partial(
        _get_text_chunks_from_transcript,
        char_count_thresh=char_count_thresh,
    )

    # Get all rows as a list for parallel processing
    df_rows = [row for _, row in df.iterrows()]

    # Process and flatten all results to a single list
    text_chunks_results = chain(
        *process_map(
            chunk_creation_func,
            df_rows,
            desc="Creating text chunks",
        )
    )

    # Remove any errors
    errors = []
    chunks = []
    for text_chunk_result in text_chunks_results:
        if isinstance(text_chunk_result, _TextChunkWithMeta):
            chunks.append(text_chunk_result)
        else:
            errors.append(text_chunk_result)

    # Log errors
    if len(errors) > 0:
        if debug:
            print("Some text chunk processing resulted in errors...")
            print("=" * 40)
            for error in errors:
                print(f"\tEvent: {error.event_id}")
                print(f"\tSession: {error.session_id}")
                print(f"\tIndex: {error.session_index}")
                print(f"\tDatetime: {error.session_datetime}")
                print(f"\tERROR: {error.error}")
                print("-" * 40)

    # Create embedding for all chunks
    embeddings = model.encode(
        [tc.text for tc in chunks],
        show_progress_bar=True,
    )

    # Combine embeddings to list of dicts
    chunks_with_embeddings = [
        {
            **tc.to_dict(),
            "embedding": embedding,
        }
        for tc, embedding in zip(chunks, embeddings)
    ]

    # Create partial text upload function
    text_upload_func = partial(
        _store_text_to_gcs,
        fs=fs,
        ds_isolation_path=ds_isolation_path,
    )

    # Threaded upload
    chunks_with_embeddings = thread_map(
        text_upload_func,
        chunks_with_embeddings,
        desc="Uploading chunks",
    )

    # Convert everything to a dataframe and store to parquet locally
    dataset = pd.DataFrame(chunks_with_embeddings)
    dataset.to_parquet("dataset.parquet")

    # Store a copy of the dataset to GCS
    remote_dataset_path = f"{ds_isolation_path}/dataset.parquet"
    dataset.to_parquet(
        f"gs://{remote_dataset_path}",
        storage_options={"token": credentials_path},
    )
    return remote_dataset_path
