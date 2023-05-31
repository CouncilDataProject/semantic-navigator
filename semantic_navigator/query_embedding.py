#!/usr/bin/env python

import pandas as pd
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim

###############################################################################


def query_results(
    query: str, num_samples: int, dataset: pd.DataFrame, model: SentenceTransformer
) -> pd.DataFrame:
    # embed the query
    query_embedding = model.encode(query)

    # get similarity of query against all docs
    dataset["similarity"] = dataset.embedding.apply(
        lambda e: cos_sim(e, query_embedding).item()
    )

    # get top n samples
    first_n_samples = dataset.sort_values(
        by="similarity",
        ascending=False,
    )[:num_samples]

    return first_n_samples
