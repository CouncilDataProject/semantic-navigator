#!/usr/bin/env python

from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from werkzeug.datastructures import ImmutableMultiDict

###############################################################################


def sort_annotations(form_data: ImmutableMultiDict) -> Tuple[list, list]:
    positives, negatives = [], []

    for key, value in form_data.items():
        key_no_section = key.replace("selection_", "")
        if value == "Relevant":
            positives.append(key_no_section)
        else:
            negatives.append(key_no_section)

    return positives, negatives


def _basic_model_train(
    positives: list,
    negatives: list,
    dataset: pd.DataFrame,
    num_negatives_examples: int = 400,
) -> LogisticRegression:
    # set random seed for reproducibility
    # and set hyper parameters for random negative examples
    np.random.seed(0)

    # using the stored chunk ids, pull the positive embeddings
    positive_embeddings = np.stack(
        dataset.loc[dataset.chunk_id.isin(positives)].embedding
    )

    # using the stored chunk ids, pull the negative embeddings
    negative_embeddings = np.stack(
        dataset.loc[dataset.chunk_id.isin(negatives)].embedding
    )

    # randomly draw embeddings to be additional negative examples
    random_embeddings_for_negative = np.stack(
        dataset.sample(num_negatives_examples).embedding
    )

    # technically the user could have not given any negative examples
    # so safety check, "should these embeddings be combined or not"
    if len(negative_embeddings) > 0:
        complete_negative_embeddings = np.concatenate(
            [negative_embeddings, random_embeddings_for_negative],
            axis=0,
        )
    else:
        complete_negative_embeddings = random_embeddings_for_negative

    # construct training data
    train_embeddings = np.concatenate(
        (positive_embeddings, complete_negative_embeddings),
        axis=0,
    )
    train_labels = np.concatenate(
        (
            # positives are 1
            np.ones(len(positive_embeddings)),
            # negatives are 0
            np.zeros(len(complete_negative_embeddings)),
        )
    )

    # create classifier
    clf = LogisticRegression(class_weight="balanced", random_state=1, max_iter=100000)

    # fit the model
    clf.fit(train_embeddings, train_labels)

    return clf


def update_dataset(
    positives: list,
    negatives: list,
    dataset: pd.DataFrame,
    num_samples: int = 20,
    num_negatives_examples: int = 400,
) -> pd.DataFrame:
    # Get model
    clf = _basic_model_train(
        positives=positives,
        negatives=negatives,
        dataset=dataset,
        num_negatives_examples=num_negatives_examples,
    )

    # removing previous annotated examples from the dataset
    new_samples = dataset.drop(
        dataset.loc[dataset.chunk_id.isin(positives)].index,
    ).drop(
        dataset.loc[dataset.chunk_id.isin(negatives)].index,
    )

    # generate probabilities to use for next annotation cycle
    predictions = clf.predict_proba(np.stack(new_samples.embedding))

    # only want to display positive probabilities
    positive_preds = predictions[:, 1]

    # attach these to resulting dataframe for sorting
    new_samples["prob"] = positive_preds

    return new_samples.sort_values(by="prob", ascending=False)[:num_samples]


def get_final_dataset(
    positives: list,
    negatives: list,
    dataset: pd.DataFrame,
    num_negatives_examples: int = 400,
) -> pd.DataFrame:
    # Get model
    clf = _basic_model_train(
        positives=positives,
        negatives=negatives,
        dataset=dataset,
        num_negatives_examples=num_negatives_examples,
    )

    # generate probabilities to use for next annotation cycle
    predictions = clf.predict_proba(np.stack(dataset.embedding))

    # only want to display positive probabilities
    positive_preds = predictions[:, 1]

    # attach these to resulting dataframe for sorting
    dataset["prob"] = positive_preds

    return dataset.sort_values(by="prob", ascending=False)
