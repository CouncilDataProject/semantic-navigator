import numpy as np
import pandas as pd
from typing import Tuple
from sklearn.linear_model import LogisticRegression

# parse out positives and negatives from json file (annotation results)
def sort_annotations(annotations) -> Tuple[list, list]:
    positives, negatives = [], []

    for chunk_id in annotations:
        if annotations[chunk_id] == 1:
            positives.append(chunk_id)
        else:
            negatives.append(chunk_id)
    return (positives, negatives)


def update_dataset(positives: list, negatives: list, DATASET: pd.DataFrame) -> pd.DataFrame:
    # set random seed for reproducibility
    # and set hyper parameters for random negative examples
    np.random.seed(0)
    n_negative_examples = 400

    # using the stored chunk ids, pull the positive embeddings
    positive_embeddings = np.stack(DATASET.loc[DATASET.chunk_id.isin(positives)].embedding)

    # using the stored chunk ids, pull the negative embeddings
    negative_embeddings = np.stack(DATASET.loc[DATASET.chunk_id.isin(negatives)].embedding)

    # randomly draw embeddings to be additional negative examples
    random_embeddings_for_negative = np.stack(DATASET.sample(n_negative_examples).embedding)

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
    train_embeddings = np.concatenate((positive_embeddings, complete_negative_embeddings), axis=0)
    train_labels = np.concatenate((
        # positives are 1
        np.ones(len(positive_embeddings)),
        # negatives are 0
        np.zeros(len(complete_negative_embeddings)),
    ))

    # create classifier
    clf = LogisticRegression(class_weight="balanced", random_state=1, max_iter=100000)

    # fit the model
    clf.fit(train_embeddings, train_labels)

    # removing previous annotated examples from the dataset
    THE_REST = DATASET.drop(DATASET.loc[DATASET.chunk_id.isin(positives)].index).drop(
        DATASET.loc[DATASET.chunk_id.isin(negatives)].index)

    # generate probabilities to use for next annotation cycle
    predictions = clf.predict_proba(np.stack(THE_REST.embedding))

    # only want to display positive probabilities
    positive_preds = predictions[:,1]

    # attach these to resulting dataframe for sorting
    THE_REST["prob"] = positive_preds
    NEXT_TWENTY_SAMPLES = THE_REST.sort_values(by="prob", ascending=False)[:20]

    return NEXT_TWENTY_SAMPLES