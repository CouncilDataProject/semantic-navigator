import pandas as pd
from sentence_transformers import SentenceTransformer
from semantic_navigator.constants import EMBEDDING_MODEL_NAME
from sentence_transformers.util import cos_sim

def query_results(query: str, num_samples: int, DATASET: pd.DataFrame) -> pd.DataFrame:
    # load the embedding model
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    # embed the query
    query_embedding = model.encode(query)
    
    # get similarity of query against all docs
    DATASET["similarity"] = DATASET.embedding.apply(lambda e: cos_sim(e, query_embedding).item())

    # get top 20 most similar text chunks
    first_20_results = DATASET.sort_values(by="similarity", ascending=False)[:num_samples]
    
    return first_20_results