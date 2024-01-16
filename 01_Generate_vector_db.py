from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance
import os 
import pandas as pd 
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import numpy as np 
from qdrant_client.http import models
import uuid 
import json 

def add_vectors(client, collection_name, payloads, vectors, ids=None, parallel=None): 
    if ids is None:
        ids = iter(lambda: uuid.uuid4().hex, None)
    else:
        ids = iter(ids)
    
    records = [models.Record(id=next(ids), payload=payload, vector=vector.tolist()) for payload, vector in zip(payloads, vectors)]
    client.upload_records(
            collection_name=collection_name,
            records=records,
            # wait=True,
            # parallel=parallel or 1
        )
DIR = r"./BAAI--bge-large-en-v1_5"
client = QdrantClient(path=os.path.join("./Vector_db/", DIR))

for folder in tqdm(os.listdir(DIR), desc="Chunksizes"):
    client.recreate_collection(
        collection_name=folder,
        vectors_config=VectorParams(size=1024, distance=Distance.COSINE),
        optimizers_config=models.OptimizersConfigDiff(indexing_threshold=0),
        #shard_number=4
    )
    vectors = {}
    metadata = {}
    for file in tqdm(os.listdir(os.path.join(DIR, folder)), desc="Reading files"):
        if file.endswith(".npy"): 
            name = file.replace(".npy", "")
            array=np.load(os.path.join(DIR, folder, file))
            vectors[name]=array
            with open(os.path.join(DIR, folder, name+"_metadata.json"), "r") as f:
                data = json.load(f) 
            metadata[name] = data
    for key in tqdm(vectors.keys(), desc="Adding to Vector DB"):
        add_vectors(client, folder, metadata[key], vectors[key], ids=None, parallel=None)
    client.update_collection(collection_name=folder, optimizer_config=models.OptimizersConfigDiff(indexing_threshold=20000))
