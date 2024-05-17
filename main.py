from qdrant_client import QdrantClient, models
import google.generativeai as genai
import json
import os
from dotenv import load_dotenv

load_dotenv()

def load_qdrant():
    return QdrantClient(
        url=os.getenv('QDRANT_URL'),
        api_key=os.getenv("QDRANT_CLUSTER_API_KEY")
    )

def create_qdrant_collection(qdrant_client):
    qdrant_client.create_collection(
        collection_name="law-collection",
        vectors_config=models.VectorParams(
            size=768,
            distance=models.Distance.COSINE
        )
    )

def qdrant_inference(qdrant_client, collection_name, content):
    return qdrant_client.search(
        collection_name=collection_name,
        query_vector=genai.embed_content(model="models/embedding-001", content=content, task_type="retrieval_document", title="Judgement")["embedding"],
        limit=3
    )

def configure_gemini():
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

def get_embeddings(contents, title):
    return genai.embed_content(
        model="models/embedding-001",
        content=contents,
        task_type="retrieval_document",
        title=title
    )

def qdrant_upload_points(qdrant_client, collection_name, points):
    qdrant_client.upload_points(
        collection_name=collection_name,
        points=points,
    )

def main():
    configure_gemini()
    qdrant_client = load_qdrant()

    contents = []
    with open("./Data/dataset1/text.data.jsonl/text.data.jsonl", 'r') as file:
        content = list(file)[1000:2000]
        for sentence in content:
            try:
                s = json.loads(sentence)
                contents.append(s["casebody"]["data"]["opinions"][0]["text"])
            except:
                pass
    results = get_embeddings(contents, "Judgement embeddings")
    qdrant_upload_points(qdrant_client, "law-collection", [models.PointStruct(id=idx, vector=results["embedding"][idx], payload={ "Case": contents[idx] }) for idx in range(len(contents))])
    inference = qdrant_inference(qdrant_client, "law-collection", "Robbery from old couple")
    for inf in inference:
        print(inf.payload, inf.score)

main()