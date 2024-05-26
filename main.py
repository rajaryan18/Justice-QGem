from qdrant_client import QdrantClient, models
import google.generativeai as genai
import json
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
openai_client = OpenAI(os.getenv("OPENAI_API_KEY"))

def load_qdrant():
    return QdrantClient(
        url=os.getenv('QDRANT_URL'),
        api_key=os.getenv("QDRANT_CLUSTER_API_KEY")
    )

def create_qdrant_collection(qdrant_client, collection_name):
    qdrant_client.create_collection(
        collection_name=collection_name,
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
    try:
        create_qdrant_collection(qdrant_client, "law-collection")
    except:
        print("Collecion Already Exists")
    contents = []
    with open("./Data/dataset1/text.data.jsonl/text.data.jsonl", 'r') as file:
        content = list(file)
        for sentence in content:
            try:
                s = json.loads(sentence)
                contents.append(s["casebody"]["data"]["opinions"][0]["text"])
            except:
                pass
    results = get_embeddings(contents, "Judgement embeddings")
    qdrant_upload_points(qdrant_client, "law-collection", [models.PointStruct(id=idx, vector=results["embedding"][idx], payload={ "Case": contents[idx] }) for idx in range(len(contents))])
    case = input("Please enter the details of the case\n")
    inference = qdrant_inference(qdrant_client, "law-collection", case)
    response = openai_client.chat.completions.create(
        model="gpt-4",
        messages=[
            {
                "role": "system",
                "content": f"{inference[0].payload}. Using the above context, pass a suitable judgement for the given description of a case file. Here is the case file: {case}"
            }
        ],
    )
    print(response)

main()