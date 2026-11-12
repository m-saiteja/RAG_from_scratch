import json
import pathlib
import requests
import numpy as np

# models used 
# LLM: llama3.2:3b
# Embedding Model: nomic-embed-text-v2-moe
EMBEDDING_MODEL = "nomic-embed-text-v2-moe"
LLM = "llama3.2:3b"
OLLAMA_URL = "http://localhost:11434"


def generate_embeddings(text):
    '''
    Generate embeddings for the given text using the specified embedding model.'''
    embedding_url = f"{OLLAMA_URL}/api/embed"
    json_payload = {
        "model": EMBEDDING_MODEL,
        "input": text
    }
    response = requests.post(embedding_url, json=json_payload, timeout=60)
    embeddings = np.array(response.json()["embeddings"])
    return embeddings


def generate_response(prompt):
    '''
    Generate a response to the given prompt using the specified LLM.
    This function does not stream the response and returns the full answer after generation.
    '''
    chat_url = f"{OLLAMA_URL}/api/chat"
    json_payload = {
        "model": LLM,
        "stream": False,
        "messages": [{"role": "user", "content": prompt}]
    }
    response = requests.post(chat_url, json=json_payload, timeout=60)
    generated_answer = response.json()["message"]["content"]
    return generated_answer


def generate_response_stream(prompt):
    '''
    Generate a response to the given prompt using the specified LLM.
    This function streams the response from the LLM and prints it in real-time.
    '''
    full_answer = ""
    with requests.post(
        f"{OLLAMA_URL}/api/chat",
        json={
            "model": LLM,
            "messages": [{"role": "user", "content": prompt}],
            "stream": True,
        },
        stream=True,
        timeout=300
    ) as response:
        response.raise_for_status()
        for line in response.iter_lines():
            if not line:
                continue
            chunk = json.loads(line)
            token = chunk["message"]["content"]
            print(token, end="", flush=True)
            full_answer += token
            if chunk.get("done"):
                break
    print()
    return full_answer


def retrive_relevant_chunks(query, chunks, chunk_vectors, k=3):
    '''
    Retrieve the top k relevant chunks based on cosine similarity between the query and chunk embeddings.
    '''
    query_embedding = generate_embeddings([query])[0]
    scores = chunk_vectors @ query_embedding / (np.linalg.norm(chunk_vectors, axis=1) * np.linalg.norm(query_embedding))
    top_indices = np.argsort(scores)[::-1][:k]
    return [chunks[i] for i in top_indices]


CHUNKS = []
for file in pathlib.Path("documents").glob('*.md'):
    for line in file.read_text(encoding="utf-8").split("\n\n"):
        if line.strip():
            CHUNKS.append(line.strip())

CHUNK_VECTORS = generate_embeddings(CHUNKS)


while True:
    user_query = input("Question: ")
    if user_query.strip().lower() in ["exit", "quit", "bye"]:
        break

    relevant_chunks = retrive_relevant_chunks(user_query, CHUNKS, CHUNK_VECTORS, k=3)

    # print("\nRelevant Context Chunks:")
    # for chunk in relevant_chunks:
    #     print(f"  - {chunk}")

    context = "\n\n".join(relevant_chunks)
    prompt = f"""
                Answer the question using only and only the context provided below.
                If the answer is NOT present in the context, say "I don't know".

                CONTEXT:
                {context}
                QUESTION: {user_query}
            """
    generate_response_stream(prompt)
    # answer = generate_response(prompt)
    # print(f"Answer: {answer}")
