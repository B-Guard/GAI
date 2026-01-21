import os
from dotenv import load_dotenv
from neo4j import GraphDatabase
from neo4j_graphrag.embeddings import SentenceTransformerEmbeddings

load_dotenv()

URI = os.getenv("NEO4J_URI")
USER = os.getenv("NEO4J_USER")
PWD = os.getenv("NEO4J_PASSWORD")
INDEX = os.getenv("NEO4J_VECTOR_INDEX", "chunk_embedding_idx")
EMB_MODEL = os.getenv("EMBEDDING_MODEL_NAME")

driver = GraphDatabase.driver(URI, auth=(USER, PWD))
embedder = SentenceTransformerEmbeddings(model=EMB_MODEL)

with driver.session() as s:
    rec = s.run("""
        MATCH (c:Chunk)
        RETURN elementId(c) AS id, c.text AS text
        LIMIT 1
    """).single()

chunk_id = rec["id"]
chunk_text = rec["text"]

qvec = embedder.embed_query(chunk_text)

with driver.session() as s:
    rec2 = s.run("""
        CALL db.index.vector.queryNodes($index, 5, $vec)
        YIELD node, score
        RETURN elementId(node) AS id, score
        ORDER BY score DESC
        LIMIT 5
    """, index=INDEX, vec=qvec).data()

print("Chunk picked:", chunk_id)
print("Top matches:")
for r in rec2:
    print(r["score"], r["id"], "<-- SAME" if r["id"] == chunk_id else "")
