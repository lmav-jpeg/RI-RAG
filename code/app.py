"""
This class present the hybridization of the concept of RAG with the concept of Relational Database.
@conceptor: Laurie MAVOUNGOU, JK AI CEO, lmavoungou@outlook.be
@assistant: Gemini
"""
import os
import chromadb
from database import DatabaseManager
from sentence_transformers import SentenceTransformer

# Disable Hugging Face symlink warning on Windows
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"


class HybridRIRAGSystem:

  def __init__(
      self, db_manager: DatabaseManager, collection_name: str = "ri_rag_store"
  ):
    self.db = db_manager
    self.chroma_client = chromadb.Client()
    self.collection = self.chroma_client.get_or_create_collection(
        name=collection_name
    )
    self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

  def ingest_hybrid_data(
      self,
      file_id: str,
      file_name: str,
      content: str,
      semantic_summary: str,
      grade: float,
      comments: str,
  ):
    """Indexes the semantic entry point in ChromaDB,

    linking back to the authoritative MySQL record via file_id (PK).
    """
    text_to_embed = f"Meaning: {semantic_summary}"
    embedding = self.embedding_model.encode(text_to_embed).tolist()

    self.collection.upsert(
        ids=[file_id],  # Primary Key acts as Vector ID
        embeddings=[embedding],
        documents=[semantic_summary],
        metadatas=[{"file_name": file_name, "file_id": file_id}],
    )
    print(f"Vector Store: Indexed semantic entry point for [PK: {file_id}]")

  def retrieve(self, query_text: str, n_results: int = 5):
    """Scans vectors for similarity, extracts Primary Key (file_id),

    then queries the MySQL relational database for complete authoritative truth.
    """
    query_embedding = self.embedding_model.encode(query_text).tolist()
    vector_results = self.collection.query(
        query_embeddings=[query_embedding], n_results=n_results
    )

    resolved_records = []

    if vector_results and vector_results["ids"] and vector_results["ids"][0]:
      matched_ids = vector_results["ids"][0]
      distances = vector_results["distances"][0]

      for file_id, distance in zip(matched_ids, distances):
        # Follow Primary Key check to MySQL via DatabaseManager
        row = self.db.fetch_by_id(file_id)
        if row:
          row["vector_distance"] = distance
          resolved_records.append(row)

    return resolved_records


# --- Quick Test Execution ---
if __name__ == "__main__":
  print("Initializing Local Test Harness...")
  real_db_config = {
      "host": "localhost",
      "database": "RIRAG",  # Your actual database name
      "user": "root",
      "password": "qwerty",
      "port": 3306,
  }

  db = DatabaseManager(real_db_config)
  system = HybridRIRAGSystem(db)

  # Sync/Ingest the vector representations for records already in MySQL
  system.ingest_hybrid_data(
      file_id="DOC-001",
      file_name="consistency_guidelines.txt",
      content=(
          "The multi-round validation loop checks model responses against"
          " persistent evidence using database primary keys."
      ),
      semantic_summary=(
          "Defines multi-round contradiction detection and evidence grounding."
      ),
      grade=9.5,
      comments="Approved by lead engineer.",
  )

  system.ingest_hybrid_data(
      file_id="DOC-002",
      file_name="agent_safety.txt",
      content=(
          "Frontier autonomous agents must maintain structural consistency to"
          " prevent cascading decision failures."
      ),
      semantic_summary="Autonomous agent safety and consistency rules.",
      grade=8.8,
      comments="Needs minor revision on edge cases.",
  )

  # Run a semantic query test
  test_query = "How do we prevent agent failure and drift?"
  print(f"\nRunning Query: '{test_query}'")

  results = system.retrieve(test_query, n_results=1)

  print("\n--- Retrieval Results ---")
  for res in results:
    print(f"Matched Primary Key : {res['file_id']}")
    print(f"File Name           : {res['file_name']}")
    print(f"Relational Grade    : {res['grade']}")
    print(f"Relational Comments : {res['comments']}")
    print(f"Authoritative Text  : {res['content']}")
    print(f"Vector Distance     : {res['vector_distance']:.4f}")