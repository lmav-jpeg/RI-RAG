"""Evaluates normal RAG against RI-RAG"""

import sys
from app import *
import time

# Disable Hugging Face symlink warning on Windows
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"


def get_deep_sizeof(obj):
  """Recursively calculates the memory footprint of Python objects in bytes."""
  seen = set()

  def inner(o):
    if id(o) in seen:
      return 0
    seen.add(id(o))
    size = sys.getsizeof(o)
    if isinstance(o, dict):
      size += sum(inner(k) + inner(v) for k, v in o.items())
    elif isinstance(o, (list, tuple, set, frozenset)):
      size += sum(inner(item) for item in o)
    return size

  return inner(obj)

# ==========================================
# 2. STANDARD RAG BASELINE SYSTEM
# ==========================================
class NormalRAGSystem:

  def __init__(self, collection_name: str = "normal_rag_store"):
    self.chroma_client = chromadb.Client()
    self.collection = self.chroma_client.get_or_create_collection(
        name=collection_name
    )
    self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

  def ingest_document(
      self,
      doc_id: str,
      file_name: str,
      content: str,
      semantic_summary: str,
      grade: float,
      comments: str,
  ):
    """Stores text, metadata, and embeddings entirely inside ChromaDB."""
    text_to_embed = f"Meaning: {semantic_summary}\nContent: {content}"
    embedding = self.embedding_model.encode(text_to_embed).tolist()

    self.collection.upsert(
        ids=[doc_id],
        embeddings=[embedding],
        documents=[content],
        metadatas=[{
            "file_name": file_name,
            "semantic_summary": semantic_summary,
            "grade": grade,
            "comments": comments,
        }],
    )

  def retrieve(self, query_text: str, n_results: int = 1):
    """Retrieves text and metadata directly from the vector store."""
    start_time = time.time()
    query_embedding = self.embedding_model.encode(query_text).tolist()
    vector_results = self.collection.query(
        query_embeddings=[query_embedding], n_results=n_results
    )
    latency = (time.time() - start_time) * 1000  # ms

    retrieved_records = []
    if vector_results and vector_results["ids"] and vector_results["ids"][0]:
      matched_ids = vector_results["ids"][0]
      documents = vector_results["documents"][0]
      metadatas = vector_results["metadatas"][0]
      distances = vector_results["distances"][0]

      for doc_id, doc, metadata, distance in zip(
          matched_ids, documents, metadatas, distances
      ):
        record = {
            "file_id": doc_id,
            "file_name": metadata.get("file_name"),
            "content": doc,
            "semantic_summary": metadata.get("semantic_summary"),
            "grade": metadata.get("grade"),
            "comments": metadata.get("comments"),
            "vector_distance": distance,
        }
        retrieved_records.append(record)

    return retrieved_records, latency


# ==========================================
# 3. SIDE-BY-SIDE BENCHMARK EXECUTION
# ==========================================
if __name__ == "__main__":
    # ==========================================
    # 2. STANDARD RAG BASELINE SYSTEM
    # ==========================================
    class NormalRAGSystem:

        def __init__(self, collection_name: str = "normal_rag_store"):
            self.chroma_client = chromadb.Client()
            self.collection = self.chroma_client.get_or_create_collection(
                name=collection_name
            )
            self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

        def ingest_document(
                self,
                doc_id: str,
                file_name: str,
                content: str,
                semantic_summary: str,
                grade: float,
                comments: str,
        ):
            # Standard RAG stores everything (heavy text + metadata) inside ChromaDB
            text_to_embed = f"Meaning: {semantic_summary}\nContent: {content}"
            embedding = self.embedding_model.encode(text_to_embed).tolist()

            self.collection.upsert(
                ids=[doc_id],
                embeddings=[embedding],
                documents=[content],
                metadatas=[{
                    "file_name": file_name,
                    "semantic_summary": semantic_summary,
                    "grade": grade,
                    "comments": comments,
                }],
            )

        def retrieve(self, query_text: str, n_results: int = 1):
            start_time = time.time()
            query_embedding = self.embedding_model.encode(query_text).tolist()
            vector_results = self.collection.query(
                query_embeddings=[query_embedding], n_results=n_results
            )
            latency = (time.time() - start_time) * 1000  # ms

            retrieved_records = []
            if vector_results and vector_results["ids"] and vector_results["ids"][0]:
                matched_ids = vector_results["ids"][0]
                documents = vector_results["documents"][0]
                metadatas = vector_results["metadatas"][0]
                distances = vector_results["distances"][0]

                for doc_id, doc, metadata, distance in zip(
                        matched_ids, documents, metadatas, distances
                ):
                    record = {
                        "file_id": doc_id,
                        "file_name": metadata.get("file_name"),
                        "content": doc,
                        "semantic_summary": metadata.get("semantic_summary"),
                        "grade": metadata.get("grade"),
                        "comments": metadata.get("comments"),
                        "vector_distance": distance,
                    }
                    retrieved_records.append(record)

            return retrieved_records, latency


# ==========================================
# 3. SIDE-BY-SIDE BENCHMARK EXECUTION
# ==========================================
if __name__ == "__main__":
    print("=" * 70)
    print("COMPARING RI-RAG vs. STANDARD RAG PERFORMANCE HARNESS")
    print("=" * 70)

    real_db_config = {
        "host": "localhost",
        "database": "RIRAGTEST",
        "user": "root",
        "password": "qwerty",
        "port": 3306,
    }

    print("[1/3] Initializing Systems & Connecting to MySQL...")
    db = DatabaseManager(real_db_config)
    rirag_system = HybridRIRAGSystem(db)

    print("[2/3] Ingesting Records into Both Environments...")
    rirag_system.ingest_from_database()  # Syncs all rows dynamically from MySQL
    normal_rag = NormalRAGSystem()

    # Test Data Records
    records = [
        {
            "file_id": "DOC-001",
            "file_name": "001_consistency_guidelines.txt",
            "content": (
                "The multi-round validation loop checks model responses against persistent evidence using database primary keys. "
                "The multi-round validation loop checks model responses against persistent evidence using database primary keys. "
                "The multi-round validation loop checks model responses against persistent evidence using database primary keys. "
                "The multi-round validation loop checks model responses against persistent evidence using database primary keys. "
                "The multi-round validation loop checks model responses against persistent evidence using database primary keys. "
                "The multi-round validation loop checks model responses against persistent evidence using database primary keys. "
                "The multi-round validation loop checks model responses against persistent evidence using database primary keys."
            ),
            "semantic_summary": "Defines multi-round contradiction detection and evidence grounding.",
            "grade": 8.0,
            "comments": "Approved by lead engineer.",
        },
        {
            "file_id": "DOC-002",
            "file_name": "002_agent_safety.txt",
            "content": (
                "Frontier autonomous agents must maintain structural consistency to prevent cascading decision failures. "
                "Frontier autonomous agents must maintain structural consistency to prevent cascading decision failures. "
                "Frontier autonomous agents must maintain structural consistency to prevent cascading decision failures. "
                "Frontier autonomous agents must maintain structural consistency to prevent cascading decision failures. "
                "Frontier autonomous agents must maintain structural consistency to prevent cascading decision failures. "
                "Frontier autonomous agents must maintain structural consistency to prevent cascading decision failures. "
                "Frontier autonomous agents must maintain structural consistency to prevent cascading decision failures. "
                "Frontier autonomous agents must maintain structural consistency to prevent cascading decision failures. "
                "Frontier autonomous agents must maintain structural consistency to prevent cascading decision failures. "
                "Frontier autonomous agents must maintain structural consistency to prevent cascading decision failures."
            ),
            "semantic_summary": "Autonomous agent safety and consistency rules.",
            "grade": 8.5,
            "comments": "Needs minor revision on edge cases.",
        },
        {
            "file_id": "DOC-003",
            "file_name": "003_rag_architecture.txt",
            "content": (
                "Relational-Indexed Retrieval-Augmented Generation decouples vector search from authoritative records using foreign keys. "
                "Relational-Indexed Retrieval-Augmented Generation decouples vector search from authoritative records using foreign keys. "
                "Relational-Indexed Retrieval-Augmented Generation decouples vector search from authoritative records using foreign keys. "
                "Relational-Indexed Retrieval-Augmented Generation decouples vector search from authoritative records using foreign keys. "
                "Relational-Indexed Retrieval-Augmented Generation decouples vector search from authoritative records using foreign keys. "
                "Relational-Indexed Retrieval-Augmented Generation decouples vector search from authoritative records using foreign keys. "
                "Relational-Indexed Retrieval-Augmented Generation decouples vector search from authoritative records using foreign keys. "
                "Relational-Indexed Retrieval-Augmented Generation decouples vector search from authoritative records using foreign keys."
            ),
            "semantic_summary": "Overview of RI-RAG architectural decoupling.",
            "grade": 9.8,
            "comments": "Production ready.",
        },
        {
            "file_id": "DOC-004",
            "file_name": "004_vector_optimization.txt",
            "content": (
                "By storing only chunk IDs and embeddings in vector indices, RAM footprint is significantly reduced. "
                "By storing only chunk IDs and embeddings in vector indices, RAM footprint is significantly reduced. "
                "By storing only chunk IDs and embeddings in vector indices, RAM footprint is significantly reduced. "
                "By storing only chunk IDs and embeddings in vector indices, RAM footprint is significantly reduced. "
                "By storing only chunk IDs and embeddings in vector indices, RAM footprint is significantly reduced. "
                "By storing only chunk IDs and embeddings in vector indices, RAM footprint is significantly reduced. "
                "By storing only chunk IDs and embeddings in vector indices, RAM footprint is significantly reduced. "
                "By storing only chunk IDs and embeddings in vector indices, RAM footprint is significantly reduced. "
                "By storing only chunk IDs and embeddings in vector indices, RAM footprint is significantly reduced."
            ),
            "semantic_summary": "Memory optimization strategies for vector databases.",
            "grade": 9.7,
            "comments": "Requires further RAM profiling.",
        },
        {
            "file_id": "DOC-005",
            "file_name": "005_rbac_security.txt",
            "content": (
                "Role-Based Access Control is enforced at the database boundary via a privileged service account. "
                "Role-Based Access Control is enforced at the database boundary via a privileged service account. "
                "Role-Based Access Control is enforced at the database boundary via a privileged service account. "
                "Role-Based Access Control is enforced at the database boundary via a privileged service account. "
                "Role-Based Access Control is enforced at the database boundary via a privileged service account. "
                "Role-Based Access Control is enforced at the database boundary via a privileged service account. "
                "Role-Based Access Control is enforced at the database boundary via a privileged service account. "
                "Role-Based Access Control is enforced at the database boundary via a privileged service account."
            ),
            "semantic_summary": "Database-level access control mechanisms.",
            "grade": 8.7,
            "comments": "Compliant with security standards.",
        },
        {
            "file_id": "DOC-006",
            "file_name": "006_audit_logging.txt",
            "content": (
                "All retrieval requests and relational hydration steps must generate immutable audit logs for traceability. "
                "All retrieval requests and relational hydration steps must generate immutable audit logs for traceability. "
                "All retrieval requests and relational hydration steps must generate immutable audit logs for traceability. "
                "All retrieval requests and relational hydration steps must generate immutable audit logs for traceability. "
                "All retrieval requests and relational hydration steps must generate immutable audit logs for traceability. "
                "All retrieval requests and relational hydration steps must generate immutable audit logs for traceability. "
                "All retrieval requests and relational hydration steps must generate immutable audit logs for traceability. "
                "All retrieval requests and relational hydration steps must generate immutable audit logs for traceability. "
                "All retrieval requests and relational hydration steps must generate immutable audit logs for traceability. "
                "All retrieval requests and relational hydration steps must generate immutable audit logs for traceability."
            ),
            "semantic_summary": "Audit trail requirements for AI retrieval pipelines.",
            "grade": 8.2,
            "comments": "Standardized across services.",
        },
        {
            "file_id": "DOC-007",
            "file_name": "007_adversarial_robustness.txt",
            "content": (
                "Neural networks must incorporate adversarial training loops to resist gradient-based input perturbations. "
                "Neural networks must incorporate adversarial training loops to resist gradient-based input perturbations. "
                "Neural networks must incorporate adversarial training loops to resist gradient-based input perturbations. "
                "Neural networks must incorporate adversarial training loops to resist gradient-based input perturbations. "
                "Neural networks must incorporate adversarial training loops to resist gradient-based input perturbations. "
                "Neural networks must incorporate adversarial training loops to resist gradient-based input perturbations. "
                "Neural networks must incorporate adversarial training loops to resist gradient-based input perturbations. "
                "Neural networks must incorporate adversarial training loops to resist gradient-based input perturbations. "
                "Neural networks must incorporate adversarial training loops to resist gradient-based input perturbations. "
                "Neural networks must incorporate adversarial training loops to resist gradient-based input perturbations. "
                "Neural networks must incorporate adversarial training loops to resist gradient-based input perturbations. "
                "Neural networks must incorporate adversarial training loops to resist gradient-based input perturbations. "
                "Neural networks must incorporate adversarial training loops to resist gradient-based input perturbations. "
                "Neural networks must incorporate adversarial training loops to resist gradient-based input perturbations."
            ),
            "semantic_summary": "Adversarial machine learning defense pipelines.",
            "grade": 8.7,
            "comments": "Under active evaluation.",
        },
        {
            "file_id": "DOC-008",
            "file_name": "008_federated_privacy.txt",
            "content": (
                "Federated learning enables decentralized model updates without exposing raw institutional data. "
                "Federated learning enables decentralized model updates without exposing raw institutional data. "
                "Federated learning enables decentralized model updates without exposing raw institutional data. "
                "Federated learning enables decentralized model updates without exposing raw institutional data. "
                "Federated learning enables decentralized model updates without exposing raw institutional data. "
                "Federated learning enables decentralized model updates without exposing raw institutional data. "
                "Federated learning enables decentralized model updates without exposing raw institutional data. "
                "Federated learning enables decentralized model updates without exposing raw institutional data. "
                "Federated learning enables decentralized model updates without exposing raw institutional data. "
                "Federated learning enables decentralized model updates without exposing raw institutional data."
            ),
            "semantic_summary": "Privacy-preserving model aggregation techniques.",
            "grade": 9.4,
            "comments": "Validated on medical datasets.",
        },
        {
            "file_id": "DOC-009",
            "file_name": "009_latency_benchmarks.txt",
            "content": (
                "Network round-trips to relational databases introduce a measurable latency trade-off compared to in-memory lookups. "
                "Network round-trips to relational databases introduce a measurable latency trade-off compared to in-memory lookups. "
                "Network round-trips to relational databases introduce a measurable latency trade-off compared to in-memory lookups. "
                "Network round-trips to relational databases introduce a measurable latency trade-off compared to in-memory lookups. "
                "Network round-trips to relational databases introduce a measurable latency trade-off compared to in-memory lookups. "
                "Network round-trips to relational databases introduce a measurable latency trade-off compared to in-memory lookups. "
                "Network round-trips to relational databases introduce a measurable latency trade-off compared to in-memory lookups. "
                "Network round-trips to relational databases introduce a measurable latency trade-off compared to in-memory lookups. "
                "Network round-trips to relational databases introduce a measurable latency trade-off compared to in-memory lookups. "
                "Network round-trips to relational databases introduce a measurable latency trade-off compared to in-memory lookups."
            ),
            "semantic_summary": "Query latency analysis for hybrid retrieval.",
            "grade": 9.3,
            "comments": "Optimizing connection pooling.",
        },
        {
            "file_id": "DOC-010",
            "file_name": "010_code_based_crypto.txt",
            "content": (
                "Quasi-cyclic error-correcting codes provide post-quantum cryptographic resilience for secure record storage. "
                "Quasi-cyclic error-correcting codes provide post-quantum cryptographic resilience for secure record storage. "
                "Quasi-cyclic error-correcting codes provide post-quantum cryptographic resilience for secure record storage. "
                "Quasi-cyclic error-correcting codes provide post-quantum cryptographic resilience for secure record storage. "
                "Quasi-cyclic error-correcting codes provide post-quantum cryptographic resilience for secure record storage. "
                "Quasi-cyclic error-correcting codes provide post-quantum cryptographic resilience for secure record storage. "
                "Quasi-cyclic error-correcting codes provide post-quantum cryptographic resilience for secure record storage. "
                "Quasi-cyclic error-correcting codes provide post-quantum cryptographic resilience for secure record storage. "
                "Quasi-cyclic error-correcting codes provide post-quantum cryptographic resilience for secure record storage. "
                "Quasi-cyclic error-correcting codes provide post-quantum cryptographic resilience for secure record storage."
            ),
            "semantic_summary": "Code-based cryptosystems and error correction.",
            "grade": 9.2,
            "comments": "Reviewed by crypto team.",
        },
        {
            "file_id": "DOC-011",
            "file_name": "011_edge_ai_integration.txt",
            "content": (
                "Edge AI devices require lightweight quantization to maintain real-time inference constraints. "
                "Edge AI devices require lightweight quantization to maintain real-time inference constraints. "
                "Edge AI devices require lightweight quantization to maintain real-time inference constraints. "
                "Edge AI devices require lightweight quantization to maintain real-time inference constraints. "
                "Edge AI devices require lightweight quantization to maintain real-time inference constraints. "
                "Edge AI devices require lightweight quantization to maintain real-time inference constraints. "
                "Edge AI devices require lightweight quantization to maintain real-time inference constraints."
            ),
            "semantic_summary": "Edge device quantization rules.",
            "grade": 9.1,
            "comments": "Approved for embedded deployment.",
        },
        {
            "file_id": "DOC-012",
            "file_name": "012_context_pruning.txt",
            "content": (
                "Context window pruning algorithms eliminate redundant tokens prior to relational hydration steps. "
                "Context window pruning algorithms eliminate redundant tokens prior to relational hydration steps. "
                "Context window pruning algorithms eliminate redundant tokens prior to relational hydration steps. "
                "Context window pruning algorithms eliminate redundant tokens prior to relational hydration steps. "
                "Context window pruning algorithms eliminate redundant tokens prior to relational hydration steps. "
                "Context window pruning algorithms eliminate redundant tokens prior to relational hydration steps."
            ),
            "semantic_summary": "Context pruning and token efficiency.",
            "grade": 8.9,
            "comments": "Needs benchmark validation.",
        },
        {
            "file_id": "DOC-013",
            "file_name": "013_distributed_caching.txt",
            "content": (
                "Distributed Redis caching layers minimize database round-trip latency for frequently accessed records. "
                "Distributed Redis caching layers minimize database round-trip latency for frequently accessed records. "
                "Distributed Redis caching layers minimize database round-trip latency for frequently accessed records. "
                "Distributed Redis caching layers minimize database round-trip latency for frequently accessed records. "
                "Distributed Redis caching layers minimize database round-trip latency for frequently accessed records. "
                "Distributed Redis caching layers minimize database round-trip latency for frequently accessed records."
            ),
            "semantic_summary": "Redis caching for hybrid retrieval pipelines.",
            "grade": 9.4,
            "comments": "Production ready.",
        },
    ]

    print("[2/3] Ingesting Records into Both Environments...")
    for rec in records:
        # Ingest into RI-RAG (MySQL + ChromaDB pointer)
        rirag_system.ingest_hybrid_data(
            rec["file_id"],
            rec["file_name"],
            rec["semantic_summary"],
        )
        # Ingest into Standard RAG (ChromaDB-only payload store)
        normal_rag.ingest_document(
            rec["file_id"],
            rec["file_name"],
            rec["content"],
            rec["semantic_summary"],
            rec["grade"],
            rec["comments"],
        )

    # Run Comparative Query
    test_queries = [
        "How do we prevent agent failure and drift?",
        "How is Role-Based Access Control enforced at the database boundary?",
        "What RAM footprint reductions are achieved by storing only chunk IDs in vector indices?"
    ]

    for i, test_query in enumerate(test_queries, 1):
        print(f"\n[{i}/{len(test_queries)}] Executing Query: '{test_query}'")
        # Insert your RI-RAG vector search & primary key hydration logic here

        # --- Test RI-RAG ---
        start_t = time.time()
        rirag_results = rirag_system.retrieve(test_query, n_results=1)
        rirag_latency = (time.time() - start_t) * 1000

        print("--- RI-RAG (Hybrid Architecture) Results ---")
        if rirag_results:
            res = rirag_results[0]
            print(f"Matched Primary Key   : {res['file_id']}")
            print(f"File Name             : {res['file_name']}")
            print(f"MySQL Relational Grade: {res['grade']}")
            print(f"MySQL Comments        : {res['comments']}")
            print(f"Authoritative Text    : {res['content']}")
            print(f"Total Execution Time  : {rirag_latency:.2f} ms")

        print("\n" + "-" * 50 + "\n")

        # --- Test Standard RAG ---
        normal_results, normal_latency = normal_rag.retrieve(test_query, n_results=1)

        print("--- Standard RAG Baseline Results ---")
        if normal_results:
            res = normal_results[0]
            print(f"Matched Document ID   : {res['file_id']}")
            print(f"File Name             : {res['file_name']}")
            print(f"Vector Store Grade    : {res['grade']}")
            print(f"Vector Store Comments : {res['comments']}")
            print(f"Retrieved Text        : {res['content']}")
            print(f"Total Execution Time  : {normal_latency:.2f} ms")

        # --- Memory Usage Evaluation ---
        print("\n" + "=" * 70)
        print("VECTOR STORE MEMORY FOOTPRINT BENCHMARK")
        print("=" * 70)

        normal_store_data = normal_rag.collection.get(
            include=["embeddings", "documents", "metadatas"]
        )
        rirag_store_data = rirag_system.collection.get(
            include=["embeddings", "documents", "metadatas"]
        )

        normal_memory_bytes = get_deep_sizeof(normal_store_data)
        rirag_memory_bytes = get_deep_sizeof(rirag_store_data)

        print(
            f"Standard RAG Vector Store Memory : {normal_memory_bytes:,} bytes"
            f" ({normal_memory_bytes / 1024:.2f} KB)"
        )
        print(
            f"RI-RAG Vector Store Memory       : {rirag_memory_bytes:,} bytes"
            f" ({rirag_memory_bytes / 1024:.2f} KB)"
        )

        if normal_memory_bytes > 0:
            memory_savings = (
                                     (normal_memory_bytes - rirag_memory_bytes) / normal_memory_bytes
                             ) * 100
            print(
                f"-> Vector Store Memory Saved by RI-RAG: {memory_savings:.1f}%"
                " reduction"
            )

        print("=" * 70)
        print("EVALUATION HARNESS EXECUTION COMPLETE.")
        print("=" * 70)