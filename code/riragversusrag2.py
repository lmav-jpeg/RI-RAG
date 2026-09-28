"""
Evaluates Standard RAG against RI-RAG
using the same MySQL authoritative dataset.

RI-RAG:
    Chroma -> semantic_summary + file_id
    MySQL  -> authoritative content + metadata

Standard RAG:
    Chroma -> full content + metadata

@conceptor: Laurie MAVOUNGOU, JK AI CEO
@assistant: Gemini
"""

import os
import sys
import time

import chromadb
import psutil

from database import DatabaseManager
from rirag import HybridRIRAGSystem
from sentence_transformers import SentenceTransformer


# ============================================================
# CONFIGURATION
# ============================================================

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

DB_CONFIG = {
    "host": "localhost",
    "database": "RIRAGTEST",
    "user": "root",
    "password": "qwerty",
    "port": 3306,
}

EMBEDDING_MODEL = "all-MiniLM-L6-v2"

STANDARD_COLLECTION_NAME = "normal_rag_store"

TEST_QUERIES = [
    "How do we prevent agent failure and drift?",
    "How is Role-Based Access Control enforced at the database boundary?",
    "What RAM footprint reductions are achieved by storing only chunk IDs in vector indices?",
]


# ============================================================
# PROCESS MEMORY
# ============================================================

PROCESS = psutil.Process(os.getpid())


def get_rss_bytes():
    """Return current Python process RSS in bytes."""
    return PROCESS.memory_info().rss


def get_rss_mb():
    """Return current Python process RSS in MB."""
    return get_rss_bytes() / (1024 * 1024)


# ============================================================
# PYTHON OBJECT SIZE
# ============================================================

def get_deep_sizeof(obj):
    """
    Recursively calculates the size of a Python object.

    IMPORTANT:
    This measures the materialized Python representation
    returned by Chroma. It is NOT total physical RAM usage.
    """

    seen = set()

    def inner(o):

        object_id = id(o)

        if object_id in seen:
            return 0

        seen.add(object_id)

        size = sys.getsizeof(o)

        if isinstance(o, dict):

            size += sum(
                inner(k) + inner(v)
                for k, v in o.items()
            )

        elif isinstance(
            o,
            (list, tuple, set, frozenset)
        ):

            size += sum(
                inner(item)
                for item in o
            )

        return size

    return inner(obj)


# ============================================================
# STANDARD RAG
# ============================================================

class NormalRAGSystem:

    def __init__(
        self,
        db_manager: DatabaseManager,
        collection_name: str = STANDARD_COLLECTION_NAME,
    ):

        self.db = db_manager

        self.chroma_client = chromadb.Client()

        self.collection = (
            self.chroma_client.get_or_create_collection(
                name=collection_name
            )
        )

        self.embedding_model = SentenceTransformer(
            EMBEDDING_MODEL
        )

    # --------------------------------------------------------
    # DATABASE INGESTION
    # --------------------------------------------------------

    def ingest_from_database(self):

        connection = self.db._get_connection()

        cursor = connection.cursor(
            dictionary=True
        )

        cursor.execute(
            """
            SELECT
                file_id,
                file_name,
                content,
                semantic_summary,
                grade,
                comments
            FROM file
            """
        )

        rows = cursor.fetchall()

        cursor.close()
        connection.close()

        for row in rows:

            self.ingest_document(
                file_id=row["file_id"],
                file_name=row["file_name"],
                content=row["content"],
                semantic_summary=row["semantic_summary"],
                grade=row["grade"],
                comments=row["comments"],
            )

        print(
            f"[Standard RAG] "
            f"Indexed {len(rows)} records."
        )

    # --------------------------------------------------------
    # STANDARD RAG DOCUMENT
    # --------------------------------------------------------

    def ingest_document(
        self,
        file_id,
        file_name,
        content,
        semantic_summary,
        grade,
        comments,
    ):

        # Standard RAG embeds the complete information.
        text_to_embed = (
            f"Meaning: {semantic_summary}\n"
            f"Content: {content}"
        )

        embedding = (
            self.embedding_model
            .encode(text_to_embed)
            .tolist()
        )

        self.collection.upsert(
            ids=[file_id],

            embeddings=[embedding],

            # IMPORTANT:
            # Standard RAG stores full content.
            documents=[content],

            metadatas=[
                {
                    "file_name": file_name,
                    "semantic_summary": semantic_summary,
                    "grade": grade,
                    "comments": comments,
                    "file_id": file_id,
                }
            ],
        )

    # --------------------------------------------------------
    # RETRIEVAL
    # --------------------------------------------------------

    def retrieve(
        self,
        query_text,
        n_results=1,
    ):

        query_embedding = (
            self.embedding_model
            .encode(query_text)
            .tolist()
        )

        vector_results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
        )

        retrieved_records = []

        if (
            vector_results
            and vector_results["ids"]
            and vector_results["ids"][0]
        ):

            matched_ids = vector_results["ids"][0]

            documents = (
                vector_results["documents"][0]
            )

            metadatas = (
                vector_results["metadatas"][0]
            )

            distances = (
                vector_results["distances"][0]
            )

            for (
                file_id,
                document,
                metadata,
                distance,
            ) in zip(
                matched_ids,
                documents,
                metadatas,
                distances,
            ):

                retrieved_records.append(
                    {
                        "file_id": file_id,

                        "file_name": metadata.get(
                            "file_name"
                        ),

                        "content": document,

                        "semantic_summary": metadata.get(
                            "semantic_summary"
                        ),

                        "grade": metadata.get(
                            "grade"
                        ),

                        "comments": metadata.get(
                            "comments"
                        ),

                        "vector_distance": distance,
                    }
                )

        return retrieved_records


# ============================================================
# LATENCY MEASUREMENT
# ============================================================

def measure_retrieval(
    system,
    query,
    n_results=1,
):
    """
    Measures the complete retrieval operation.

    For RI-RAG this includes:

        embedding
        +
        Chroma search
        +
        MySQL hydration

    For Standard RAG this includes:

        embedding
        +
        Chroma search
    """

    start = time.perf_counter()

    results = system.retrieve(
        query,
        n_results=n_results,
    )

    elapsed_ms = (
        time.perf_counter() - start
    ) * 1000

    return results, elapsed_ms


# ============================================================
# MATERIALIZED VECTOR STORE SIZE
# ============================================================

def measure_vector_store_size(collection):

    data = collection.get(
        include=[
            "embeddings",
            "documents",
            "metadatas",
        ]
    )

    return get_deep_sizeof(data)


# ============================================================
# RETRIEVAL BENCHMARK
# ============================================================

def benchmark_latency(
    system,
    queries,
    repetitions=10,
):

    measurements = []

    # --------------------------------------------------------
    # Warm-up
    # --------------------------------------------------------

    for query in queries:

        system.retrieve(
            query,
            n_results=1,
        )

    # --------------------------------------------------------
    # Measurements
    # --------------------------------------------------------

    for _ in range(repetitions):

        for query in queries:

            start = time.perf_counter()

            system.retrieve(
                query,
                n_results=1,
            )

            elapsed_ms = (
                time.perf_counter() - start
            ) * 1000

            measurements.append(
                elapsed_ms
            )

    return measurements


def calculate_statistics(values):

    if not values:
        return {
            "mean": 0,
            "min": 0,
            "max": 0,
        }

    return {
        "mean": sum(values) / len(values),
        "min": min(values),
        "max": max(values),
    }


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("=" * 75)
    print("RI-RAG vs STANDARD RAG")
    print("=" * 75)

    # ========================================================
    # 1. DATABASE
    # ========================================================

    print("\n[1/6] Connecting to MySQL...")

    db = DatabaseManager(DB_CONFIG)

    # ========================================================
    # 2. INITIALIZE RI-RAG
    # ========================================================

    print("[2/6] Initializing RI-RAG...")

    rirag_system = HybridRIRAGSystem(
        db,
        collection_name="ri_rag_store",
    )

    # ========================================================
    # 3. INITIALIZE STANDARD RAG
    # ========================================================

    print("[3/6] Initializing Standard RAG...")

    normal_rag = NormalRAGSystem(
        db_manager=db,
        collection_name=STANDARD_COLLECTION_NAME,
    )

    # ========================================================
    # 4. INGESTION
    # ========================================================

    print("\n[4/6] Synchronizing both systems from MySQL...")

    # Each system is ingested exactly once.

    rss_before_rirag = get_rss_bytes()

    rirag_system.ingest_from_database()

    rss_after_rirag = get_rss_bytes()

    rss_before_standard = get_rss_bytes()

    normal_rag.ingest_from_database()

    rss_after_standard = get_rss_bytes()

    print(
        f"\nRI-RAG ingestion RSS delta: "
        f"{(rss_after_rirag - rss_before_rirag) / (1024 * 1024):.2f} MB"
    )

    print(
        f"Standard RAG ingestion RSS delta: "
        f"{(rss_after_standard - rss_before_standard) / (1024 * 1024):.2f} MB"
    )

    # ========================================================
    # 5. QUERY COMPARISON
    # ========================================================

    print("\n[5/6] Running retrieval comparison...")

    print("=" * 75)

    for i, query in enumerate(
        TEST_QUERIES,
        start=1,
    ):

        print(
            f"\nQuery {i}: {query}"
        )

        # ----------------------------------------------------
        # RI-RAG
        # ----------------------------------------------------

        rirag_results, rirag_latency = (
            measure_retrieval(
                rirag_system,
                query,
                n_results=1,
            )
        )

        print("\n--- RI-RAG ---")

        if rirag_results:

            result = rirag_results[0]

            print(
                f"Matched Primary Key : "
                f"{result['file_id']}"
            )

            print(
                f"File Name           : "
                f"{result['file_name']}"
            )

            print(
                f"MySQL Grade         : "
                f"{result['grade']}"
            )

            print(
                f"MySQL Comments      : "
                f"{result['comments']}"
            )

            print(
                f"Authoritative Text  : "
                f"{result['content']}"
            )

            print(
                f"Vector Distance     : "
                f"{result['vector_distance']:.4f}"
            )

            print(
                f"Total Retrieval     : "
                f"{rirag_latency:.2f} ms"
            )

        else:

            print("No RI-RAG result.")

        # ----------------------------------------------------
        # STANDARD RAG
        # ----------------------------------------------------

        normal_results, normal_latency = (
            measure_retrieval(
                normal_rag,
                query,
                n_results=1,
            )
        )

        print("\n--- STANDARD RAG ---")

        if normal_results:

            result = normal_results[0]

            print(
                f"Matched Document ID : "
                f"{result['file_id']}"
            )

            print(
                f"File Name           : "
                f"{result['file_name']}"
            )

            print(
                f"Vector Store Grade  : "
                f"{result['grade']}"
            )

            print(
                f"Vector Store Comments: "
                f"{result['comments']}"
            )

            print(
                f"Retrieved Text      : "
                f"{result['content']}"
            )

            print(
                f"Vector Distance     : "
                f"{result['vector_distance']:.4f}"
            )

            print(
                f"Total Retrieval     : "
                f"{normal_latency:.2f} ms"
            )

        else:

            print("No Standard RAG result.")

    # ========================================================
    # 6. VECTOR STORE SIZE
    # ========================================================

    print("\n" + "=" * 75)
    print("VECTOR STORE REPRESENTATION SIZE")
    print("=" * 75)

    normal_size = (
        measure_vector_store_size(
            normal_rag.collection
        )
    )

    rirag_size = (
        measure_vector_store_size(
            rirag_system.collection
        )
    )

    print(
        f"Standard RAG : "
        f"{normal_size:,} bytes "
        f"({normal_size / 1024:.2f} KB)"
    )

    print(
        f"RI-RAG      : "
        f"{rirag_size:,} bytes "
        f"({rirag_size / 1024:.2f} KB)"
    )

    if normal_size > 0:

        reduction = (
            (normal_size - rirag_size)
            / normal_size
        ) * 100

        print(
            f"\nRI-RAG materialized vector "
            f"representation reduction: "
            f"{reduction:.2f}%"
        )

    # ========================================================
    # REPEATED LATENCY
    # ========================================================

    print("\n" + "=" * 75)
    print("REPEATED QUERY PERFORMANCE")
    print("=" * 75)

    rirag_latencies = benchmark_latency(
        rirag_system,
        TEST_QUERIES,
        repetitions=10,
    )

    normal_latencies = benchmark_latency(
        normal_rag,
        TEST_QUERIES,
        repetitions=10,
    )

    rirag_stats = calculate_statistics(
        rirag_latencies
    )

    normal_stats = calculate_statistics(
        normal_latencies
    )

    print(
        f"Standard RAG mean: "
        f"{normal_stats['mean']:.2f} ms"
    )

    print(
        f"RI-RAG mean     : "
        f"{rirag_stats['mean']:.2f} ms"
    )

    print(
        f"\nStandard RAG min: "
        f"{normal_stats['min']:.2f} ms"
    )

    print(
        f"Standard RAG max: "
        f"{normal_stats['max']:.2f} ms"
    )

    print(
        f"RI-RAG min     : "
        f"{rirag_stats['min']:.2f} ms"
    )

    print(
        f"RI-RAG max     : "
        f"{rirag_stats['max']:.2f} ms"
    )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print("\n" + "=" * 75)
    print("FINAL SUMMARY")
    print("=" * 75)

    if normal_size > 0:

        reduction = (
            (normal_size - rirag_size)
            / normal_size
        ) * 100

        print(
            f"Vector representation reduction: "
            f"{reduction:.2f}%"
        )

    print(
        f"Standard RAG average latency: "
        f"{normal_stats['mean']:.2f} ms"
    )

    print(
        f"RI-RAG average latency: "
        f"{rirag_stats['mean']:.2f} ms"
    )

    print("\nInterpretation:")

    print(
        "1. Standard RAG stores the full payload "
        "inside Chroma."
    )

    print(
        "2. RI-RAG stores only the semantic summary "
        "and primary-key relationship in Chroma."
    )

    print(
        "3. RI-RAG resolves the primary key through "
        "MySQL to obtain authoritative content."
    )

    print(
        "4. The vector-size comparison measures the "
        "materialized representation returned by Chroma."
    )

    print(
        "5. The RI-RAG retrieval latency includes "
        "the relational hydration step."
    )

    print("\n" + "=" * 75)
    print("BENCHMARK COMPLETE")
    print("=" * 75)