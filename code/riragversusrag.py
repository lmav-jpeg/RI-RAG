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
import multiprocessing as mp

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

# ============================================================
# INDEPENDENT PROCESS BENCHMARK
# ============================================================

def run_rag_process(rag_type, result_queue):
    """
    Runs one complete RAG benchmark in an isolated process.

    Each process owns:
        - its own Python interpreter
        - its own SentenceTransformer model
        - its own Chroma client
        - its own Chroma collection
        - its own database connection

    This prevents memory allocations from one RAG system
    contaminating the measurement of the other system.
    """

    process = psutil.Process(os.getpid())

    def rss_bytes():
        return process.memory_info().rss

    def rss_mb():
        return rss_bytes() / (1024 * 1024)

    print(
        f"\n[{rag_type}] "
        f"Starting isolated process "
        f"(PID={os.getpid()})"
    )

    # ========================================================
    # 1. DATABASE
    # ========================================================

    db = DatabaseManager(DB_CONFIG)

    # ========================================================
    # 2. INITIALIZE SYSTEM
    # ========================================================

    if rag_type == "RI-RAG":

        collection_name = (
            f"ri_rag_eval_{os.getpid()}"
        )

        system = HybridRIRAGSystem(
            db,
            collection_name=collection_name,
        )

    elif rag_type == "STANDARD":

        collection_name = (
            f"standard_rag_eval_{os.getpid()}"
        )

        system = NormalRAGSystem(
            db_manager=db,
            collection_name=collection_name,
        )

    else:
        raise ValueError(
            f"Unknown RAG type: {rag_type}"
        )

    # ========================================================
    # 3. LOAD DATABASE DATA
    # ========================================================

    connection = db._get_connection()

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

    print(
        f"[{rag_type}] "
        f"Loaded {len(rows)} records."
    )

    # ========================================================
    # 4. INGESTION MEMORY BENCHMARK
    # ========================================================

    baseline_rss = rss_bytes()
    peak_rss = baseline_rss

    print(
        f"[{rag_type}] "
        f"Baseline RSS: "
        f"{baseline_rss / (1024 * 1024):.2f} MB"
    )

    # --------------------------------------------------------
    # RI-RAG
    # --------------------------------------------------------

    if rag_type == "RI-RAG":

        for row in rows:

            system.ingest_hybrid_data(
                file_id=row["file_id"],
                file_name=row["file_name"],
                semantic_summary=row["semantic_summary"],
            )

            current_rss = rss_bytes()

            if current_rss > peak_rss:
                peak_rss = current_rss

    # --------------------------------------------------------
    # STANDARD RAG
    # --------------------------------------------------------

    else:

        for row in rows:

            system.ingest_document(
                file_id=row["file_id"],
                file_name=row["file_name"],
                content=row["content"],
                semantic_summary=row["semantic_summary"],
                grade=row["grade"],
                comments=row["comments"],
            )

            current_rss = rss_bytes()

            if current_rss > peak_rss:
                peak_rss = current_rss

    final_rss = rss_bytes()

    # ========================================================
    # 5. MEMORY RESULTS
    # ========================================================

    peak_increase_mb = (
        peak_rss - baseline_rss
    ) / (1024 * 1024)

    final_increase_mb = (
        final_rss - baseline_rss
    ) / (1024 * 1024)

    print(
        f"[{rag_type}] "
        f"Peak RSS: "
        f"{peak_rss / (1024 * 1024):.2f} MB"
    )

    print(
        f"[{rag_type}] "
        f"Peak increase: "
        f"{peak_increase_mb:.2f} MB"
    )

    # ========================================================
    # 6. RETRIEVAL TEST
    # ========================================================

    print(
        f"[{rag_type}] "
        f"Running retrieval queries..."
    )

    retrieval_results = []

    for i, query in enumerate(
        TEST_QUERIES,
        start=1,
    ):

        start = time.perf_counter()

        results = system.retrieve(
            query,
            n_results=1,
        )

        elapsed_ms = (
            time.perf_counter()
            - start
        ) * 1000

        if results:

            if rag_type == "RI-RAG":

                # RI-RAG returns:
                # [
                #   [
                #       record
                #   ]
                # ]

                if (
                    isinstance(results, list)
                    and results
                    and isinstance(results[0], list)
                ):
                    result = (
                        results[0][0]
                        if results[0]
                        else None
                    )
                else:
                    result = None

            else:

                # Standard RAG returns:
                # [
                #   record
                # ]

                result = results[0]

        else:
            result = None

        retrieval_results.append(
            {
                "query": query,
                "latency_ms": elapsed_ms,
                "result": result,
            }
        )

        print(
            f"[{rag_type}] "
            f"Query {i}: "
            f"{elapsed_ms:.2f} ms"
        )

    # ========================================================
    # 7. BATCH RETRIEVAL
    # ========================================================

    batch_latency_ms = None

    if rag_type == "RI-RAG":

        print(
            f"[{rag_type}] "
            f"Running batched retrieval..."
        )

        start = time.perf_counter()

        batch_results = system.retrieve(
            TEST_QUERIES,
            n_results=1,
        )

        batch_latency_ms = (
            time.perf_counter()
            - start
        ) * 1000

        print(
            f"[{rag_type}] "
            f"Batch latency for "
            f"{len(TEST_QUERIES)} queries: "
            f"{batch_latency_ms:.2f} ms"
        )

    # ========================================================
    # 8. MATERIALIZED CHROMA REPRESENTATION
    # ========================================================

    print(
        f"[{rag_type}] "
        f"Measuring Chroma representation..."
    )

    vector_store_size = (
        measure_vector_store_size(
            system.collection
        )
    )

    print(
        f"[{rag_type}] "
        f"Materialized Chroma representation: "
        f"{vector_store_size:,} bytes"
    )

    # ========================================================
    # 9. REPEATED RETRIEVAL BENCHMARK
    # ========================================================

    print(
        f"[{rag_type}] "
        f"Running repeated retrieval benchmark..."
    )

    latencies = benchmark_latency(
        system,
        TEST_QUERIES,
        repetitions=10,
    )

    statistics = calculate_statistics(
        latencies
    )

    print(
        f"[{rag_type}] "
        f"Mean latency: "
        f"{statistics['mean']:.2f} ms"
    )

    # ========================================================
    # 10. RETURN RESULTS
    # ========================================================

    result_queue.put(
        {
            "rag_type": rag_type,

            "pid": os.getpid(),

            "records": len(rows),

            "baseline_rss_mb": (
                baseline_rss
                / (1024 * 1024)
            ),

            "peak_rss_mb": (
                peak_rss
                / (1024 * 1024)
            ),

            "final_rss_mb": (
                final_rss
                / (1024 * 1024)
            ),

            "peak_increase_mb": (
                peak_increase_mb
            ),

            "final_increase_mb": (
                final_increase_mb
            ),

            "vector_store_size_bytes": (
                vector_store_size
            ),

            "retrieval_results": (
                retrieval_results
            ),

            "batch_latency_ms": (
                batch_latency_ms
            ),

            "latency_statistics": (
                statistics
            ),
        }
    )

    print(
        f"[{rag_type}] "
        f"Benchmark complete."
    )


# ============================================================
# RUN ONE PROCESS AT A TIME
# ============================================================

def run_isolated_benchmark(rag_type):
    """
    Start one RAG benchmark in a completely separate process.

    The process is terminated after the benchmark, ensuring
    the next benchmark starts with a clean Python process.
    """

    result_queue = mp.Queue()

    process = mp.Process(
        target=run_rag_process,
        args=(
            rag_type,
            result_queue,
        ),
    )

    process.start()

    process.join()

    if process.exitcode != 0:

        raise RuntimeError(
            f"{rag_type} benchmark process "
            f"failed with exit code "
            f"{process.exitcode}"
        )

    if result_queue.empty():

        raise RuntimeError(
            f"{rag_type} benchmark "
            f"returned no results."
        )

    return result_queue.get()


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    # Required on Windows.
    mp.freeze_support()

    print("=" * 75)
    print("RI-RAG vs STANDARD RAG")
    print("ISOLATED PROCESS BENCHMARK")
    print("=" * 75)

    print(
        "\nEach system will run in its own "
        "independent Python process."
    )

    # ========================================================
    # STANDARD RAG PROCESS
    # ========================================================

    print("\n" + "=" * 75)
    print("PROCESS 1: STANDARD RAG")
    print("=" * 75)

    standard_results = (
        run_isolated_benchmark(
            "STANDARD"
        )
    )

    # ========================================================
    # RI-RAG PROCESS
    # ========================================================

    print("\n" + "=" * 75)
    print("PROCESS 2: RI-RAG")
    print("=" * 75)

    rirag_results = (
        run_isolated_benchmark(
            "RI-RAG"
        )
    )

    # ========================================================
    # FINAL COMPARISON
    # ========================================================

    print("\n" + "=" * 75)
    print("FINAL COMPARISON")
    print("=" * 75)

    # --------------------------------------------------------
    # MEMORY
    # --------------------------------------------------------

    print("\nMEMORY")

    print(
        f"\nStandard RAG:"
        f"\n  Baseline RSS       : "
        f"{standard_results['baseline_rss_mb']:.2f} MB"
        f"\n  Peak RSS           : "
        f"{standard_results['peak_rss_mb']:.2f} MB"
        f"\n  Peak increase      : "
        f"{standard_results['peak_increase_mb']:.2f} MB"
    )

    print(
        f"\nRI-RAG:"
        f"\n  Baseline RSS       : "
        f"{rirag_results['baseline_rss_mb']:.2f} MB"
        f"\n  Peak RSS           : "
        f"{rirag_results['peak_rss_mb']:.2f} MB"
        f"\n  Peak increase      : "
        f"{rirag_results['peak_increase_mb']:.2f} MB"
    )

    standard_peak = (
        standard_results["peak_increase_mb"]
    )

    rirag_peak = (
        rirag_results["peak_increase_mb"]
    )

    if standard_peak > 0:

        memory_difference = (
            (
                standard_peak
                - rirag_peak
            )
            / standard_peak
        ) * 100

        print(
            f"\nRI-RAG peak RSS difference "
            f"relative to Standard RAG: "
            f"{memory_difference:.2f}%"
        )

    # --------------------------------------------------------
    # VECTOR REPRESENTATION
    # --------------------------------------------------------

    print("\n" + "-" * 75)
    print("MATERIALIZED CHROMA REPRESENTATION")
    print("-" * 75)

    standard_size = (
        standard_results[
            "vector_store_size_bytes"
        ]
    )

    rirag_size = (
        rirag_results[
            "vector_store_size_bytes"
        ]
    )

    print(
        f"Standard RAG : "
        f"{standard_size:,} bytes "
        f"({standard_size / 1024:.2f} KB)"
    )

    print(
        f"RI-RAG       : "
        f"{rirag_size:,} bytes "
        f"({rirag_size / 1024:.2f} KB)"
    )

    if standard_size > 0:

        vector_reduction = (
            (
                standard_size
                - rirag_size
            )
            / standard_size
        ) * 100

        print(
            f"\nRI-RAG materialized "
            f"representation difference: "
            f"{vector_reduction:.2f}%"
        )

    # --------------------------------------------------------
    # LATENCY
    # --------------------------------------------------------

    print("\n" + "-" * 75)
    print("REPEATED RETRIEVAL LATENCY")
    print("-" * 75)

    standard_stats = (
        standard_results[
            "latency_statistics"
        ]
    )

    rirag_stats = (
        rirag_results[
            "latency_statistics"
        ]
    )

    print(
        f"\nStandard RAG:"
        f"\n  Mean : "
        f"{standard_stats['mean']:.2f} ms"
        f"\n  Min  : "
        f"{standard_stats['min']:.2f} ms"
        f"\n  Max  : "
        f"{standard_stats['max']:.2f} ms"
    )

    print(
        f"\nRI-RAG:"
        f"\n  Mean : "
        f"{rirag_stats['mean']:.2f} ms"
        f"\n  Min  : "
        f"{rirag_stats['min']:.2f} ms"
        f"\n  Max  : "
        f"{rirag_stats['max']:.2f} ms"
    )

    # --------------------------------------------------------
    # BATCH
    # --------------------------------------------------------

    if (
        rirag_results["batch_latency_ms"]
        is not None
    ):

        print("\n" + "-" * 75)
        print("RI-RAG BATCH RETRIEVAL")
        print("-" * 75)

        print(
            f"Queries: "
            f"{len(TEST_QUERIES)}"
        )

        print(
            f"Total batch latency: "
            f"{rirag_results['batch_latency_ms']:.2f} ms"
        )

    # --------------------------------------------------------
    # RETRIEVAL RESULTS
    # --------------------------------------------------------

    print("\n" + "-" * 75)
    print("RETRIEVAL RESULTS")
    print("-" * 75)

    for i in range(
        len(TEST_QUERIES)
    ):

        query = TEST_QUERIES[i]

        standard_item = (
            standard_results[
                "retrieval_results"
            ][i]
        )

        rirag_item = (
            rirag_results[
                "retrieval_results"
            ][i]
        )

        print(
            f"\nQuery {i + 1}: "
            f"{query}"
        )

        print(
            f"  Standard RAG:"
            f" "
            f"{standard_item['latency_ms']:.2f} ms"
        )

        if standard_item["result"]:

            print(
                f"    ID: "
                f"{standard_item['result']['file_id']}"
            )

        else:

            print(
                "    No result."
            )

        print(
            f"  RI-RAG:"
            f" "
            f"{rirag_item['latency_ms']:.2f} ms"
        )

        if rirag_item["result"]:

            print(
                f"    ID: "
                f"{rirag_item['result']['file_id']}"
            )

        else:

            print(
                "    No result."
            )

    # ========================================================
    # INTERPRETATION
    # ========================================================

    print("\n" + "=" * 75)
    print("BENCHMARK INTERPRETATION")
    print("=" * 75)

    print(
        "\n1. Standard RAG stores full content "
        "inside Chroma."
    )

    print(
        "2. RI-RAG stores the semantic summary "
        "and file_id in Chroma."
    )

    print(
        "3. RI-RAG retrieves authoritative "
        "content and metadata from MySQL."
    )

    print(
        "4. Peak RSS is measured independently "
        "inside each process."
    )

    print(
        "5. Chroma representation size is "
        "measured while the collection actually "
        "contains the benchmark dataset."
    )

    print(
        "6. RI-RAG retrieval latency includes "
        "the MySQL hydration step."
    )

    print(
        "\nIMPORTANT: these measurements should "
        "be repeated across larger datasets "
        "before making scalability claims."
    )

    print("\n" + "=" * 75)
    print("BENCHMARK COMPLETE")
    print("=" * 75)