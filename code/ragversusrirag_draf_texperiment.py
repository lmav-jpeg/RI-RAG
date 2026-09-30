"""
@author: Laurie MAVOUNGOU, JK AI CEO, lmavoungou@outlook.be
@assistant: Gemini
"""

import math
import statistics
import time
import os
import psutil

from database import DatabaseManager
from rirag import HybridRIRAGSystem
from riragversusrag import NormalRAGSystem


def calculate_p95(data):
    """Calculates the 95th percentile without external dependencies like numpy."""
    if not data:
        return 0.0

    sorted_data = sorted(data)
    index = int(math.ceil(0.95 * len(sorted_data))) - 1

    return sorted_data[max(0, index)]


def get_process_memory_mb(process):
    """
    Returns the current Resident Set Size (RSS) of the process in MB.

    RSS represents the physical memory currently occupied by the
    Python process, including memory allocated by native libraries.
    """
    return process.memory_info().rss / (1024 * 1024)


def benchmark_system(
    name,
    system,
    test_queries,
    iterations=50,
    is_normal_rag=False
):
    """
    Benchmarks a RAG system across:

    - Baseline RSS memory
    - Peak RSS memory
    - Peak RSS increase
    - Retrieval latency
    - Retrieval accuracy
    """

    print(f"\n--- Benchmarking: {name} ---")

    # ------------------------------------------------------------
    # 1. Initialize process-level memory measurement
    # ------------------------------------------------------------

    process = psutil.Process(os.getpid())

    # Memory immediately before benchmark execution.
    baseline_memory_mb = get_process_memory_mb(process)

    # Initialize peak memory with the baseline.
    peak_memory_mb = baseline_memory_mb

    latencies = []
    successful_retrievals = 0

    total_queries = len(test_queries) * iterations

    # ------------------------------------------------------------
    # 2. Execute retrieval benchmark
    # ------------------------------------------------------------

    for _ in range(iterations):

        for query_item in test_queries:

            query_text = query_item["query"]
            expected_id = query_item["expected_id"]

            # ----------------------------------------------------
            # Measure retrieval latency
            # ----------------------------------------------------

            start_time = time.perf_counter()

            # Handle return differences:
            #
            # NormalRAGSystem:
            #     returns (records, latency)
            #
            # HybridRIRAGSystem:
            #     returns records
            #
            if is_normal_rag:

                results, _ = system.retrieve(
                    query_text,
                    n_results=1
                )

            else:

                results = system.retrieve(
                    query_text,
                    n_results=1
                )

            elapsed_ms = (
                time.perf_counter() - start_time
            ) * 1000

            latencies.append(elapsed_ms)

            # ----------------------------------------------------
            # Measure current RSS after retrieval
            # ----------------------------------------------------

            current_memory_mb = get_process_memory_mb(process)

            if current_memory_mb > peak_memory_mb:
                peak_memory_mb = current_memory_mb

            # ----------------------------------------------------
            # Evaluate retrieval accuracy
            # ----------------------------------------------------

            if results and len(results) > 0:

                top_result = results[0]

                # Extract file_id safely across different
                # possible return structures.

                if isinstance(top_result, dict):

                    result_id = (
                        top_result.get("file_id")
                        or top_result.get("id")
                    )

                elif hasattr(top_result, "file_id"):

                    result_id = top_result.file_id

                elif hasattr(top_result, "id"):

                    result_id = top_result.id

                else:

                    result_id = top_result[0]

                if result_id == expected_id:

                    successful_retrievals += 1

    # ------------------------------------------------------------
    # 3. Calculate memory statistics
    # ------------------------------------------------------------

    peak_memory_delta_mb = (
        peak_memory_mb - baseline_memory_mb
    )

    # ------------------------------------------------------------
    # 4. Statistical calculations for latency
    # ------------------------------------------------------------

    mean_latency = (
        statistics.mean(latencies)
        if latencies
        else 0.0
    )

    std_latency = (
        statistics.stdev(latencies)
        if len(latencies) > 1
        else 0.0
    )

    p95_latency = calculate_p95(latencies)

    # 95% confidence interval for the mean latency.

    margin_of_error = (
        1.96 *
        (std_latency / math.sqrt(len(latencies)))
        if len(latencies) > 0
        else 0.0
    )

    ci_lower = mean_latency - margin_of_error
    ci_upper = mean_latency + margin_of_error

    # ------------------------------------------------------------
    # 5. Retrieval accuracy
    # ------------------------------------------------------------

    accuracy = (
        (successful_retrievals / total_queries) * 100.0
        if total_queries > 0
        else 0.0
    )

    # ------------------------------------------------------------
    # 6. Store benchmark metrics
    # ------------------------------------------------------------

    metrics = {

        "baseline_memory_mb":
            baseline_memory_mb,

        "peak_memory_mb":
            peak_memory_mb,

        "peak_memory_delta_mb":
            peak_memory_delta_mb,

        "mean_latency":
            mean_latency,

        "std_latency":
            std_latency,

        "p95_latency":
            p95_latency,

        "ci_lower":
            ci_lower,

        "ci_upper":
            ci_upper,

        "accuracy":
            accuracy,
    }

    # ------------------------------------------------------------
    # 7. Print individual system report
    # ------------------------------------------------------------

    print(
        f"Iterations (Total Queries) : "
        f"{len(latencies)}"
    )

    print(
        f"Baseline RSS Memory       : "
        f"{baseline_memory_mb:.2f} MB"
    )

    print(
        f"Peak RSS Memory           : "
        f"{peak_memory_mb:.2f} MB"
    )

    print(
        f"Peak RSS Increase         : "
        f"{peak_memory_delta_mb:.2f} MB"
    )

    print(
        f"Average Latency           : "
        f"{mean_latency:.4f} ms"
    )

    print(
        f"Standard Deviation        : "
        f"{std_latency:.4f} ms"
    )

    print(
        f"p95 Latency               : "
        f"{p95_latency:.4f} ms"
    )

    print(
        f"95% Confidence Interval   : "
        f"[{ci_lower:.4f}, {ci_upper:.4f}] ms"
    )

    print(
        f"Retrieval Accuracy        : "
        f"{accuracy:.2f}%"
    )

    return metrics


def test_rag_comparison_suite():
    """Runs comparative evaluation suite between Conventional RAG and RI-RAG."""

    print(
        "\n[Test] Running Comparative RAG Benchmark Suite "
        "(Conventional vs RI-RAG)..."
    )

    # ------------------------------------------------------------
    # Database configuration
    # ------------------------------------------------------------

    real_db_config = {
        "host": "localhost",
        "database": "RIRAGSTRESS",
        "user": "root",
        "password": "qwerty",
        "port": 3306,
    }

    db = DatabaseManager(real_db_config)

    # ------------------------------------------------------------
    # 1. Initialize systems
    # ------------------------------------------------------------

    normal_rag = NormalRAGSystem(db)

    rirag_system = HybridRIRAGSystem(db)

    # ------------------------------------------------------------
    # 2. Ingest records into both environments
    # ------------------------------------------------------------

    print(
        "[2/3] Ingesting Records into Both Environments..."
    )

    rirag_system.ingest_from_database()
    normal_rag.ingest_from_database()

    # ------------------------------------------------------------
    # 3. Dynamically build test queries directly from the database records
    # ------------------------------------------------------------

    expected_ids = [
        "DOC-00030", "DOC-00005", "DOC-00336", "DOC-00335",
        "DOC-00001", "DOC-00002", "DOC-00003", "DOC-00010",
        "DOC-00027", "DOC-00329", "DOC-00330", "DOC-00331",
        "DOC-00333", "DOC-00340", "DOC-00347", "DOC-00348",
        "DOC-00352", "DOC-00354", "DOC-00358", "DOC-00363",
        "DOC-00369"
    ]

    conn = db._get_connection()
    cursor = conn.cursor(dictionary=True)

    test_queries = []
    for doc_id in expected_ids:
        # Assumes your MySQL table is named 'documents' with columns 'file_id' and 'content'
        cursor.execute("SELECT file_id, content FROM file WHERE file_id = %s", (doc_id,))
        row = cursor.fetchone()

        if row and row.get("content"):
            query_text = row["content"]
        else:
            query_text = f"Record {doc_id}"

        test_queries.append({
            "query": query_text,
            "expected_id": doc_id
        })

    cursor.close()
    conn.close()

    # ------------------------------------------------------------
    # 4. Run benchmarks
    # ------------------------------------------------------------

    conv_metrics = benchmark_system(
        "Conventional RAG",
        normal_rag,
        test_queries,
        iterations=25,
        is_normal_rag=True
    )

    ri_metrics = benchmark_system(
        "RI-RAG (Hybrid Architecture)",
        rirag_system,
        test_queries,
        iterations=25,
        is_normal_rag=False
    )

    # ------------------------------------------------------------
    # 5. Final comparative summary
    # ------------------------------------------------------------

    print("\n" + "=" * 90)

    print(
        f"{'Metric':<30} | "
        f"{'Conventional RAG':<20} | "
        f"{'RI-RAG (Hybrid)':<20}"
    )

    print("-" * 90)

    print(
        f"{'Baseline RSS (MB)':<30} | "
        f"{conv_metrics['baseline_memory_mb']:<20.2f} | "
        f"{ri_metrics['baseline_memory_mb']:<20.2f}"
    )

    print(
        f"{'Peak RSS (MB)':<30} | "
        f"{conv_metrics['peak_memory_mb']:<20.2f} | "
        f"{ri_metrics['peak_memory_mb']:<20.2f}"
    )

    print(
        f"{'Peak RSS Increase (MB)':<30} | "
        f"{conv_metrics['peak_memory_delta_mb']:<20.2f} | "
        f"{ri_metrics['peak_memory_delta_mb']:<20.2f}"
    )

    print(
        f"{'Average Latency (ms)':<30} | "
        f"{conv_metrics['mean_latency']:<20.4f} | "
        f"{ri_metrics['mean_latency']:<20.4f}"
    )

    print(
        f"{'p95 Latency (ms)':<30} | "
        f"{conv_metrics['p95_latency']:<20.4f} | "
        f"{ri_metrics['p95_latency']:<20.4f}"
    )

    print(
        f"{'Retrieval Accuracy (%)':<30} | "
        f"{conv_metrics['accuracy']:<20.2f} | "
        f"{ri_metrics['accuracy']:<20.2f}"
    )

    print(
        f"{'Storage Strategy':<30} | "
        f"{'Monolithic Vector':<20} | "
        f"{'Decoupled (MySQL+Chroma)':<20}"
    )

    print("=" * 90)


if __name__ == "__main__":
    test_rag_comparison_suite()