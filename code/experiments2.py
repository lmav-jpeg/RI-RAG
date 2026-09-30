"""
@author: Laurie MAVOUNGOU, JK AI CEO, lmavoungou@outlook.be
@assistant: Gemini
"""

import math
import statistics
import time
from database import DatabaseManager
from rirag import HybridRIRAGSystem  # Assuming your system class is imported here
from Experiment3 import NormalRAGSystem


def test_ri_rag_latency_statistics():
  """Test complete end-to-end RI-RAG pipeline statistics:

  - Average Latency
  - Standard Deviation
  - 95% Confidence Interval
  """
  print(
      "\n[Test] Running Complete End-to-End RI-RAG Statistical Latency"
      " Evaluation..."
  )

  real_db_config = {
      "host": "localhost",
      "database": "RIRAGTEST",
      "user": "test_user",
      "password": "",
      "port": 3306,
  }

  db = DatabaseManager(real_db_config)
  #system = HybridRIRAGSystem(db)
  system = NormalRAGSystem()

  iterations = 50
  latencies = []
  test_query = "How do we prevent agent failure and drift?"

  for _ in range(iterations):
    start_time = time.perf_counter()

    # Executes the complete hybrid round-trip:
    # 1. Query Embedding -> 2. ChromaDB Search (PK extraction) -> 3. MySQL Hydration
    _ = system.retrieve(test_query, n_results=1)

    elapsed_ms = (time.perf_counter() - start_time) * 1000
    latencies.append(elapsed_ms)

  mean_latency = statistics.mean(latencies)
  std_latency = statistics.stdev(latencies)

  margin_of_error = 1.96 * (std_latency / math.sqrt(iterations))
  ci_lower = mean_latency - margin_of_error
  ci_upper = mean_latency + margin_of_error

  print(f"Iterations             : {iterations}")
  print(f"Average Hybrid Latency : {mean_latency:.4f} ms")
  print(f"Standard Deviation     : {std_latency:.4f} ms")
  print(f"95% Confidence Interval: [{ci_lower:.4f}, {ci_upper:.4f}] ms")


if __name__ == "__main__":
  test_ri_rag_latency_statistics()