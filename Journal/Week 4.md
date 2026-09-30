# RI-RAG — Research Progress

## Completed

The RI-RAG prototype and second experimental phase have been completed.

### Architecture

* Implemented **RI-RAG (Relational-Indexed Retrieval-Augmented Generation)**.
* ChromaDB is used for semantic retrieval.
* MySQL stores authoritative record content and relational metadata.
* Implemented vector retrieval → record ID → MySQL hydration.
* Implemented batched query retrieval.
* Implemented a Conventional RAG baseline for comparison.
* Used `all-MiniLM-L6-v2` embeddings (384 dimensions).

### Current Dataset

* 13 records (`DOC-001`–`DOC-013`).
* Synthetic enterprise/security-oriented data.
* 10 ground-truth queries used for retrieval evaluation.

## Preliminary Results

| Metric                                              |     Result |
| --------------------------------------------------- | ---------: |
| Vector-store representation reduction               |  **24.7%** |
| Python-process peak RSS increase reduction          |  **31.0%** |
| Combined Python + MySQL peak RSS increase reduction | **12.65%** |
| Top-1 retrieval accuracy                            |    **90%** |
| Ground-truth queries                                |     **10** |

### Latency

Preliminary repeated measurements:

* Conventional RAG: **135.81 ms mean**
* RI-RAG: **216.32 ms mean**
* RI-RAG 3-query batch: **435.73 ms**

The latency results are considered preliminary and require more controlled repeated trials.

## Validation Completed

* Vector-store representation comparison
* Python memory measurement
* Combined Python + MySQL memory measurement
* Retrieval accuracy evaluation
* Evidence grounding through MySQL
* Database boundary verification
* Direct retrieval consistency checks between Conventional RAG and RI-RAG

## Current Status

**Prototype:** ✅ Complete
**Preliminary evaluation:** ✅ Complete
**Second experimental phase:** ✅ Complete
**Large-scale evaluation:** ⏳ Pending funding

## Next Phase

The next phase will scale the experiments to larger datasets and more controlled repeated trials.

Planned targets:

* **10,000 records**
* **100,000 records**
* Larger ground-truth query sets
* Repeated memory and latency measurements
* Controlled database/cache conditions
* Content vs. semantic-summary validation
* Scalability analysis

The current research is paused while approximately **$300 in funding** is sought to support dataset generation/validation and AI/LLM usage.

> **Current research question:** Do the efficiency characteristics observed in the 13-record prototype persist as RI-RAG scales to substantially larger datasets?
