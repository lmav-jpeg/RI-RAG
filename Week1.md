# 🛠️ RI-RAG — Week 1: Data Engine & Ingestion Pipeline

Welcome to the foundational phase of **RI-RAG** (Relational-Indexed Retrieval-Augmented Generation). Week 1 focused on building the core relational data layer, designing the MySQL schema for high-volume network telemetry, and implementing a memory-bounded streaming ingestion pipeline.

---

## 📌 Objectives & Scope

* Establish the MySQL database foundation for storing raw and processed Cyber Threat Intelligence (CTI).
* Ingest high-volume **Zeek network connection logs (`conn.log`)** from the OTRF (Open Threat Research Framework) dataset.
* Ensure memory efficiency ($O(1)$ RAM usage) when handling multi-gigabyte log dumps.
* Implement high-throughput batching and duplicate collision handling for multi-day threat captures.

---

## 🏗️ System Architecture & Data Flow

```text
[ Raw OTRF Logs ] ---> [ Line Buffer Stream ] ---> [ Key Normalization & Cleaning ]
                                                               |
                                                               v
[ MySQL Database ] <--- [ 5,000-Row Chunk Commit ] <--- [ Dynamic Query Construction ]
