# Relational-Indexed Retrieval-Augmented Generation (RI-RAG)

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Domain: Cybersecurity & AI Systems](https://img.shields.io/badge/Domain-Cybersecurity%20%26%20AI-red.svg)]()
[![Status: Core Architecture & Active Development](https://img.shields.io/badge/Status-Active%20Development-green.svg)]()

## 📌 Architectural Overview

**Relational-Indexed Retrieval-Augmented Generation (RI-RAG)** is a hybrid storage and retrieval paradigm engineered to solve the memory scaling limits and lack of deterministic grounding in standard vector-only RAG pipelines.

While standard vector databases store both high-dimensional embeddings and heavy raw text payloads directly within volatile memory (RAM), RI-RAG decouples structural payload storage from similarity indexing. By enforcing **deterministic foreign key relationships** between lightweight vector candidate keys and disk-backed relational database management systems (RDBMS), RI-RAG delivers enterprise-grade auditability, strict access control, and sub-linear memory scaling.
