# Relational-Indexed Retrieval-Augmented Generation (RI-RAG)

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Domain: Cybersecurity & AI Systems](https://img.shields.io/badge/Domain-Cybersecurity%20%26%20AI-red.svg)]()
[![Status: Core Architecture & Active Development](https://img.shields.io/badge/Status-Active%20Development-green.svg)]()

## 📌 Overview

I built a maintainable hybrid architecture that pairs standard vector databases with traditional relational storage. The core design shifts heavy document payloads out of memory into relational storage, keeping only lightweight semantic pointers in the vector database. Instead of burning RAM on stored text, the CPU manages just-in-time data transfers between the relational layer and the AI model.

Research requires validation, not just invention. I will perform a controlled benchmark using the same dataset, embedding model, vector database, hardware, and workload for both conventional RAG and RI-RAG. I will measure peak RAM consumption, average and p95 latency, throughput, CPU utilization, and retrieval/answer accuracy.

Why does this matter? RI-RAG makes high-level RAG projects scalable using fewer hardware resources. Organizations with tight budgets and massive datasets should not abandon their vision due to infrastructure costs. RI-RAG delivers a practical path forward.
---
### ⭐ AI-Assisted Development Statement
All code in this repository was developed with AI assistance (leveraging models like Gemini and GPT). The system architecture, experimental design, prompt formulation, code integration, testing, result interpretation, and limitation analyses were entirely driven, directed, and validated by the author.