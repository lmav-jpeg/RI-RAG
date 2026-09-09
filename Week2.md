# 🛠️ RI-RAG — Week 2: Relational-Indexed Retrieval-Augmented Generation Architecture

📌 **Objectives & Scope**  
Design, refine, and layout a high-performance, compact single-page technical specification document for the **Relational-Indexed Retrieval-Augmented Generation (RI-RAG)** architecture. The core goal was to establish a hybrid data storage and retrieval framework that decouples vector embedding lookups from bulky payload storage, routing queries through relational foreign keys to enforce strict access controls and minimize memory footprint.

🏗️ **What Progress We Made (What Was Done)**  
- **Architectural Specification & Schema Design:** Formulated the hybrid dual-database structure, pairing a vector index entry point with an authoritative MySQL relational table schema (`policies`).
- **Workflow & Pipeline Mapping:** Documented the multi-stage system workflow covering semantic entry-point ingestion, relational hydration, database sync sniffers, dynamic SQL ranking engines, and security metadata filtering.
- **Architecture Design Ready:**ri_rag_architecture_design.pdf