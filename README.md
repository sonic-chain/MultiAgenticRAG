## MultiAgentic RAG

This repository showcases the implementation of a **Multi-Agent Research RAG (Retriever-Augmented Generation) Tool** built with **LangGraph**. This project leverages the capabilities of agent-based frameworks to handle complex queries by breaking them down into manageable steps, dynamically utilizing tools, and ensuring response accuracy through error correction and hallucination checks.

## Getting Started

To get started with this project, follow these steps:

### 1. Clone the Repository

First, clone the repository to your local machine:

```bash
git clone https://github.com/nicoladisabato/MultiAgenticRAG.git
cd MultiAgenticRAG
```

Then open the config.yml file located in the root directory of the project. Set the value of load_documents to **true** to ensure the necessary documents are loaded into the vector database:

Then run:

```bash
python3 -m retriever.retriever
```

Once the PDF has been processed and indexed, you can start the application by running the following command:

```bash
python3 app.py
```

Now ask your question!

# Multi-Agent RAG Research with LangGraph

### Naive vs. Agentic RAG

While traditional Naive RAG systems struggle with:

- **Limited query understanding**: No support for multi-step query resolution.
- **Error handling**: Inability to verify and correct generated responses.
- **Dynamic tool integration**: Lack of adaptability to external APIs or databases.

Agentic RAG systems address these challenges by:

- **Dynamic routing and tool usage**: Classify and route user queries to appropriate workflows.
- **Multi-step planning**: Decompose complex queries into actionable research steps.
- **Self-correction with human-in-the-loop**: Introduce mechanisms to identify and resolve hallucinations.
- **Shared global state**: Maintain consistency across the workflow.

## Project Overview

This project integrates **LangGraph** with advanced retrieval and reasoning capabilities. The workflow is organized into modular graph-based steps:

1. **Analyze and Route Query**:
   - Classify the query and route it for further processing or response generation.

2. **Research Plan Generation**:
   - Generate a structured research plan with actionable steps for answering complex queries.

3. **Research Subgraph Execution**:
   - For each step:
     - Generate search queries.
     - Retrieve relevant documents using an **Ensemble Retriever** (hybrid search combining BM25, similarity search, and MMR).
     - Re-rank results using **Cohere's Contextual Compression Retriever**.

4. **Generate Response**:
   - Formulate a comprehensive answer from retrieved documents.

5. **Hallucination Check**:
   - Verify the response’s grounding in retrieved facts. If inconsistencies are detected, engage a human-in-the-loop mechanism.

6. **Human Approval**:
   - Allow users to validate or regenerate responses when needed.

### Key Components

- **Vector Store**:
  - Built using **ChromaDB**, with documents chunked by paragraph using Docling and LangChain.
  - Persistent storage ensures efficient search and retrieval.

- **Main Graph and Subgraphs**:
  - Main graph oversees query analysis, plan creation, and response generation.
  - Subgraphs handle query generation, retrieval, and ranking.


## Technical Highlights

1. **Document Parsing**:
   - Utilized **Docling** for robust extraction from complex PDF structures, converting content to Markdown for chunking.

2. **Advanced Retrieval**:
   - Implemented an **Ensemble Retriever** and **Cohere Reranker** to ensure accurate and relevant document selection.

3. **LangGraph State Management**:
   - Leveraged shared graph states for seamless transition and consistency across multi-step workflows.

## Conclusion

Agentic RAG systems redefine how we approach complex information retrieval and reasoning:

- **Scalability**: Handles intricate queries requiring multiple steps and diverse data sources.
- **Accuracy**: Minimizes hallucinations through verification and self-correction.
- **Adaptability**: Dynamically integrates tools and human feedback.

By combining advanced retrieval, reasoning, and verification mechanisms, this project sets a new standard for intelligent, reliable AI systems.



