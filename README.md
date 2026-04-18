# E-Commerce Microservices: Agentic RAG Chat Client

This repository contains the `chat-client` microservice for the `ecomm-micro` project. It provides an API-driven Agentic Retrieval-Augmented Generation (RAG) capabilities, built with FastAPI, LangGraph, and Google Gemini models.

## Overview

The Chat Client serves as an intelligent querying interface capable of searching retrieved context from pre-defined URLs, assessing document relevance, and rewriting queries if necessary. It leverages LangGraph to construct a stateful workflow that decides when to use search tools versus when to directly respond.

### Key Components

*   **FastAPI Backend (`server/app.py`)**: Exposes REST API endpoints (`/chat`, `/generate`, `/health`) to interact with the LLM flow.
*   **Agentic Workflow (`workflow.py`)**: Uses **LangGraph** to manage the cycle between assessing the need for retrieval, executing retrieval, grading the context, and generating a final answer.
*   **LLM Integration (`response_model.py`)**: Uses `gemini-2.5-flash` for reasoning, grading relevance, and rewriting questions.
*   **Vector Retreat (`retriever.py` & `ingestion.py`)**: Uses an `InMemoryVectorStore` coupled with `gemini-embedding-2-preview` embeddings to fetch query context from ingested web URLs.

## Agentic RAG Flow

The application follows an advanced RAG routing strategy:
1.  **Generate Query or Respond**: The model first evaluates whether to call the retrieval tool or respond directly based on the user's input.
2.  **Retrieve Context**: Using an initialized `retriever_tool`, it fetches document splits.
3.  **Grade Context**: A grader evaluates if the retrieved context is relevant.
    *   *If `yes`:* Flow passes to **Generate Answer**.
    *   *If `no`:* Flow passes to **Rewrite Question**, which edits the prompt and loops back to point #1.

## API Endpoints

The server runs by default on port `8086`.

*   **`GET /health`**
    Checks if the API is running.
*   **`POST /chat`**
    Main entry point for conversational agent queries.
    *Body*: `{ "message": "Your question here..." }`
*   **`POST /generate`**
    Endpoint for generating product descriptions.

## Setup and Installation

### Prerequisites

*   Python >= 3.14
*   A valid Google Gemini API Key

### Local Setup

Ensure that you have your environment variables set up properly, particularly for Google integration (e.g., `GOOGLE_API_KEY`). You can list source URLs to index in `config.py`.

Install the dependencies via pip:

```bash
pip install -r requirements.txt
```

Start the FastAPI server:

```bash
python server/app.py
```

### Docker Setup

A Dockerfile is included to containerize the service.

```bash
docker build -t ecomm-chat-client .
docker run -p 8086:8086 ecomm-chat-client
```

## Technologies Used

*   **Frameworks**: FastAPI, Uvicorn, LangChain, LangGraph
*   **Models**: Google Generative AI (`gemini-2.5-flash`, `gemini-embedding-2-preview`)
*   **Python Stack**: Pydantic, Beautiful Soup (for ingestion)
