import os
import sys
from typing import Any, Dict

import uvicorn
from fastapi import FastAPI, Response, status
from models import ChatRequest, ChatResponse

# Add the parent directory to sys.path to resolve root-level imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import set_env
from workflow import workflow

app = FastAPI(debug=False)
graph = None


@app.on_event("startup")
def startup_event():
    global graph
    graph = workflow()


@app.get("/health", status_code=200)
def health(response: Response) -> Dict:
    """Health endpoint."""

    response.status_code = status.HTTP_200_OK
    return {"message": "healthy"}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> Dict[str, Any]:
    """Endpoint to interact with the LangGraph agentic RAG."""
    if graph is None:
        response = Response(status_code=500, content="Graph not initialized")
        return response

    inputs = {"messages": [("user", request.message)]}
    result = graph.invoke(inputs)

    # Extract the final answer from the graph
    final_message = result["messages"][-1].content
    return {"response": final_message}


@app.post("/generate", response_model=ChatResponse)
def generate(request: ChatRequest) -> Dict[str, Any]:
    """Endpoint for generating product descriptions"""

    return {"response": "product is ass"}


def main():
    uvicorn.run(app, host="0.0.0.0", port=8086)


if __name__ == "__main__":
    main()
