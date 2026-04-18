from IPython.display import Image, display
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from grader import grade_documents
from nodes import generate_answer, generate_query_or_respond, rewrite_question
from tools import retriever_tool


def workflow() -> StateGraph:

    workflow = StateGraph(MessagesState)

    # Define the nodes we will cycle between
    workflow.add_node(generate_query_or_respond)
    workflow.add_node("retrieve", ToolNode([retriever_tool]))
    workflow.add_node(rewrite_question)
    workflow.add_node(generate_answer)

    workflow.add_edge(START, "generate_query_or_respond")

    # Decide whether to retrieve
    workflow.add_conditional_edges(
        "generate_query_or_respond",
        # Assess LLM decision (call `retriever_tool` tool or respond to the user)
        tools_condition,
        {
            # Translate the condition outputs to nodes in our graph
            "tools": "retrieve",
            END: END,
        },
    )

    # Edges taken after the `action` node is called.
    workflow.add_conditional_edges(
        "retrieve",
        # Assess agent decision
        grade_documents,
    )
    workflow.add_edge("generate_answer", END)
    workflow.add_edge("rewrite_question", "generate_query_or_respond")

    # Compile
    graph = workflow.compile()
    return graph


def main():
    graph = workflow()
    img = Image(graph.get_graph().draw_mermaid_png())

    with open("graph.png", "wb") as f:
        f.write(img.data)


if __name__ == "__main__":
    main()
