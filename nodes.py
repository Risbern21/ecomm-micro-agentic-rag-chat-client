from langchain.messages import HumanMessage
from langgraph.graph import MessagesState

from response_model import response_model
from tools import retriever_tool

GENERATE_PROMPT = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer the question. "
    "If you don't know the answer, just say that you don't know. "
    "Use three sentences maximum and keep the answer concise.\n"
    "Question: {question} \n"
    "Context: {context}"
)
REWRITE_PROMPT = (
    "Look at the input and try to reason about the underlying semantic intent / meaning\n"
    "Here is the initial question:"
    "\n ------- \n"
    "{question}"
    "\n ------- \n"
    "Formulate an improved question:"
)


def generate_query_or_respond(state: MessagesState):
    """Call the model to generate a response based on the current state. Given
    the question, it will decide to retrieve using the retriever tool, or simply respond to the user.
    """
    response = response_model.bind_tools([retriever_tool]).invoke(state["messages"])
    return {"messages": [response]}


def generate_answer(state: MessagesState):
    """Generate an answer."""

    question = state["messages"][0].content
    context = state["messages"][-1].content
    prompt = GENERATE_PROMPT.format(question=question, context=context)
    response = response_model.invoke([{"role": "user", "content": prompt}])
    return {"messages": [response]}


def rewrite_question(state: MessagesState):
    """Rewrite the original user question."""

    messages = state["messages"]
    question = messages[0].content
    prompt = REWRITE_PROMPT.format(question=question)
    response = response_model.invoke([{"role": "user", "content": prompt}])
    return {"messages": [HumanMessage(content=response.content)]}
