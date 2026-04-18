from typing import Literal

from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import MessagesState
from pydantic import BaseModel, Field

GRADE_PROMPT = (
    "You are a grader assessing relevance of a retrieved document to a user question. \n "
    "Here is the retrieved document: \n\n {context} \n\n"
    "Here is the user question: {question} \n"
    "If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n"
    "Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."
)


class RelevenceGrader(BaseModel):
    """Grade documents using a binary score for relevance check."""

    binary_score: str = Field(
        description="Relevance score: 'yes' if relevant, or 'no' if not relevant"
    )


grader_model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
)


def grade_documents(
    state: MessagesState,
) -> Literal["generate_answer", "rewrite_question"]:
    """Determine whether the retrieved documents are relevant to the question."""
    question = state["messages"][0].content
    context = state["messages"][-1].content

    prompt = GRADE_PROMPT.format(question=question, context=context)
    response = grader_model.with_structured_output(RelevenceGrader).invoke(
        [{"role": "user", "content": prompt}]
    )
    score = response.binary_score

    if score == "yes":
        return "generate_answer"
    return "rewrite_question"
