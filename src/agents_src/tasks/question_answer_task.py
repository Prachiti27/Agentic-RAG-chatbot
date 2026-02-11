from crewai import Task
from pydantic import BaseModel
from typing import List

from src.agents_src.agents.question_answer_agent import qa_agent


class AnswerStructure(BaseModel):
    answer: str
    sources: List[str]
    tool_used: str
    rationale: str


qa_task = Task(
    agent=qa_agent,
    name="Question Answering Task",
    description=(
        "Answer the user query '{user_query}' using a Retrieval-Augmented Generation (RAG) pipeline.\n\n"
        "Chat History: '{chat_history}'\n\n"
        "Instructions:\n"
        "- Retrieve relevant context from the document store.\n"
        "- Prioritize evidence that directly addresses the query.\n"
        "- Synthesize a clear, accurate answer grounded in the retrieved sources or chat history.\n"
        "- If the query cannot be answered from the knowledge source or chat history, do NOT generate your own response.\n"
        "  Instead, clearly state: 'The knowledge source does not contain the required information.'\n"
        "- Provide transparency by including references, tool usage, and reasoning steps."
    ),
    expected_output=(
        "Return a structured JSON object with the following schema:\n\n"
        "{\n"
        "  \"answer\": \"Direct response to the query (1–3 paragraphs, clear and accurate). "
        "If no answer is found, return: 'The knowledge source does not contain the required information.'\",\n"
        "  \"sources\": [\"List of document titles, sections, or citations used (empty list if none)\"],\n"
        "  \"tool_used\": \"Name of the retrieval/analysis tool invoked "
        "(e.g., RAG Retriever, VectorDB, ChatHistory, etc.)\",\n"
        "  \"rationale\": \"Brief explanation of why this answer was chosen, "
        "or why no relevant information was found\"\n"
        "}"
    ),
    output_pydantic=AnswerStructure,
)
