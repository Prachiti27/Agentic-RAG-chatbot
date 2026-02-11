import logging
from src.agents_src.crew import qa_crew

logger = logging.getLogger(__name__)


def get_answer(chat_history: list) -> dict:
    logger.info(f"Received chat_history: {chat_history}")
    last_user_message = chat_history[-1]
    user_query = last_user_message["content"]
    logger.info(f"Extracted user_query: {user_query}")
    history_without_last = chat_history[:-1]
    input_data = {
        "user_query": user_query,
        "chat_history": history_without_last,
    }
    logger.debug(f"Input data for qa_crew: {input_data}")
    result = qa_crew.kickoff(input_data)
    result_dict = result.to_dict()
    logger.info(f"Result from qa_crew: {result_dict}")
    return result_dict

