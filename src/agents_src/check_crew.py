from pprint import pprint
from src.agents_src.crew import qa_crew

input_dat = {
    "user_query": "Explain about Evolution",
    "chat_history": {}
}

result = qa_crew.kickoff(input_dat)
result_dict = result.to_dict()

pprint(result_dict)