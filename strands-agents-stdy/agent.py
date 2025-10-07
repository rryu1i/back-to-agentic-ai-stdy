
from strands import Agent
from strands.models.openai import OpenAIModel  

from dotenv import load_dotenv
import os


load_dotenv()


api_key = os.getenv("OPENAI_API_KEY")


openai_model = OpenAIModel(
    client_args={"api_key": api_key},
    model_id="gpt-4o-mini",
    params={"max_tokens": 1000, "temperature": 0.7}
)


agent = Agent(model=openai_model)


agent("Teel me about agentic AI")