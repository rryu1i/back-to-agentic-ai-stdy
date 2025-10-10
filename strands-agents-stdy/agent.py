
import asyncio
import tempfile
from typing import List, Optional
from pydantic import BaseModel, Field
from strands import Agent
from strands.models.openai import OpenAIModel  
from enum import Enum

from dotenv import load_dotenv
import os
import json


load_dotenv()


api_key = os.getenv("OPENAI_API_KEY")

openai_model = OpenAIModel(
    client_args={"api_key": api_key},
    model_id="gpt-4o-mini",
    params={"max_tokens": 5000, "temperature": 0.1}
)


agent = Agent(model=openai_model)

#%%

# --- 1. Pydantic Models (Data Schema for AI) ---
class ProjectType(str, Enum):
    """Enumeration for the type of project."""
    NEW_PROJECT = "New Project"
    FEATURE_ENHANCEMENT = "Feature Enhancement"

class Feature(BaseModel):
    """Describes a single feature with a title and description."""
    title: str = Field(description="A short, clear title for the feature.")
    description: str = Field(description="A detailed description of what the feature should do.")

class ProjectRequirements(BaseModel):
    """Structured model for capturing project requirements."""
    project_name: str = Field(description="The official name of the project or feature.")
    project_type: ProjectType = Field(description="Is this a new project or an enhancement?")
    stakeholder: str = Field(description="The name of the person requesting the project.")
    problem_statement: str = Field(description="The business problem this project solves.")
    features: List[Feature] = Field(description="A list of the key features required.")
    success_metrics: List[str] = Field(description="How success will be measured.")
    constraints: Optional[List[str]] = Field(default=None, description="Any known limitations.")


SYSTEM_PROMPT = """
You are a 'Project Intake Agent'. Your sole purpose is to gather requirements for a new project or feature from a user.

Follow these rules strictly:
1.  Your goal is to fill out the information for the `ProjectRequirements` schema.
2.  Ask one clear question at a time. Do not overwhelm the user.
3.  Do NOT act as a general chatbot. If the user mentions a topic like 'Dota', do not provide information about it. Acknowledge their answer and ask the next relevant project question.
4.  Keep your responses concise and professional.
5.  When you believe you have gathered all the necessary information, respond with the single word: [DONE]
"""
#%%

def run_dynamic_session():
    # We are NOT using verbose=False here as it's not a valid argument.
    agent = Agent(model=openai_model, system_prompt=SYSTEM_PROMPT)
    
    print("🤖 Hello! I'm here to help you scope out your new project or feature.")
    print("Let's begin.\n")
    
    # We'll start with a generic response to kick off the conversation.
    user_response = "Please start by asking me the first question."
    
    while True:
        # The agent call will now handle PRINTING the question.
        agent_response = agent(
            f"""The user's response is: '{user_response}'.

            Your task is to have a conversation to fill the `ProjectRequirements` schema. Follow these steps:
            1.  **Evaluate the user's last response.** Did it sufficiently answer your previous question?
            2.  **If the response was vague, unclear, or insufficient** (e.g., "I don't know", "no idea"), you MUST ask a clarifying question or rephrase your previous question. DO NOT move on to a new topic.
            3.  **If the response was sufficient**, ask the single next logical question to continue filling the schema.
            4.  **If you have enough information for all fields**, respond with the single word: [DONE]"""
        )
        
        agent_response_text = str(agent_response)

        if "[DONE]" in agent_response_text:
            break
        
        user_response = input("👤 You: ")
        print()

    print("\n----------------------------------------------------")
    print("Thank you! I have all the information I need.")
    print("Processing the conversation to finalize requirements...")
    print("----------------------------------------------------\n")

    try:
        extracted_requirements = agent.structured_output(
            ProjectRequirements,
            "Based on our entire conversation, extract the project requirements into the structured format. The user's name is the stakeholder."
        )

        print("📝 Extracted Requirements:\n")
        print(json.dumps(extracted_requirements.model_dump(), indent=2))
        # save_requirements_to_db(extracted_requirements)

    except Exception as e:
        print(f"An error occurred during extraction: {e}")
        print("Could not finalize requirements. Please try again.")


#%%

if __name__ == '__main__':
    run_dynamic_session()