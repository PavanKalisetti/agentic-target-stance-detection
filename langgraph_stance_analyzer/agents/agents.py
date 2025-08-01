
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms import Ollama

def create_agent(llm, system_prompt):
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )
    return prompt | llm

def linguistic_agent(llm):
    system_prompt = "You are a linguistic expert. Analyze the given text and identify key linguistic features, such as sentiment, tone, and style. Stream the output token by token."
    return create_agent(llm, system_prompt)

def target_agent(llm):
    system_prompt = "You are an expert in target identification. Analyze the given text and identify the main target of the text. The target can be a person, an organization, a product, or a concept. Stream the output token by token."
    return create_agent(llm, system_prompt)

def stance_agent(llm):
    system_prompt = "You are an expert in stance detection. Analyze the given text and determine the stance towards the identified target. The stance can be positive, negative, or neutral. Stream the output token by token."
    return create_agent(llm, system_prompt)
