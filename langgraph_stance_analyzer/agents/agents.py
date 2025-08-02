from langchain_core.prompts import ChatPromptTemplate

def create_agent(llm, prompt_path):
    """
    Creates a LangChain agent from a prompt file.
    """
    with open(prompt_path, 'r') as f:
        system_prompt = f.read()
    
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )
    return prompt | llm

def linguistic_agent(llm):
    """
    Returns the linguistic analysis agent.
    """
    return create_agent(llm, "/home/rgukt/Documents/major-project/langgraph_stance_analyzer/prompts/linguistic_agent.md")

def target_agent(llm):
    """
    Returns the target identification agent.
    """
    return create_agent(llm, "/home/rgukt/Documents/major-project/langgraph_stance_analyzer/prompts/target_agent.md")

def stance_agent(llm):
    """
    Returns the stance detection agent.
    """
    return create_agent(llm, "/home/rgukt/Documents/major-project/langgraph_stance_analyzer/prompts/stance_agent.md")

def final_agent(llm):
    """
    Returns the final response generation agent.
    """
    return create_agent(llm, "/home/rgukt/Documents/major-project/langgraph_stance_analyzer/prompts/final_agent.md")