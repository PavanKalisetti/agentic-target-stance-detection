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
    return create_agent(llm, "/home/rgukt/Documents/major project/major-project/langgraph_stance_analyzer/prompts/linguistic_agent.md")


def implicit_target_agent(llm):
    """
    Returns the implicit target identification agent.
    """
    return create_agent(llm, "/home/rgukt/Documents/major project/major-project/langgraph_stance_analyzer/prompts/implicit_target_agent.md")


def explicit_target_agent(llm):
    """
    Returns the explicit target identification agent.
    """
    return create_agent(llm, "/home/rgukt/Documents/major project/major-project/langgraph_stance_analyzer/prompts/explicit_target_agent.md")


def target_decider_agent(llm):
    """
    Returns the target decider agent.
    """
    return create_agent(llm, "/home/rgukt/Documents/major project/major-project/langgraph_stance_analyzer/prompts/target_decider_agent.md")


def debate_agent(llm):
    """
    Returns the debate agent.
    """
    return create_agent(llm, "/home/rgukt/Documents/major project/major-project/langgraph_stance_analyzer/prompts/debate_agent.md")


def stance_agent(llm):
    """
    Returns the stance detection agent.
    """
    return create_agent(llm, "/home/rgukt/Documents/major project/major-project/langgraph_stance_analyzer/prompts/stance_agent.md")

def final_agent(llm):
    """
    Returns the final response generation agent.
    """
    return create_agent(llm, "/home/rgukt/Documents/major project/major-project/langgraph_stance_analyzer/prompts/final_agent.md")