from langchain_ollama.llms import OllamaLLM
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, List
import operator
import xml.etree.ElementTree as ET

from langgraph_stance_analyzer.agents.agents import (
    linguistic_agent,
    implicit_target_agent,
    explicit_target_agent,
    target_decider_agent,
    debate_agent,
    stance_agent,
    final_agent,
)
from langgraph_stance_analyzer.tools import web_search


class AgentState(TypedDict):
    input: str
    linguistic_analysis: str
    target: str
    target_info: str # New field for external info
    stance: str
    final_response: Annotated[str, operator.add]
    debate_history: List[str]
    max_turns: int


llm = OllamaLLM(model="llama3.1:8b")

linguistic_runnable = linguistic_agent(llm)
implicit_target_runnable = implicit_target_agent(llm)
explicit_target_runnable = explicit_target_agent(llm)
target_decider_runnable = target_decider_agent(llm)
debate_runnable = debate_agent(llm)
stance_runnable = stance_agent(llm)
final_runnable = final_agent(llm)


def get_linguistic_analysis(state):
    print("Linguistic Analysis:", end=" ", flush=True)
    response_content = ""
    for token in linguistic_runnable.stream({"input": state["input"]}):
        print(token, end="", flush=True)
        response_content += token
    print()
    return {"linguistic_analysis": response_content, "debate_history": []}


def decide_target_type(state):
    print("Deciding Target Type:", end=" ", flush=True)
    response_content = ""
    for token in target_decider_runnable.stream(
        {"linguistic_analysis": state["linguistic_analysis"], "input": state["input"]} 
    ):
        print(token, end="", flush=True)
        response_content += token
    print()
    if "implicit" in response_content.lower():
        return "implicit_target_identification"
    else:
        return "explicit_target_identification"


def get_implicit_target(state):
    print("Implicit Target:", end=" ", flush=True)
    response_content = ""
    for token in implicit_target_runnable.stream({"input": state["input"]}):
        print(token, end="", flush=True)
        response_content += token
    print()
    return {"target": response_content}

def get_explicit_target(state):
    print("Explicit Target:", end=" ", flush=True)
    response_content = ""
    for token in explicit_target_runnable.stream({"input": state["input"]}):
        print(token, end="", flush=True)
        response_content += token
    print()
    return {"target": response_content}

def get_target_info(state):
    """
    Fetches information about the target using the web_search tool.
    """
    print("Fact Checking Target:", end=" ", flush=True)
    target = state.get("target", "").strip()
    if not target:
        print("No target to search.")
        return {"target_info": "No information found."}
    
    # Call the web search tool
    search_result = web_search(target)
    print("----------search result-------------")
    print(search_result)
    print("-----------------------")
    return {"target_info": search_result}

def debate_turn(state):
    print(f"Debate Turn {len(state['debate_history']) + 1}:", end=" ", flush=True)
    response_content = ""
    # The debate agent now returns XML, so we handle it as a single string
    for token in debate_runnable.stream(
        {
            "input": state["input"],
            "debate_history": "\n".join(state["debate_history"]),
            "target_info": state["target_info"], # Pass new info
        }
    ):
        print(token, end="", flush=True)
        response_content += token
    print()

    try:
        # Parse the XML response
        root = ET.fromstring(response_content)
        agree_element = root.find("agree")

        if agree_element is not None and agree_element.text == "false":
            new_target_element = root.find("new_target")
            if new_target_element is not None and new_target_element.text:
                new_target = new_target_element.text.strip()
                # When the target changes, we should probably re-run the fact check.
                # For now, we just update the target and continue the debate.
                return {"target": new_target, "debate_history": state["debate_history"] + [response_content]}

        # If agree is true, or if the XML is malformed but we don't want to error out,
        # just add the response to history without changing the target.
        return {"debate_history": state["debate_history"] + [response_content]}

    except ET.ParseError:
        # Handle cases where the response is not valid XML
        print("\nWarning: Could not parse XML from debate agent. Treating as disagreement.")
        # Add the raw (and broken) response to history and continue the debate
        return {"debate_history": state["debate_history"] + [response_content]}

def continue_debate(state):
    # If the target was changed in the last turn, we should re-run fact-checking.
    last_response = state["debate_history"][-1]
    if "<new_target>" in last_response:
         return "get_target_info" # Go back to fact-checking with the new target

    if len(state["debate_history"]) >= state["max_turns"]:
        return "stance_detection"

    try:
        root = ET.fromstring(last_response)
        agree_element = root.find("agree")
        if agree_element is not None and agree_element.text == "true":
            return "stance_detection"
    except ET.ParseError:
        
        return "debate"

    return "debate"

def get_stance(state):
    print("Stance:", end=" ", flush=True)
    input_for_stance = f"Text: {state['input']}\nTarget: {state['target']}\nBackground Information: {state['target_info']}"
    response_content = ""
    for token in stance_runnable.stream({"input": input_for_stance}):
        print(token, end="", flush=True)
        response_content += token
    print()
    return {"stance": response_content}

def get_final_response(state):
    print("Final Response:", end=" ", flush=True)

    input_dict = {
        "linguistic_analysis": state["linguistic_analysis"],
        "target": state["target"],
        "stance": state["stance"],
        "input": "",
    }

    response_content = ""
    for token in final_runnable.stream(input_dict):
        print(token, end="", flush=True)
        response_content += token
    print()
    return {"final_response": response_content}


workflow = StateGraph(AgentState)

workflow.add_node("linguistic_analysis", get_linguistic_analysis)
workflow.add_node("implicit_target_identification", get_implicit_target)
workflow.add_node("explicit_target_identification", get_explicit_target)
workflow.add_node("get_target_info", get_target_info) # New node
workflow.add_node("debate", debate_turn)
workflow.add_node("stance_detection", get_stance)
workflow.add_node("final_response_generation", get_final_response)

workflow.set_entry_point("linguistic_analysis")

workflow.add_conditional_edges(
    "linguistic_analysis",
    decide_target_type,
    {
        "implicit_target_identification": "implicit_target_identification",
        "explicit_target_identification": "explicit_target_identification",
    },
)

workflow.add_edge("implicit_target_identification", "get_target_info") # Edge to new node
workflow.add_edge("explicit_target_identification", "get_target_info") # Edge to new node
workflow.add_edge("get_target_info", "debate") # Edge from new node

workflow.add_conditional_edges(
    "debate",
    continue_debate,
    {
        "stance_detection": "stance_detection",
        "debate": "debate",
        "get_target_info": "get_target_info", # Loop back if target changes
    },
)

workflow.add_edge("stance_detection", "final_response_generation")
workflow.add_edge("final_response_generation", END)

app = workflow.compile()


def main():
    print("Enter your text to analyze (or 'quit' to exit):")
    while True:
        try:
            user_input = input("> ")
            if user_input.lower() == "quit":
                break

            print("-- Analysis --")
            # Set the initial target to an empty string
            initial_state = {"input": user_input, "target": "", "max_turns": 5}
            app.invoke(initial_state)
            print("------------------")

        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"An error occurred: {e}")
            break


if __name__ == "__main__":
    main()

