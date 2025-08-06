from langchain_ollama.llms import OllamaLLM
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
import operator

from langgraph_stance_analyzer.agents.agents import linguistic_agent, target_agent, stance_agent, final_agent


class AgentState(TypedDict):
    input: str
    linguistic_analysis: str
    target: str
    stance: str
    final_response: Annotated[str, operator.add]


llm = OllamaLLM(model="llama3.1:8b")

linguistic_runnable = linguistic_agent(llm)
target_runnable = target_agent(llm)
stance_runnable = stance_agent(llm)
final_runnable = final_agent(llm)

def get_linguistic_analysis(state):
    print("Linguistic Analysis:", end=" ", flush=True)
    response_content = ""
    for token in linguistic_runnable.stream({"input": state["input"]}):
        print(token, end="", flush=True)
        response_content += token
    print()  
    return {"linguistic_analysis": response_content}

def get_target(state):
    print("Target:", end=" ", flush=True)
    response_content = ""
    for token in target_runnable.stream({"input": state["input"]}):
        print(token, end="", flush=True)
        response_content += token
    print()  
    return {"target": response_content}


def get_stance(state):
    print("Stance:", end=" ", flush=True)
    input_for_stance = f"Text: {state['input']}\nTarget: {state['target']}"
    response_content = ""
    for token in stance_runnable.stream({"input": input_for_stance}):
        print(token, end="", flush=True)
        response_content += token
    print()  
    return {"stance": response_content}


def get_final_response(state):
    print("Final Response:", end=" ", flush=True)

    input_dict = {
        "linguistic_analysis": state['linguistic_analysis'],
        "target": state['target'],
        "stance": state['stance'],
        "input": ""  
    }

    response_content = ""
    for token in final_runnable.stream(input_dict):
        print(token, end="", flush=True)
        response_content += token
    print()  
    return {"final_response": response_content}


workflow = StateGraph(AgentState)

workflow.add_node("linguistic_analysis", get_linguistic_analysis)
workflow.add_node("target_identification", get_target)
workflow.add_node("stance_detection", get_stance)
workflow.add_node("final_response_generation", get_final_response)

workflow.set_entry_point("linguistic_analysis")
workflow.add_edge("linguistic_analysis", "target_identification")
workflow.add_edge("target_identification", "stance_detection")
workflow.add_edge("stance_detection", "final_response_generation")
workflow.add_edge("final_response_generation", END)

app = workflow.compile()


def main():
    print("Enter your text to analyze (or 'quit' to exit):")
    while True:
        try:
            user_input = input("> ")
            if user_input.lower() == 'quit':
                break

            print("--- Analysis ---")
            app.invoke({"input": user_input})
            print("------------------")

        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"An error occurred: {e}")
            break


if __name__ == "__main__":
    main()