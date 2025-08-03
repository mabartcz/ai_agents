from dotenv import load_dotenv
from langchain_community.tools import BraveSearch
from typing import Annotated

from langchain.chat_models import init_chat_model
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from visualizer import visualize

def init():

    # Define tools
    websearch_tool = BraveSearch.from_search_kwargs(search_kwargs={"count": 1})
    tools = [websearch_tool]

    class State(TypedDict):
        messages: Annotated[list, add_messages]
        user_input: str
        is_question: bool
        simple_answer: str
        search_answer: str
        final_answer: str

    graph_builder = StateGraph(State)

    # Initialize different LLMs for different purposes
    question_detector_llm = init_chat_model("openai:gpt-4.1-nano")
    simple_llm = init_chat_model("openai:gpt-4.1-nano")
    search_llm = init_chat_model("openai:gpt-4.1-nano")
    search_llm_with_tools = search_llm.bind_tools(tools)
    verification_llm = init_chat_model("openai:gpt-4.1-nano")

    tool_node = ToolNode(tools)

    def question_detector(state: State):
        """Determine if the user input is a question"""
        print(f"🔍 Question Detector: Analyzing input: '{state['user_input']}'")

        prompt = f"""Analyze the following user input and determine if it's a question that needs an answer.
        
        User input: "{state['user_input']}"
        
        Respond with only "YES" if it's a question that needs an answer, or "NO" if it's not a question.
        Consider statements, greetings, commands, or non-question inputs as "NO"."""

        response = question_detector_llm.invoke([{"role": "user", "content": prompt}])
        is_question = "YES" in response.content.upper()

        print(f"🔍 Question Detector Result: {'YES' if is_question else 'NO'} - '{response.content.strip()}'")

        return {"is_question": is_question}

    def check_if_question(state: State):
        """Conditional edge function to route based on whether input is a question"""
        if state["is_question"]:
            print("➡️ Routing to: Simple Answer (detected question)")
            return "simple_answer"
        else:
            print("➡️ Routing to: Not Question Handler")
            return "not_question"

    def not_question_handler(state: State):
        """Handle non-question inputs"""
        print("❌ Not Question Handler: Input is not a question")
        return {"final_answer": "Please ask a question."}

    def simple_answer_node(state: State):
        """Generate a simple answer without tools"""
        print(f"💭 Simple LLM: Generating basic answer for: '{state['user_input']}'")

        prompt = f"Answer this question directly and concisely: {state['user_input']}"
        response = simple_llm.invoke([{"role": "user", "content": prompt}])

        print(f"💭 Simple LLM Answer: {response.content[:100]}{'...' if len(response.content) > 100 else ''}")

        return {"simple_answer": response.content}

    def search_answer_node(state: State):
        """Generate an answer using search tools"""
        print(f"🔍 Search Agent: Generating search-enhanced answer for: '{state['user_input']}'")

        messages = [{"role": "user", "content": state['user_input']}]
        response = search_llm_with_tools.invoke(messages)

        # Check if tools were called
        if hasattr(response, 'tool_calls') and response.tool_calls:
            print(f"🛠️ Search Agent: Called {len(response.tool_calls)} tool(s)")
            for i, tool_call in enumerate(response.tool_calls):
                print(f"   Tool {i+1}: {tool_call['name']} with args: {tool_call['args']}")
        else:
            print("🛠️ Search Agent: No tools called, using direct response")

        # Get the content or indicate tool call is needed
        search_content = response.content if response.content else ""
        if not search_content and hasattr(response, 'tool_calls') and response.tool_calls:
            search_content = "Search tool called - waiting for results"

        print(f"🔍 Search Agent Answer: {search_content[:100]}{'...' if len(search_content) > 100 else ''}")

        return {"messages": [response], "search_answer": search_content}

    # Custom tool node with better logging
    def custom_tool_node(state: State):
        """Process tool calls with logging"""
        print("🛠️ Tool Node: Processing tool calls...")

        # Get the last message which should contain tool calls
        last_message = state["messages"][-1]

        if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
            print(f"🛠️ Tool Node: Executing {len(last_message.tool_calls)} tool call(s)")

            # Use the original tool node to process
            result = tool_node.invoke(state)

            # Log the tool results
            if "messages" in result:
                for msg in result["messages"]:
                    if hasattr(msg, 'content') and msg.content:
                        print(f"🛠️ Tool Result: {msg.content[:150]}{'...' if len(msg.content) > 150 else ''}")

            return result
        else:
            print("🛠️ Tool Node: No tool calls found")
            return {"messages": []}

    # Modified search answer node to handle tool results
    def search_answer_with_results(state: State):
        """Generate final search answer after tools have been executed"""
        print("🔍 Search Agent: Processing tool results and generating final answer")

        # Get all messages including tool results
        all_messages = state.get("messages", [])

        # Find tool results in messages
        tool_results = []
        for msg in all_messages:
            if hasattr(msg, 'content') and msg.content and "search" in str(type(msg)).lower():
                tool_results.append(msg.content)

        if tool_results:
            print(f"🔍 Search Agent: Found {len(tool_results)} tool result(s)")
            # Create a comprehensive prompt with search results
            search_context = "\n".join(tool_results)
            prompt = f"Based on the search results below, answer this question: {state['user_input']}\n\nSearch Results:\n{search_context}"

            response = search_llm.invoke([{"role": "user", "content": prompt}])
            search_answer = response.content
        else:
            # Fallback if no tool results
            print("🔍 Search Agent: No tool results found, generating direct answer")
            response = search_llm.invoke([{"role": "user", "content": state['user_input']}])
            search_answer = response.content

        print(f"🔍 Search Agent Final Answer: {search_answer[:100]}{'...' if len(search_answer) > 100 else ''}")

        return {"search_answer": search_answer}

    def verification_node(state: State):
        """Compare simple and search answers and provide the best response"""
        print("⚖️ Verification LLM: Comparing answers and generating final response")
        print(f"   Simple answer length: {len(state['simple_answer'])} chars")
        print(f"   Search answer length: {len(state['search_answer'])} chars")

        prompt = f"""Compare these two answers to the question: "{state['user_input']}"

        Simple Answer: {state['simple_answer']}
        
        Search-Enhanced Answer: {state['search_answer']}
        
        Determine which answer is better and more accurate. Provide a final, comprehensive answer that combines
         the best aspects of both if needed. If the search-enhanced answer has more current or specific information,
          favor that. If the simple answer is sufficient and accurate, you can use that."""

        response = verification_llm.invoke([{"role": "user", "content": prompt}])

        print(f"⚖️ Verification Complete: Final answer ready ({len(response.content)} chars)")

        return {"final_answer": response.content}

    # Add nodes to the graph
    graph_builder.add_node("question_detector", question_detector)
    graph_builder.add_node("not_question", not_question_handler)
    graph_builder.add_node("simple_answer", simple_answer_node)
    graph_builder.add_node("search_answer", search_answer_node)
    graph_builder.add_node("tools", custom_tool_node)
    graph_builder.add_node("search_with_results", search_answer_with_results)
    graph_builder.add_node("verification", verification_node)

    # Add edges
    graph_builder.add_edge(START, "question_detector")

    graph_builder.add_conditional_edges(
        "question_detector",
        check_if_question,
        {"simple_answer": "simple_answer", "not_question": "not_question"}
    )

    graph_builder.add_edge("not_question", END)
    graph_builder.add_edge("simple_answer", "search_answer")

    graph_builder.add_conditional_edges(
        "search_answer",
        tools_condition,
        {"tools": "tools", "__end__": "verification"}
    )

    graph_builder.add_edge("tools", "search_with_results")
    graph_builder.add_edge("verification", END)

    graph = graph_builder.compile()
    visualize(graph, "graph.png")

    return graph



if __name__ == "__main__":
    load_dotenv()

    graph = init()


    def stream_graph_updates(user_input: str):
        initial_state = {
            "messages": [],
            "user_input": user_input,
            "is_question": False,
            "simple_answer": "",
            "search_answer": "",
            "final_answer": ""
        }

        for event in graph.stream(initial_state):
            for key, value in event.items():
                if key == "verification" and "final_answer" in value:
                    print("Assistant:", value["final_answer"])
                elif key == "not_question" and "final_answer" in value:
                    print("Assistant:", value["final_answer"])


    while True:
        user_input = input("User: ")
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break

        stream_graph_updates(user_input)
