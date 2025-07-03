import os
import json
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Set up OpenAI client
llm_client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
)

def evaluate_expression(expression: str):
    """Safely evaluate a simple mathematical expression."""
    try:
        # Only allow safe built-ins
        allowed_names = {k: v for k, v in vars(__builtins__).items() if k in ("abs", "round")}
        result = eval(expression, {"__builtins__": allowed_names}, {})
        return {"expression": expression, "result": result}
    except Exception as e:
        return {"expression": expression, "error": str(e)}

# Register available tools for the LLM
tool_definitions = [
    {
        "type": "function",
        "function": {
            "name": "evaluate_expression",
            "description": "Evaluate a simple mathematical expression (e.g. 2+2*3).",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "A valid Python mathematical expression, e.g. 2+2*3."},
                },
                "required": ["expression"],
            },
        },
    },
]

function_map = {
    "evaluate_expression": evaluate_expression,
}

def process_conversation(chat_history, model="gpt-4o"):
    # First LLM call
    reply = llm_client.chat.completions.create(
        model=model,
        messages=chat_history,
        tools=tool_definitions,
        tool_choice="auto"
    )
    message = reply.choices[0].message
    print("LLM initial reply:", message)

    if message.tool_calls:
        tool_call = message.tool_calls[0]
        func_name = tool_call.function.name
        func_args = json.loads(tool_call.function.arguments)
        call_id = tool_call.id
        # Call the tool
        tool_result = function_map[func_name](**func_args)
        print("Tool result:", tool_result)
        # Add tool call and result to chat history
        chat_history.append({
            "role": "assistant",
            "tool_calls": [
                {
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": func_name,
                        "arguments": json.dumps(func_args),
                    }
                }
            ]
        })
        chat_history.append({
            "role": "tool",
            "tool_call_id": call_id,
            "name": func_name,
            "content": json.dumps(tool_result),
        })
        # Second LLM call for final answer
        followup = llm_client.chat.completions.create(
            model=model,
            messages=chat_history,
            tools=tool_definitions,
            tool_choice="auto"
        )
        final_message = followup.choices[0].message
        print("LLM follow-up:", final_message)
        return final_message
    return "No tool call detected."

# Example run
conversation = [
    {"role": "system", "content": "You are a smart assistant."},
    {"role": "user", "content": "What is the result of (3 + 5) * 2 - 4 / 2?"},
]

result = process_conversation(conversation)

print("\n--- Final message: ---")
print(result.content)
