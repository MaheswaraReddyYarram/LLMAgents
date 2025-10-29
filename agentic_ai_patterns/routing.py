import requests
import json

def fetch_temperature(lat: float, lon: float) -> float:
    """Gets the current temperature in Celsius for a given latitude and longitude."""
    print(f"Calling Weather API for lat={lat}, lon={lon}...")
    base_url = "https://api.open-meteo.com/v1/forecast"
    params = {"latitude": lat, "longitude": lon, "current": "temperature_2m"}
    response = requests.get(base_url, params=params)
    response.raise_for_status()
    data = response.json()
    return data["current"]["temperature_2m"]

def retrieve_from_kb(question: str) -> dict:
    """Retrieves information from the Educative knowledge base."""
    print(f"Calling Knowledge Base for question: '{question}'...")
    with open("educative_kb.json", "r") as f:
        return json.load(f)

master_tool_registry = [
    {
        "type": "function",
        "function": {
            "name": "fetch_temperature",
            "description": "Return the current temperature (°C) for a given location by its coordinates.",
            "parameters": {
                "type": "object",
                "properties": {
                    "lat": {"type": "number", "description": "The latitude of the location."},
                    "lon": {"type": "number", "description": "The longitude of the location."}
                },
                "required": ["lat", "lon"],
            },
        }
    },
    {
        "type": "function",
        "function": {
            "name": "retrieve_from_kb",
            "description": "Answer questions about Educative courses and content.",
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {"type": "string", "description": "The user's question about Educative."}
                },
                "required": ["question"]
            }
        }
    }
]

def execute_function_call(name: str, args: dict):
    if name == 'fetch_temperature':
        return fetch_temperature(**args)
    elif name == 'retrieve_from_kb':
        return retrieve_from_kb(**args)
    else:
        return f"Error function {name} not found"


def run_agentic_router(user_query: str):
    print(f"\n--- User Query: '{user_query}' ---")

    # We give the LLM a special instruction to encourage it to guess coordinates.
    system_prompt = (
        "You are a helpful assistant with access to tools. "
        "For the 'fetch_temperature' tool, if the user provides a location name "
        "but not coordinates, **use your own general knowledge to determine the latitude and "
        "longitude, then call the function with those deduced values.** "
        "If you are unsure or the location is ambiguous, ask the user for clarification."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_query}
    ]

    print("Step 1: Asking LLM to deduce parameters and choose a tool...")
    first_response = client.chat.completions.create(
        model="gpt-4.1",
        messages=messages,
        tools=master_tool_registry,
    )

    response_message = first_response.choices[0].message

    # This 'if' statement is the core of the routing logic.
    if response_message.tool_calls:
        # --- PATH A: The LLM chose a tool ---
        tool_name = response_message.tool_calls[0].function.name
        print(f"Step 2: Model decided to use '{tool_name}' and deduced the arguments.")
        function_args = json.loads(response_message.tool_calls[0].function.arguments)
        print(f"   > Deduced Arguments: {function_args}")

        messages.append(response_message)
        tool_output = execute_function_call(name=tool_name, args=function_args)

        messages.append({
            "role": "tool", "tool_call_id": response_message.tool_calls[0].id, "content": json.dumps(tool_output)
        })

        print("Step 3: Generating a final response...")
        second_response = client.chat.completions.create(model="gpt-4.1", messages=messages)
        final_answer = second_response.choices[0].message.content
        print(f"✅ Final Assistant Response: {final_answer}\n")

    else:
        # --- PATH B: The LLM did not choose a tool ---
        print("Step 2: Model decided it could not use a tool.")
        final_answer = response_message.content
        print(f"✅ Final Assistant Response: {final_answer}\n")


if __name__ == "__main__":
    run_agentic_router("Can you check how hot it is in Paris right now?")

    run_agentic_router("What new AI course is Educative releasing?")

    run_agentic_router("Can you write me a short poem about a robot?")