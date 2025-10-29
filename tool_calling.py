import requests

def fetch_temperature(lat, lon):
    response = requests.get(
        f"https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat}&longitude={lon}"
        f"&current=temperature_2m"
    )
    data = response.json()
    return data["current"]["temperature_2m"]

from openai import OpenAI
import json

client = OpenAI(api_key=("{{OPENAI_API_KEY}}"))

tool_registry = [{
    "type": "function",
    "name": "fetch_temperature",
    "description": "Returns the current temperature (in Celsius) for a given location's coordinates.",
    "parameters": {
        "type": "object",
        "properties": {
            "lat": {"type": "number"},
            "lon": {"type": "number"}
        },
        "required": ["lat", "lon"],
        "additionalProperties": False
    },
    "strict": True
}]

conversation = [{"role": "user", "content": "Can you check how hot it is in Tokyo right now?"}]

first_response = client.responses.create(
    model="gpt-4.1",
    input=conversation,
    tools=tool_registry,
)

print(first_response.output)

tool_suggestion = first_response.output[0]
tool_args = json.loads(tool_suggestion.arguments)

temp_result = fetch_temperature(tool_args["lat"], tool_args["lon"])
print(temp_result)

conversation.append(tool_suggestion)  # include model’s tool call
conversation.append({                 # include your function’s actual output
    "type": "function_call_output",
    "call_id": tool_suggestion.call_id,
    "output": str(temp_result)
})

completion_final = client.responses.create(
    model="gpt-4.1",
    input=conversation,
    tools=tool_registry,
)
print(completion_final.output_text)