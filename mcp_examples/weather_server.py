import os
import requests
from mcp.server.fastmcp import FastMCP, Context
from dotenv import load_dotenv
import logging
from pathlib import Path
from typing import List
load_dotenv()
OPENWEATHERMAP_API_KEY = os.getenv('OPENWEATHERMAP_API_KEY')

mcp = FastMCP("WeatherAssistant")

@mcp.tool()
async def get_weather(location: str, ctx: Context) -> dict:
    """
    Fetches the current weather for a specified location suing OpenWeatherMap API.

    Args:
        location: The city name and optional country code (e.g., "London,uk").

    Returns:
        A dictionary containing the weather information or error information
    """
    await ctx.info(f"Received request for weather in location: {location}")
    print(f"Fetching weather for location: {location}")
    if not OPENWEATHERMAP_API_KEY:
        await ctx.error("Server-side error: OpenWeatherMap API key is not set.")
        return {"error": "OpenWeatherMap API key is not set."}

    base_url = "http://api.openweathermap.org/data/2.5/weather"
    params = {
        "q": location,
        "appid": OPENWEATHERMAP_API_KEY,
        "units": "metric"  # Use "imperial" for Fahrenheit
    }
    try:
        await ctx.debug(f"Making API request to OpenWeatherMap API: {base_url} with params:{params}")
        response = requests.get(base_url, params=params)
        print(f"API Response Status Code: {response.status_code}")
        response.raise_for_status()

        data = response.json()
        await ctx.info("Successfully fetched weather data.")

        # extracting relevant information
        weather_description = data['weather'][0]['description']
        temperature = data['main']['temp']
        feels_like = data['main']['feels_like']
        humidity = data['main']['humidity']
        wind_speed = data['wind']['speed']


        return {
            "location": data['name'],
            "weather_description": weather_description,
            "temperature_celsius": temperature,
            "feels_like_celsius": feels_like,
            "humidity_percent": humidity,
            "wind_speed_m_s": wind_speed
        }
    except requests.exceptions.HTTPError as http_err:
        if response.status_code == 404:
            return {"error": f"Could not find weather data for '{location}'. Please check the location name."}
        elif response.status_code == 401:
            return {"error": "Authentication failed. The API key is likely invalid or inactive."}
        else:
            return {"error": f"An HTTP error occurred: {http_err}"}
    except requests.exceptions.RequestException as req_err:
        return {"error": f"A network error occurred: {req_err}"}
    except KeyError:
        return {"error": "Received unexpected data format from the weather API."}
    except Exception as e:
        return {"error": f"An unexpected error occurred: {e}"}


@mcp.prompt()
def compare_weather_prompt(location_a: str, location_b: str) -> str:
    """
    Generates a clear comparative summary of the weather between two specified locations.
    This is the best choice when a user asks to compare, contrast, or see the difference in weather between two places.

    Args:
        location_a: The first city for comparison (e.g., "London").
        location_b: The second city for comparison (e.g., "Paris").
    """
    return f"""
    You are acting as a helpful weather analyst. Your goal is to provide a clear and easy-to-read comparison of the weather in two different locations for a user.

    The user wants to compare the weather between "{location_a}" and "{location_b}".

    To accomplish this, follow these steps:
    1.  First, gather the necessary weather data for both "{location_a}" and "{location_b}".
    2.  Once you have the weather data for both locations, DO NOT simply list the raw results.
    3.  Instead, synthesize the information into a concise summary. Your final response should highlight the key differences, focusing on temperature, the general conditions (e.g., 'sunny' vs 'rainy'), and wind speed.
    4.  Present the comparison in a structured format, like a markdown table or a clear bulleted list, to make it easy for the user to understand at a glance.
    """

@mcp.resource("file://delivery_log")
def delivery_log_resources() -> List[str]:
    """
    Reads a delivery log file and returns its contents as a list of lines.
    Each line contains an order number and a delivery location.
    :return:
    """
    try:
        log_file = Path("delivery_log.txt")
        if not log_file.exists():
            return ["Error: delivery_log.txt log file not found on the server."]

        #Read the file and return lines as a list
        return log_file.read_text(encoding='utf-8').strip().splitlines()

    except Exception as e:
        return [f"An unexpected error occurred while reading the delivery log: {str(e)}"]


if __name__ == '__main__':
    logging.getLogger("mcp").setLevel(logging.WARNING)
    load_dotenv()  # Load environment variables from a .env file if present
    mcp.run(transport="stdio")
