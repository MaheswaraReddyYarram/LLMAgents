import logging
import os
import base64
import mimetypes
from pathlib import Path
from google import genai
from google.genai import types
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP, Context
load_dotenv()

# Initialize the FastMCP server
mcp = FastMCP("VisualAnalysisServer")
google_api_key = os.getenv('GOOGLE_API_KEY')
@mcp.tool()
async def load_image_from_path(file_path: str, ctx: Context) -> dict:
    """
    Loads an image from server accessible path , encodes to base64 and determines its mimetype

    :param file_path: absolute file path

    :return:
    """
    try:
        await ctx.info(f"Received request with file path:{file_path}")
        print(f"Received request with file path:{file_path}")
        image_path = Path(file_path)
        if not image_path.is_file():
            return {"error": f"File not found: {file_path}"}

        # open file in binary mode
        with open(image_path, "rb") as img_file:
            img_data = img_file.read()

        # encode binary data to base64
        base64_string = base64.b64encode(img_data).decode('utf-8')

        # guess MIME type
        mime_type, _ = mimetypes.guess_type(image_path)
        await ctx.info(f"Read file successfully with mime_type:{mime_type}")

        if not mime_type:
            mime_type = "application/octet-stream"

        return {
            "base64_image_string": base64_string,
            "mime_type": mime_type
        }
    except FileNotFoundError as e:
        return {"error": f"File not found: {file_path}"}
    except Exception as e:
        return {"error": f"An unexpected error occurred while processing image: str(e)"}

@mcp.tool()
async def get_image_description(base64_image_string: str, mime_type: str, ctx: Context) -> str:
    """
    Performs a deep analysis of a Base64 encoded image and returns a detailed,
    descriptive paragraph about its content. If the image is of a known landmark,
    it will be specifically identified. This description is intended to be used
    as a high-quality search query for a research tool.

    Args:
        base64_image_string: The image file encoded as a Base64 string.
        mime_type: The MIME type of the image (e.g., "image/jpeg", "image/png").

    Returns:
        A single string containing a detailed description of the image.
        Returns an error message if analysis fails.

    """
    try:
        await ctx.info(f"received request to get image description for mime_type:{mime_type}")
        print(f"received request to get image description for mime_type:{mime_type}")
        image_bytes = base64.b64decode(base64_image_string)

        #This explicitly creates the image part of the prompt using the official SDK type
        image_part = types.Part.from_bytes(mime_type= mime_type, data = image_bytes)

        # prompt text
        prompt_text = (
            "Analyze this image in detail. Provide a concise, one-paragraph description. "
            "If it is a famous landmark, work of art, or specific location, identify it by name. "
            "Focus on the most important and defining elements in the image that would be useful for a web search. "
            "For example, instead of 'a building', say 'the Eiffel Tower in Paris'. "
            "Do not add any conversational filler; return only the description."
        )

        #genai.configure(api_key=google_api_key)
        #client = genai.GenerativeModel('gemini-2.0-flash-exp')

        client = genai.Client('gemini-2.5-flash',api_key=google_api_key)
        response = client.generate_content([image_part,prompt_text])

        #response = client.generate_content([image_part, prompt_text])

        description = response.text.strip()
        await ctx.info(f"Image description is {description}")
        return description
    except Exception as e:
        await ctx.error("An error occurred during image analysis")
        return f"An error occurred during image analysis: {str(e)}"

if __name__ == '__main__':
    logging.getLogger("mcp").setLevel(logging.DEBUG)
    mcp.run(transport="stdio")
