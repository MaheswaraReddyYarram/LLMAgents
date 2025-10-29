import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
import mcp.types as types


# --- Subclass ClientSession to Handle Notifications ---
class LoggingClientSession(ClientSession):
    """
    A custom ClientSession that overrides the notification handler
    to print log messages from the server.
    """

    async def _received_notification(self, notification: types.JSONRPCNotification) -> None:
        """
        This method is automatically called by the base class's internal
        event loop whenever a notification is received.
        """
        # The library pre-parses the notification. The actual notification object
        # is in the .root attribute of the object passed to this handler
        if hasattr(notification, 'root') and notification.root.method == "notifications/message":
            try:
                # The `notification.root.params` object is already a parsed Pydantic model
                # We can access its attributes like `level` and `data` directly
                log_params = notification.root.params
                level = log_params.level.upper()
                message = log_params.data
                print(f"[{level} from Server]: {message}")
            except AttributeError as e:
                print(
                    f"[CLIENT PARSE ERROR]: Log message params have an unexpected structure: {notification.root.params}. Error: {e}")
            except Exception as e:
                print(f"[CLIENT PARSE ERROR]: An unexpected error occurred while processing log message: {e}")


# ---  Main Application Logic ---
async def main():
    server_params = StdioServerParameters(
        command="python",
        args=["weather_server.py"]
    )

    # The 'async with' block handles session start and cleanup
    async with stdio_client(server_params) as (read, write):
        # --- 3. Instantiate our CUSTOM session class ---
        async with LoggingClientSession(read, write) as session:

            await session.initialize()
            print("Logging client is ready.")
            print("Type a location (e.g., 'London') to get weather, or 'quit' to exit.")

            while True:
                user_input = await asyncio.to_thread(input, "\nYou: ")
                if user_input.lower() in {"exit", "quit", "q"}:
                    break

                try:
                    print("--- Calling get_weather tool ---")

                    # Call the tool. Our custom session class will handle
                    # logging automatically in the background.
                    result = await session.call_tool(
                        name="get_weather",
                        arguments={"location": user_input}
                    )

                    print("\n--- Tool Result ---")
                    if result.isError:
                        print(f"Error from tool: {result.content[0].text}")
                    else:
                        print(f"Success: {result.content[0].text}")

                except Exception as e:
                    print(f"An error occurred: {e}")


if __name__ == "__main__":
    asyncio.run(main())