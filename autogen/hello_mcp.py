import asyncio
import asyncio
import logging
import os
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_agentchat.agents import AssistantAgent 
from autogen_ext.tools.mcp import McpWorkbench, StdioServerParams
from autogen_agentchat.ui import Console

# Configure logging
logger = logging.getLogger()
file_handler = logging.FileHandler('log.txt', mode='w')
file_handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s')
file_handler.setFormatter(formatter)

# 4. Add the file handler to the logger
logger.addHandler(file_handler)

# 5. Set the logging level for the logger
logging.getLogger("autogen_core").setLevel(logging.DEBUG)
# ----------------------------------------------------


async def main() -> None:
    try:
        # Configure the model client for Ollama
        model_client = OpenAIChatCompletionClient(
            model="gpt-oss:latest",
            base_url="http://localhost:11434/v1/",
            api_key="sk-not-needed",
            model_info={
                "vision": False,
                "function_calling": True,
                "json_output": True,
                "structured_output": True,
                "family": "gpt-oss"
            }
        )

        # Configure the weather MCP server (use the local weather_mcp package)
        weather_server_params = StdioServerParams(
            command="python",
            args=["mcp/weather.py"],
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # project root
        )
        
        # Start the weather MCP server and create the agent
        async with McpWorkbench(weather_server_params) as weather_mcp:
            agent = AssistantAgent(
                "weather_assistant",
                system_message="""You are a helpful assistant that can provide weather information.""",
                model_client=model_client,
                workbench=weather_mcp,
                model_client_stream=True,
                max_tool_iterations=10,
            )

            await Console(agent.run_stream(task="What's the weather like in Los Angeles, California(34.0549° N, 118.2426° W) today?"))
        
        await model_client.close()

            
    except Exception as e:        
        # Print the error message to the console for user feedback
        print(f"\n--- ERROR OCCURRED ---")
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    asyncio.run(main())