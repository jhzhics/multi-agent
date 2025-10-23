import asyncio
import asyncio
import logging
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_agentchat.agents import AssistantAgent 
from autogen_agentchat.tools import AgentTool
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

        math_agent = AssistantAgent(
            "math_expert",
            model_client=model_client,
            system_message="You are a math expert.",
            description="A math expert assistant.",
            model_client_stream=True,
        )
        math_agent_tool = AgentTool(math_agent, return_value_as_last_message=True)

        chemistry_agent = AssistantAgent(
            "chemistry_expert",
            model_client=model_client,
            system_message="You are a chemistry expert.",
            description="A chemistry expert assistant.",
            model_client_stream=True,
        )
        chemistry_agent_tool = AgentTool(chemistry_agent, return_value_as_last_message=True)

        agent = AssistantAgent(
            "assistant",
            system_message="You are a general assistant. Use expert tools when needed.",
            model_client=model_client,
            model_client_stream=True,
            tools=[math_agent_tool, chemistry_agent_tool],
            max_tool_iterations=10,
        )
        await Console(agent.run_stream(task="What is the integral of x^2?"))
        await Console(agent.run_stream(task="What is the molecular weight of water?"))

            
    except Exception as e:        
        # Print the error message to the console for user feedback
        print(f"\n--- ERROR OCCURRED ---")
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    asyncio.run(main())