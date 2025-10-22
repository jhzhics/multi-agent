import asyncio
import logging
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_agentchat.agents import AssistantAgent 

# ----------------------------------------------------
# MODIFICATIONS FOR FILE LOGGING
# ----------------------------------------------------

# 1. Define the logger object
logger = logging.getLogger()

# 2. Create a file handler
file_handler = logging.FileHandler('log.txt', mode='w') # 'w' overwrites, 'a' appends
file_handler.setLevel(logging.DEBUG)

# 3. Create a formatter and attach it to the file handler
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s')
file_handler.setFormatter(formatter)

# 4. Add the file handler to the logger
logger.addHandler(file_handler)

# 5. Set the logging level for the logger
logging.getLogger("autogen_core").setLevel(logging.DEBUG)
# ----------------------------------------------------


async def main() -> None:

    try:
        model_client = OpenAIChatCompletionClient(model="gpt-oss:latest",
                                                base_url="http://localhost:28563/v1/",
                                                api_key="not-required",
                                                model_info={
                                                        "vision": False,
                                                        "function_calling": True,
                                                        "json_output": True,
                                                        "structured_output": True,
                                                        "family": "gpt-oss"
                                                })

        agent = AssistantAgent("assistant", model_client=model_client)
        print(await agent.run(task="What's the weather like in Los Angeles, California(34.0549° N, 118.2426° W) today?"))
        await model_client.close()
        
    except Exception as e:        
        # Print the error message to the console for user feedback
        print(f"\n--- ERROR OCCURRED ---")
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    asyncio.run(main())