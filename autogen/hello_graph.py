import asyncio
import asyncio
import logging
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_agentchat.agents import AssistantAgent 
from autogen_agentchat.teams import DiGraphBuilder, GraphFlow
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

        writer = AssistantAgent("writer", model_client=model_client, system_message="Draft a short paragraph on climate change.")

        # Create two editor agents
        editor1 = AssistantAgent("editor1", model_client=model_client, system_message="Edit the paragraph for grammar.")

        editor2 = AssistantAgent("editor2", model_client=model_client, system_message="Edit the paragraph for style.")

        # Create the final reviewer agent
        final_reviewer = AssistantAgent(
            "final_reviewer",
            model_client=model_client,
            system_message="Consolidate the grammar and style edits into a final version.",
        )

        # Build the workflow graph
        builder = DiGraphBuilder()
        builder.add_node(writer).add_node(editor1).add_node(editor2).add_node(final_reviewer)

        # Fan-out from writer to editor1 and editor2
        builder.add_edge(writer, editor1)
        builder.add_edge(writer, editor2)

        # Fan-in both editors into final reviewer
        builder.add_edge(editor1, final_reviewer)
        builder.add_edge(editor2, final_reviewer)

        # Build and validate the graph
        graph = builder.build()

        # Create the flow
        flow = GraphFlow(
            participants=builder.get_participants(),
            graph=graph,
        )

        # Run the workflow
        await Console(flow.run_stream(task="Write a short paragraph about climate change."))

            
    except Exception as e:        
        # Print the error message to the console for user feedback
        print(f"\n--- ERROR OCCURRED ---")
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    asyncio.run(main())