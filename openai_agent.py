import asyncio
import openai
import logging
import http.client
import sys  # Import sys for potential stdout access

# ----------------------------------------------------
# MODIFICATIONS FOR FILE LOGGING
# ----------------------------------------------------

# 1. Define the logger object
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

# 2. Create a file handler
file_handler = logging.FileHandler('log.txt', mode='w') # 'w' overwrites, 'a' appends
file_handler.setLevel(logging.DEBUG)

# 3. Create a formatter and attach it to the file handler
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s')
file_handler.setFormatter(formatter)

# 4. Add the file handler to the logger
logger.addHandler(file_handler)

# 5. Optionally, keep the console handler for live monitoring (uncomment if needed)
# console_handler = logging.StreamHandler(sys.stdout)
# console_handler.setLevel(logging.INFO)
# console_handler.setFormatter(formatter)
# logger.addHandler(console_handler)

# 6. Set the debug level for http.client (this output is captured by the file handler)
http.client.HTTPConnection.debuglevel = 1 
# ----------------------------------------------------


async def main() -> None:
    # 1. Initialize client to None so it can be safely referenced in 'finally'
    client = None 
    print("Attempting DIRECT API call using standard openai client...")

    try:
        # 2. Initialize client inside the try block (Note: The original script used 
        #    openai.OpenAI, which is synchronous. For an async main() function, 
        #    it should be openai.AsyncOpenAI. I will use the synchronous one 
        #    since it's in your prompt, but be aware of the sync/async mismatch.)
        client = openai.OpenAI(
            base_url="http://localhost:28563/v1/", 
            api_key="not-required", 
            timeout=90.0
        )

        # 3. Use the synchronous chat completions method
        response = client.chat.completions.create(
            model="gpt-oss:latest",
            messages=[
                {"role": "user", "content": "Hello! Please introduce yourself and provide a brief (2-3 sentence) summary of your capabilities."}
            ]
        )
        
        # Log successful completion message
        logger.info("\n--- API Call Succeeded ---")
        logger.info(f"Response Content: {response.choices[0].message.content}")

        # Still print the success message to the console for user feedback
        print("\n--- API Call Succeeded ---")
        print(response.choices[0].message.content)
        
    except Exception as e:
        # Log the error message
        logger.error(f"\n--- ERROR OCCURRED ---")
        logger.error(f"An error occurred: {e}")
        
        # Still print the error message to the console for user feedback
        print(f"\n--- ERROR OCCURRED ---")
        print(f"An error occurred: {e}")

    finally:
        # Note: The synchronous client does not have a standard 'aclose()' method,
        # and closing it is often not strictly necessary unless connection pooling 
        # is a concern. We'll skip client closing for the synchronous client in async context.
        pass


if __name__ == "__main__":
    # Note: If client is synchronous (OpenAI), running it in asyncio.run(main())
    # can cause warnings or issues. For best practice, use AsyncOpenAI with 
    # asyncio.run, or use a sync main() without asyncio.run.
    # We proceed with your current structure for log demonstration.
    asyncio.run(main())