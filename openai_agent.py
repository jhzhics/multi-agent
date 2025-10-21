import asyncio
import openai

async def main() -> None:
    # 1. Initialize client to None so it can be safely referenced in 'finally'
    client = None 
    print("Attempting DIRECT API call using standard openai client...")

    try:
        # 2. Initialize client inside the try block
        client = openai.OpenAI(
            base_url="http://localhost:28563/v1/", 
            api_key="not-required", 
            timeout=90.0
        )

        # 3. Use the standard asynchronous chat completions method
        response = client.chat.completions.create(
            model="gpt-oss:latest",
            messages=[
                {"role": "user", "content": "Hello! Please introduce yourself and provide a brief (2-3 sentence) summary of your capabilities."}
            ]
        )
        
        print("\n--- API Call Succeeded ---")
        print(response.choices[0].message.content)
        
    except Exception as e:
        # This catches the real error (the root cause, like a 503)
        print(f"\n--- ERROR OCCURRED ---")
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    asyncio.run(main())