import os
from openai import OpenAI

# Initialize client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def test_llm():
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "What is Artificial Intelligence?"}],
        max_tokens=200
    )
    print("LLM Response:", response.choices[0].message.content.strip())

if __name__ == "__main__":
    test_llm()
