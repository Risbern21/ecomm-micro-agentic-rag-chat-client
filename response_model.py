from langchain_google_genai import ChatGoogleGenerativeAI

response_model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
    max_tokens=None,
    max_retries=3,
)
