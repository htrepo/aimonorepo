from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI

load_dotenv()

# Centralized model configuration
# DEFAULT_MODEL = "gemini-2.5-flash-lite"
DEFAULT_MODEL = "gpt-4o-mini"

_llm = None


def get_llm(model_name=None, temperature=0):
    """
    Returns a consistent LLM instance.
    Supported prefixes: 'gpt' for OpenAI, others for Google Gemini.
    """
    global _llm

    # If a specific model is requested, we don't use the singleton
    if model_name:
        return _create_llm(model_name, temperature)

    # Otherwise return the shared singleton
    if _llm is None:
        _llm = _create_llm(DEFAULT_MODEL, temperature)

    # print which llm is finally being used
    print(f"Using LLM: {DEFAULT_MODEL}")
    return _llm


def _create_llm(model_name, temperature):
    if model_name.startswith("gpt"):
        print(f"Creating OpenAI LLM: {model_name} with temperature: {temperature}")
        return ChatOpenAI(model=model_name, temperature=temperature)
    else:
        print(f"Creating Google LLM: {model_name} with temperature: {temperature}")
        return ChatGoogleGenerativeAI(model=model_name, temperature=temperature)
