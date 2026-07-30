import os
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI


MAX_USER_REQUESTS = 7


def load_env(path: str = ".env") -> None:
    """Load key=value pairs from a .env file into environment variables."""
    env_path = Path(path)
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")

        if key and key not in os.environ:
            os.environ[key] = value


def build_system_prompt(topic: str) -> str:
    """Create the instruction that keeps the assistant focused on one topic."""
    return (
        f"You are a topic expert on: {topic}.\n"
        "Only answer questions about this topic.\n"
        "If the user asks about anything outside this topic, politely say that "
        "the session is limited to the selected topic and invite them to ask a "
        "question about it.\n"
        "Use the previous messages for continuity."
    )


def create_chat_model() -> ChatOpenAI:
    """Create the LangChain chat model connected to OpenAI."""
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    return ChatOpenAI(
        model=model,
        temperature=0.4,
        max_completion_tokens=75,
        api_key=os.environ["OPENAI_API_KEY"],
    )


def chat_about_topic(topic: str) -> None:
    """Run a 7-request interactive topic-only chat session."""
    chat = create_chat_model()

    # This list is the session memory. Each request and response is appended so
    # the model can continue the conversation with context.
    history = [SystemMessage(content=build_system_prompt(topic))]

    print(f"\nTopic expert session started for: {topic}")
    print(f"You can ask {MAX_USER_REQUESTS} questions. Type 'exit' to stop early.\n")

    for request_number in range(1, MAX_USER_REQUESTS + 1):
        user_text = input(f"You ({request_number}/{MAX_USER_REQUESTS}): ").strip()
        if user_text.lower() in {"exit", "quit"}:
            print("Session ended early.")
            return

        if not user_text:
            print("Please enter a question.")
            continue

        # Add the user's message to history before calling OpenAI through
        # LangChain. Passing the full history gives continuity.
        history.append(HumanMessage(content=user_text))

        response = chat.invoke(history)
        reply = str(response.content).strip()

        # Add the assistant response back into history for the next turn.
        history.append(response)

        print(f"\nExpert: {reply}\n")

    print("Session ended. You used all 7 requests.")


def main() -> None:
    """Collect the topic and start the expert chat."""
    load_env()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or api_key == "dummy":
        raise RuntimeError("Add your OpenAI token to OPENAI_API_KEY in .env first.")

    topic = input("Enter the topic for the expert chat: ").strip()
    if not topic:
        raise RuntimeError("Please provide a topic.")

    chat_about_topic(topic)


if __name__ == "__main__":
    main()

