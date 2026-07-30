import math
import os
import re
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings


FAQ_FILE = Path("faqs.txt")
MAX_USER_MESSAGES = 10
MAX_ASSISTANT_MESSAGES = 10
TOP_K_FAQS = 4


def load_env(path: str = ".env") -> None:
    """Load OPENAI_API_KEY and other settings from a local .env file."""
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


def split_faqs(raw_text: str) -> list[str]:
    """Split faqs.txt into separate FAQ chunks for vector search."""
    cleaned = raw_text.strip()
    parts = re.split(r"\n(?=\d+\.\s)", cleaned)

    chunks = []
    for part in parts:
        text = part.strip()
        if not text:
            continue

        # Keep the heading attached to the first FAQ so the college context is
        # available during retrieval.
        chunks.append(text)

    return chunks


def cosine_similarity(first: list[float], second: list[float]) -> float:
    """Compare two embedding vectors."""
    dot_product = sum(a * b for a, b in zip(first, second))
    first_length = math.sqrt(sum(a * a for a in first))
    second_length = math.sqrt(sum(b * b for b in second))
    if not first_length or not second_length:
        return 0.0
    return dot_product / (first_length * second_length)


def build_vector_knowledge_base() -> list[dict[str, object]]:
    """Vectorize faqs.txt using LangChain OpenAI embeddings."""
    if not FAQ_FILE.exists():
        raise RuntimeError("faqs.txt was not found in the project folder.")

    faq_text = FAQ_FILE.read_text(encoding="utf-8")
    faq_chunks = split_faqs(faq_text)
    if not faq_chunks:
        raise RuntimeError("faqs.txt does not contain any FAQ content.")

    embeddings = OpenAIEmbeddings(api_key=os.environ["OPENAI_API_KEY"])
    vectors = embeddings.embed_documents(faq_chunks)

    return [
        {"text": chunk, "vector": vector}
        for chunk, vector in zip(faq_chunks, vectors)
    ]


def retrieve_relevant_faqs(
    question: str,
    knowledge_base: list[dict[str, object]],
) -> str:
    """Find the FAQ entries closest to the user's question."""
    embeddings = OpenAIEmbeddings(api_key=os.environ["OPENAI_API_KEY"])
    query_vector = embeddings.embed_query(question)

    scored_chunks = []
    for item in knowledge_base:
        score = cosine_similarity(query_vector, item["vector"])
        scored_chunks.append((score, str(item["text"])))

    scored_chunks.sort(reverse=True, key=lambda item: item[0])
    best_chunks = [chunk for _, chunk in scored_chunks[:TOP_K_FAQS]]
    return "\n\n".join(best_chunks)


def create_chat_model() -> ChatOpenAI:
    """Create the LangChain chat model used for ACT lab answers."""
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    return ChatOpenAI(
        model=model,
        temperature=0,
        max_completion_tokens=120,
        api_key=os.environ["OPENAI_API_KEY"],
    )


def build_system_prompt() -> str:
    """Create the rule set that keeps the chatbot limited to ACT lab support."""
    return (
        "You are the ABC College of Technology IT Lab support assistant.\n"
        "Answer only ACT IT Lab questions using the provided FAQ context.\n"
        "Use chat history only to maintain continuity about ACT IT Lab queries.\n"
        "Do not answer general technology, personal, coding, college admission, "
        "or unrelated questions.\n"
        "If the answer is not in the FAQ context or the question is outside ACT "
        "IT Lab scope, say: 'I can help only with ACT IT Lab queries from the FAQ knowledge base.'\n"
        "Keep answers short and practical."
    )


def answer_question(
    chat: ChatOpenAI,
    history: list[HumanMessage | AIMessage],
    question: str,
    knowledge_base: list[dict[str, object]],
) -> AIMessage:
    """Retrieve FAQ context and generate a scoped answer."""
    faq_context = retrieve_relevant_faqs(question, knowledge_base)

    messages = [
        SystemMessage(content=build_system_prompt()),
        SystemMessage(content=f"FAQ context:\n{faq_context}"),
        *history,
        HumanMessage(content=question),
    ]

    response = chat.invoke(messages)
    return AIMessage(content=str(response.content).strip())


def run_chatbot() -> None:
    """Run the 10-message command-line ACT IT Lab chatbot session."""
    chat = create_chat_model()
    knowledge_base = build_vector_knowledge_base()
    history: list[HumanMessage | AIMessage] = []
    assistant_replies = 0

    print("ACT IT Lab FAQ Chatbot")
    print("Ask up to 10 ACT IT Lab questions. Type 'exit' to stop early.\n")

    for user_messages in range(1, MAX_USER_MESSAGES + 1):
        question = input(f"You ({user_messages}/{MAX_USER_MESSAGES}): ").strip()
        if question.lower() in {"exit", "quit"}:
            print("Session ended early.")
            return

        if not question:
            print("Please enter an ACT IT Lab question.")
            continue

        if assistant_replies >= MAX_ASSISTANT_MESSAGES:
            print("Assistant response limit reached.")
            return

        answer = answer_question(chat, history, question, knowledge_base)
        assistant_replies += 1

        history.append(HumanMessage(content=question))
        history.append(answer)

        print(f"\nAssistant ({assistant_replies}/{MAX_ASSISTANT_MESSAGES}): {answer.content}\n")

    print("Session ended. You used all 10 input messages.")


def main() -> None:
    """Validate configuration and start the chatbot."""
    load_env()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or api_key == "dummy":
        raise RuntimeError("Add your OpenAI token to OPENAI_API_KEY in .env first.")

    run_chatbot()


if __name__ == "__main__":
    main()
