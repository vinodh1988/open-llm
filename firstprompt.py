import os
import sys
from pathlib import Path

from openai import OpenAI


def load_env(path: str = ".env") -> None:
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


def main() -> None:
    load_env()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or api_key == "dummy":
        raise RuntimeError("Add your OpenAI token to OPENAI_API_KEY in .env first.")

    text = " ".join(sys.argv[1:]).strip() if len(sys.argv) > 1 else input("Text: ").strip()
    if not text:
        raise RuntimeError("Please provide text to classify.")

    client = OpenAI(api_key=api_key)
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    response = client.chat.completions.create(
        model=model,
        temperature=0,
        max_tokens=3,
        messages=[
            {
                "role": "system",
                "content": (
                    "Classify the user's text sentiment. Reply with exactly one word: "
                    "positive, negative, or neutral."
                ),
            },
            {"role": "user", "content": text},
        ],
    )

    sentiment = response.choices[0].message.content.strip().lower()
    allowed = {"positive", "negative", "neutral"}
    if sentiment not in allowed:
        raise RuntimeError(f"OpenAI returned an invalid label: {sentiment!r}")

    print(sentiment)


if __name__ == "__main__":
    main()
