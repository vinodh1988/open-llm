import json
import os
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


def collect_properties() -> list[str]:
    properties = []

    while True:
        property_name = input("Enter property name (-1 to finish): ").strip()
        if property_name == "-1":
            break
        if property_name:
            properties.append(property_name)

    return properties


def generate_json(entity_name: str, properties: list[str]) -> list[dict]:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or api_key == "your_openai_api_key_here":
        raise RuntimeError("Add your OpenAI token to OPENAI_API_KEY in .env first.")

    client = OpenAI(api_key=api_key)
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    prompt = (
        f"Entity name: {entity_name}\n"
        f"Properties: {', '.join(properties)}\n\n"
        "Analyze the entity and properties, then generate realistic sample data. "
        "Return exactly one JSON array containing 5 objects. "
        "Each object must contain exactly the listed properties as keys. "
        "Do not include markdown, explanations, comments, or extra text."
    )

    response = client.chat.completions.create(
        model=model,
        temperature=0.7,
        max_tokens=800,
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": (
                    "You generate valid JSON only. Return an object with one key named "
                    "'data'. The value of 'data' must be an array of exactly 5 objects."
                ),
            },
            {"role": "user", "content": prompt},
        ],
    )

    content = response.choices[0].message.content
    parsed = json.loads(content)
    data = parsed.get("data")

    if not isinstance(data, list) or len(data) != 5:
        raise RuntimeError("OpenAI did not return an array of exactly 5 objects.")

    expected_keys = set(properties)
    for item in data:
        if not isinstance(item, dict) or set(item.keys()) != expected_keys:
            raise RuntimeError("OpenAI returned objects that do not match the requested properties.")

    return data


def main() -> None:
    load_env()

    entity_name = input("Enter entity name: ").strip()
    if not entity_name:
        raise RuntimeError("Entity name is required.")

    properties = collect_properties()
    if not properties:
        raise RuntimeError("At least one property is required.")

    data = generate_json(entity_name, properties)
    print(json.dumps(data, indent=2))


if __name__ == "__main__":
    main()
