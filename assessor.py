import json
import os
import re
from datetime import datetime
from pathlib import Path

from openai import OpenAI


QUESTIONS_DIR = Path("questoins")


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


def safe_filename(text: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_-]+", "_", text.strip())
    cleaned = cleaned.strip("_")
    return cleaned or "topic"


def get_questions(subject: str) -> list[dict[str, str]]:
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    response = client.chat.completions.create(
        model=model,
        temperature=0.3,
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": (
                    "Create exactly 5 multiple-choice assessment questions. "
                    "Return only JSON with this shape: "
                    '{"result":[{"q":"question","op1":"option","op2":"option",'
                    '"op3":"option","op4":"option","right_answer":"op1"}]}. '
                    "The right_answer value must be exactly one of op1, op2, op3, op4."
                ),
            },
            {"role": "user", "content": f"Subject matter: {subject}"},
        ],
    )

    content = response.choices[0].message.content
    if not content:
        raise RuntimeError("OpenAI returned an empty response.")

    parsed = json.loads(content)
    questions = parsed.get("result")
    if not isinstance(questions, list) or len(questions) != 5:
        raise RuntimeError("OpenAI did not return exactly 5 questions in result.")

    required = {"q", "op1", "op2", "op3", "op4", "right_answer"}
    allowed_answers = {"op1", "op2", "op3", "op4"}

    for index, question in enumerate(questions, start=1):
        if not isinstance(question, dict) or set(question) != required:
            raise RuntimeError(f"Question {index} has an invalid format.")

        if question["right_answer"] not in allowed_answers:
            raise RuntimeError(f"Question {index} has an invalid right_answer.")

        for key in required:
            if not isinstance(question[key], str) or not question[key].strip():
                raise RuntimeError(f"Question {index} has an empty {key}.")

    return questions


def save_questions(subject: str, questions: list[dict[str, str]]) -> Path:
    QUESTIONS_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{safe_filename(subject)}_{timestamp}.txt"
    output_path = QUESTIONS_DIR / filename

    output_path.write_text(
        json.dumps({"result": questions}, indent=2),
        encoding="utf-8",
    )
    return output_path


def ask_question(number: int, question: dict[str, str]) -> bool:
    print(f"\nQuestion {number}: {question['q']}")
    print(f"1. {question['op1']}")
    print(f"2. {question['op2']}")
    print(f"3. {question['op3']}")
    print(f"4. {question['op4']}")

    option_map = {"1": "op1", "2": "op2", "3": "op3", "4": "op4"}
    while True:
        answer = input("Your answer (1-4): ").strip()
        if answer in option_map:
            break
        print("Please enter 1, 2, 3, or 4.")

    selected = option_map[answer]
    correct = selected == question["right_answer"]
    if correct:
        print("Correct.")
    else:
        right_option = question["right_answer"]
        print(f"Incorrect. Right answer: {question[right_option]}")

    return correct


def conduct_assessment(questions: list[dict[str, str]]) -> int:
    score = 0
    for number, question in enumerate(questions, start=1):
        if ask_question(number, question):
            score += 1
    return score


def main() -> None:
    load_env()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or api_key == "dummy":
        raise RuntimeError("Add your OpenAI token to OPENAI_API_KEY in .env first.")

    subject = input("Enter subject matter: ").strip()
    if not subject:
        raise RuntimeError("Please provide a subject matter.")

    print("\nGenerating questions...")
    questions = get_questions(subject)
    saved_path = save_questions(subject, questions)

    print(f"Questions saved to: {saved_path}")
    print("\nStarting assessment.")

    score = conduct_assessment(questions)
    total = len(questions)
    percentage = (score / total) * 100

    print("\nAssessment result")
    print(f"Score: {score}/{total}")
    print(f"Percentage: {percentage:.2f}%")


if __name__ == "__main__":
    main()
