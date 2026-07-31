import csv
import json
import os
import smtplib
from email.message import EmailMessage
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from cryptography.fernet import Fernet, InvalidToken
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI


EVENTS_FILE = Path("events.csv")
BOOKINGS_FILE = Path("bookings.csv")
HISTORY_FILE = Path("chat_history.json")
DATE_FORMAT = "%Y-%m-%d"
DEFAULT_SMTP_HOST = "smtpout.secureserver.net"
DEFAULT_SMTP_PORT = 465
DEFAULT_EMAIL_TEMPLATE = """Hello {participant_name},

Your booking is confirmed.

Booking ID: {booking_id}
Event: {event_name}
Speaker: {speaker}
Date: {date}
Time: {time}

Regards,
Technical Events Team
"""


def load_env(path: str = ".env") -> None:
    """Load local environment variables without requiring python-dotenv."""
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


def ensure_files_exist() -> None:
    """Create storage files if a user runs the bot in a fresh folder."""
    if not EVENTS_FILE.exists():
        EVENTS_FILE.write_text(
            "name,speaker,seats,booked,time,date\n"
            "Python Automation Bootcamp,Dr. Nisha Rao,40,40,10:00,2026-07-15\n"
            "Intro to DevOps Pipelines,Vikram Sethi,32,28,13:30,2026-07-22\n"
            "Cloud Native Containers,Arjun Menon,35,12,14:00,2026-07-31\n"
            "Observability War Room,Neha Thomas,25,9,17:00,2026-07-31\n"
            "AI Agents with LangChain,Meera Kapoor,50,18,11:00,2026-08-08\n"
            "Secure API Design,Rahul Iyer,30,7,15:30,2026-08-20\n"
            "Kubernetes Troubleshooting Clinic,Farah Ali,36,21,10:30,2026-08-28\n"
            "Serverless Patterns Lab,Joel Fernandes,28,3,16:00,2026-09-12\n"
            "Data Engineering Pipelines,Sana Joseph,45,45,09:30,2026-09-05\n"
            "Prompt Engineering for Teams,Ishaan Bhat,60,14,12:00,2026-09-18\n"
            "GenAI Security Roundtable,Aditi Sharma,22,22,18:00,2026-10-03\n"
            "MLOps Deployment Sprint,Karan Malhotra,40,11,11:45,2026-10-17\n",
            encoding="utf-8",
        )

    if not BOOKINGS_FILE.exists():
        BOOKINGS_FILE.write_text(
            "booking_id,event_name,participant_name,email,date,time,booked_at\n",
            encoding="utf-8",
        )


def read_events() -> List[Dict[str, str]]:
    """Read all technical events from events.csv."""
    ensure_files_exist()
    with EVENTS_FILE.open("r", newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def write_events(events: List[Dict[str, str]]) -> None:
    """Persist updated event seat counts back to events.csv."""
    fieldnames = ["name", "speaker", "seats", "booked", "time", "date"]
    with EVENTS_FILE.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(events)


def event_status(event_date: str) -> str:
    """Classify an event as past, today/current, or upcoming."""
    event_day = datetime.strptime(event_date, DATE_FORMAT).date()
    today = date.today()
    if event_day < today:
        return "past"
    if event_day == today:
        return "today/current"
    return "upcoming"


def seats_available(event: Dict[str, str]) -> int:
    """Calculate remaining seats for an event."""
    return int(event["seats"]) - int(event["booked"])


def find_event(events: List[Dict[str, str]], event_name: str) -> Tuple[Optional[int], Optional[Dict[str, str]]]:
    """Find an event by exact or case-insensitive partial name."""
    normalized = event_name.strip().lower()
    for index, event in enumerate(events):
        if event["name"].strip().lower() == normalized:
            return index, event

    for index, event in enumerate(events):
        if normalized in event["name"].strip().lower():
            return index, event

    return None, None


def public_event_view(event: Dict[str, str]) -> Dict[str, object]:
    """Return event fields with computed availability and booking policy."""
    status = event_status(event["date"])
    available = seats_available(event)
    return {
        "name": event["name"],
        "speaker": event["speaker"],
        "seats": int(event["seats"]),
        "booked": int(event["booked"]),
        "available": available,
        "time": event["time"],
        "date": event["date"],
        "status": status,
        "booking_allowed": status == "upcoming" and available > 0,
    }


def smtp_settings() -> Dict[str, object]:
    """Read GoDaddy SMTP settings from environment variables."""
    return {
        "sender": os.environ.get("EVENT_EMAIL_SENDER", "cloudadmin@vcloudlabs.in"),
        "password": os.environ.get("EVENT_EMAIL_PASSWORD", ""),
        "host": os.environ.get("EVENT_SMTP_HOST", DEFAULT_SMTP_HOST),
        "port": int(os.environ.get("EVENT_SMTP_PORT", str(DEFAULT_SMTP_PORT))),
        "use_ssl": os.environ.get("EVENT_SMTP_SSL", "true").strip().lower()
        in {"1", "true", "yes", "y"},
    }


def decrypted_email_template() -> str:
    """Read and decrypt the booking email template from environment variables."""
    encrypted_template = os.environ.get("EVENT_EMAIL_TEMPLATE_ENCRYPTED", "").strip()
    encryption_key = os.environ.get("EVENT_EMAIL_TEMPLATE_KEY", "").strip()
    if not encrypted_template:
        return DEFAULT_EMAIL_TEMPLATE
    if not encryption_key:
        raise RuntimeError("EVENT_EMAIL_TEMPLATE_KEY is missing.")

    try:
        return Fernet(encryption_key.encode("utf-8")).decrypt(
            encrypted_template.encode("utf-8")
        ).decode("utf-8")
    except (InvalidToken, ValueError) as exc:
        raise RuntimeError("EVENT_EMAIL_TEMPLATE_ENCRYPTED could not be decrypted.") from exc


def booking_email_body(booking: Dict[str, object]) -> str:
    """Render the configured booking email template with booking details."""
    try:
        return decrypted_email_template().format(**booking)
    except KeyError as exc:
        missing_field = str(exc).strip("'")
        raise RuntimeError(
            f"Email template references unknown booking field: {missing_field}"
        ) from exc


def send_booking_email(booking: Dict[str, object]) -> str:
    """Send a booking confirmation email to the participant through SMTP."""
    load_env()
    settings = smtp_settings()
    sender = str(settings["sender"])
    password = str(settings["password"])
    if not password:
        return "Email not sent because EVENT_EMAIL_PASSWORD is missing."

    message = EmailMessage()
    message["Subject"] = f"Booking confirmed: {booking['event_name']}"
    message["From"] = sender
    message["To"] = str(booking["email"])
    message.set_content(booking_email_body(booking))

    try:
        if bool(settings["use_ssl"]):
            with smtplib.SMTP_SSL(str(settings["host"]), int(settings["port"]), timeout=20) as smtp:
                smtp.login(sender, password)
                smtp.send_message(message)
        else:
            with smtplib.SMTP(str(settings["host"]), int(settings["port"]), timeout=20) as smtp:
                smtp.starttls()
                smtp.login(sender, password)
                smtp.send_message(message)
    except Exception as exc:
        return f"Email could not be sent: {exc}"

    return f"Confirmation email sent to {booking['email']}."


@tool
def list_events(status: str = "all") -> str:
    """List technical events. Status can be all, past, today, current, or upcoming."""
    wanted = status.strip().lower()
    if wanted == "current":
        wanted = "today"

    results = []
    for event in read_events():
        details = public_event_view(event)
        event_state = str(details["status"])
        if wanted == "all" or wanted in event_state:
            results.append(details)

    if not results:
        return f"No events found for status: {status}"

    return json.dumps(results, indent=2)


@tool
def get_event_details(event_name: str) -> str:
    """Get details, availability, and booking eligibility for one event."""
    _, event = find_event(read_events(), event_name)
    if event is None:
        return f"No event found matching '{event_name}'."

    return json.dumps(public_event_view(event), indent=2)


@tool
def book_event(event_name: str, participant_name: str, email: str) -> str:
    """Book one seat for a future event using participant name and email."""
    events = read_events()
    event_index, event = find_event(events, event_name)
    if event is None or event_index is None:
        return f"No event found matching '{event_name}'."

    details = public_event_view(event)
    if details["status"] != "upcoming":
        return "Booking is not allowed for past events or events happening today."

    if int(details["available"]) <= 0:
        return "Booking is not allowed because all seats are already booked."

    booked_count = int(event["booked"]) + 1
    event["booked"] = str(booked_count)
    events[event_index] = event
    write_events(events)

    booking_id = f"BKG-{datetime.now().strftime('%Y%m%d%H%M%S')}"
    with BOOKINGS_FILE.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "booking_id",
                "event_name",
                "participant_name",
                "email",
                "date",
                "time",
                "booked_at",
            ],
        )
        writer.writerow(
            {
                "booking_id": booking_id,
                "event_name": event["name"],
                "participant_name": participant_name,
                "email": email,
                "date": event["date"],
                "time": event["time"],
                "booked_at": datetime.now().isoformat(timespec="seconds"),
            }
        )

    confirmation = {
        "message": "Booking confirmed.",
        "booking_id": booking_id,
        "event_name": event["name"],
        "speaker": event["speaker"],
        "participant_name": participant_name,
        "email": email,
        "date": event["date"],
        "time": event["time"],
        "booked": booked_count,
        "seats": int(event["seats"]),
        "available": int(event["seats"]) - booked_count,
    }
    confirmation["email_status"] = send_booking_email(confirmation)
    return json.dumps(confirmation, indent=2)


TOOLS = [list_events, get_event_details, book_event]
TOOLS_BY_NAME = {tool_item.name: tool_item for tool_item in TOOLS}


def load_history() -> List:
    """Load previous chat history for memory across console restarts."""
    if not HISTORY_FILE.exists():
        return []

    raw_messages = json.loads(HISTORY_FILE.read_text(encoding="utf-8"))
    messages = []
    for item in raw_messages:
        if item["type"] == "human":
            messages.append(HumanMessage(content=item["content"]))
        elif item["type"] == "ai":
            messages.append(AIMessage(content=item["content"]))
    return messages[-20:]


def save_history(messages: List) -> None:
    """Save compact human/assistant history for the next run."""
    serializable = []
    for message in messages[-20:]:
        if isinstance(message, HumanMessage):
            serializable.append({"type": "human", "content": message.content})
        elif isinstance(message, AIMessage) and isinstance(message.content, str):
            serializable.append({"type": "ai", "content": message.content})

    HISTORY_FILE.write_text(json.dumps(serializable, indent=2), encoding="utf-8")


def system_prompt() -> SystemMessage:
    """Agent instructions that define event policy and tool usage."""
    return SystemMessage(
        content=(
            "You are a technical events booking agent. Use the provided tools for event data "
            "and booking; do not invent seat counts or dates. Explain past, current/today, "
            "and upcoming events clearly. Booking is allowed only for future/upcoming events "
            "with available seats. Booking is never allowed for past events or today's events. "
            "If a participant wants to attend, ask for the event name, participant name, and "
            "email if any are missing. Confirm booking only after the book_event tool succeeds."
        )
    )


def run_agent_turn(llm_with_tools, history: List, user_text: str) -> AIMessage:
    """Run one tool-calling agent turn using LangChain messages and history."""
    messages = [system_prompt(), *history, HumanMessage(content=user_text)]
    ai_message = llm_with_tools.invoke(messages)
    messages.append(ai_message)

    while ai_message.tool_calls:
        for tool_call in ai_message.tool_calls:
            selected_tool = TOOLS_BY_NAME[tool_call["name"]]
            tool_result = selected_tool.invoke(tool_call["args"])
            messages.append(
                ToolMessage(
                    content=str(tool_result),
                    tool_call_id=tool_call["id"],
                    name=tool_call["name"],
                )
            )
        ai_message = llm_with_tools.invoke(messages)
        messages.append(ai_message)

    history.append(HumanMessage(content=user_text))
    history.append(ai_message)
    save_history(history)
    return ai_message


def main() -> None:
    """Start the console event agent."""
    load_env()
    ensure_files_exist()

    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is missing. Add it to .env or your environment.")

    model_name = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
    llm = ChatOpenAI(model=model_name, temperature=0)
    llm_with_tools = llm.bind_tools(TOOLS)
    history = load_history()

    print("Technical Events Agent")
    print("Ask about past, today's/current, or upcoming events. Type 'exit' to quit.")
    print("Booking needs event name, participant name, and email.\n")

    while True:
        user_text = input("You: ").strip()
        if user_text.lower() in {"exit", "quit", "bye"}:
            print("Agent: Thanks. Your chat history has been saved.")
            break
        if not user_text:
            continue

        try:
            response = run_agent_turn(llm_with_tools, history, user_text)
            print(f"Agent: {response.content}\n")
        except Exception as exc:
            print(f"Agent: Sorry, something went wrong: {exc}\n")


if __name__ == "__main__":
    main()



