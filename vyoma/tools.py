"""Receptionist tools the LLM can call, plus their Ollama schemas."""

import logging

import requests

from .db import Database

logger = logging.getLogger(__name__)

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "capture_visitor",
            "description": (
                "Register a visitor's details after asking for them. "
                "Call this once you know at least the visitor's name."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Visitor's full name"},
                    "phone": {"type": "string", "description": "Phone number, if given"},
                    "company": {"type": "string", "description": "Company/organisation, if given"},
                },
                "required": ["name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "log_visit",
            "description": (
                "Record this visit in the visitor log. Call when you know "
                "who the visitor is and why they came."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "visitor_name": {"type": "string"},
                    "purpose": {"type": "string", "description": "Reason for the visit"},
                    "whom_to_meet": {"type": "string", "description": "Staff member or department they came to see"},
                },
                "required": ["visitor_name", "purpose"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "notify_staff",
            "description": (
                "Notify a staff member that a visitor is waiting for them "
                "at the reception."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "staff_name": {"type": "string"},
                    "message": {"type": "string", "description": "Short message, e.g. 'Ravi from Acme is waiting at reception'"},
                },
                "required": ["staff_name", "message"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "end_conversation",
            "description": (
                "End the conversation politely. Call when the visitor says "
                "goodbye or has no more questions."
            ),
            "parameters": {"type": "object", "properties": {}},
        },
    },
]


class Tools:
    """Executes tool calls issued by the LLM."""

    def __init__(self, db: Database, webhook_url: str = ""):
        self.db = db
        self.webhook_url = webhook_url
        self.conversation_ended = False

    def execute(self, name: str, args: dict) -> str:
        handler = getattr(self, name, None)
        if handler is None:
            logger.warning("LLM called unknown tool: %s", name)
            return f"Error: unknown tool '{name}'"
        try:
            return handler(**args)
        except TypeError as e:
            logger.warning("Bad tool arguments for %s: %s", name, e)
            return f"Error: invalid arguments for {name}"
        except Exception as e:
            logger.exception("Tool %s failed", name)
            return f"Error: {e}"

    def capture_visitor(self, name: str, phone: str | None = None,
                        company: str | None = None) -> str:
        self.db.upsert_visitor(name=name, phone=phone, company=company)
        return f"Visitor '{name}' registered."

    def log_visit(self, visitor_name: str, purpose: str,
                  whom_to_meet: str | None = None) -> str:
        if not visitor_name or not visitor_name.strip():
            return ("Not logged: visitor name unknown. Just answer the "
                    "question; only log a visit once the visitor has "
                    "given their name.")
        self.db.log_visit(visitor_name, purpose, whom_to_meet)
        return f"Visit by '{visitor_name}' logged."

    def notify_staff(self, staff_name: str, message: str) -> str:
        delivered = False
        if self.webhook_url:
            try:
                requests.post(self.webhook_url,
                              data=f"[{staff_name}] {message}".encode(),
                              timeout=5)
                delivered = True
            except requests.RequestException as e:
                logger.warning("Webhook notification failed: %s", e)
        self.db.add_notification(staff_name, message, delivered)
        via = "webhook" if delivered else "reception log"
        return f"Notification for {staff_name} recorded via {via}."

    def end_conversation(self) -> str:
        self.conversation_ended = True
        return "Conversation marked as finished. Say a short goodbye."
