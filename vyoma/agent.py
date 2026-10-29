"""Agentic controller: conversation state, RAG grounding and tool calling."""

import json
import logging

import ollama

from . import events
from .kb import KnowledgeBase
from .tools import TOOL_SCHEMAS, Tools

logger = logging.getLogger(__name__)


def _looks_like_tool_json(content: str | None) -> bool:
    if not content or not content.lstrip().startswith("{"):
        return False
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        return False
    return isinstance(data, dict) and "name" in data


SYSTEM_PROMPT = """\
You are {assistant_name}, the receptionist at {organisation}. You speak with \
visitors at the front desk. Your replies are spoken aloud, so keep them to \
one or two short, warm, natural sentences. Never use lists, markdown or \
emojis.

Your duties:
- Greet visitors and answer their questions using the FACTS provided to you.
- If a visitor has come to meet someone, ask their name and purpose if you \
don't know it yet, then use the capture_visitor, log_visit and notify_staff \
tools.
- If you don't know an answer and the FACTS don't cover it, say so honestly \
and offer to notify a staff member.
- When the visitor says goodbye or is done, call the end_conversation \
tool and reply with a short goodbye. Never say the words "end \
conversation" to the visitor.

Only state facts about {organisation} that appear in the FACTS context or \
in the organisation details below. Do not invent staff names, timings or \
directions.{identity}\
"""


def _identity_block(org: dict) -> str:
    """Contact details from config, so basics like address and phone are
    always answerable even when retrieval misses."""
    labels = (("address", "Address"), ("phone", "Phone"),
              ("email", "Email"), ("website", "Website"))
    lines = [f"- {label}: {org[key]}"
             for key, label in labels if org.get(key)]
    if not lines:
        return ""
    return "\n\nOrganisation details (you may state these):\n" + \
        "\n".join(lines)


class ReceptionistAgent:
    def __init__(self, cfg: dict, kb: KnowledgeBase, tools: Tools):
        self.cfg = cfg["llm"]
        self.kb = kb
        self.tools = tools
        self.client = ollama.Client(host=self.cfg["host"])
        self.model = self._select_model()
        org = cfg["organisation"]
        self.org_name = org["name"]
        self.assistant_name = org["assistant_name"]
        self.messages: list[dict] = [{
            "role": "system",
            "content": SYSTEM_PROMPT.format(
                assistant_name=org["assistant_name"],
                organisation=org["name"],
                identity=_identity_block(org),
            ),
        }]

    def _select_model(self) -> str:
        available = {m.model for m in self.client.list().models}
        for candidate in (self.cfg["model"], self.cfg["fallback_model"]):
            if candidate in available or f"{candidate}:latest" in available:
                logger.info("Using LLM: %s", candidate)
                # Load the model into VRAM now and keep it resident
                self.client.generate(model=candidate,
                                     keep_alive=self.cfg["keep_alive"])
                return candidate
        raise RuntimeError(
            f"Neither '{self.cfg['model']}' nor '{self.cfg['fallback_model']}' "
            "is available in Ollama. Run: ollama pull " + self.cfg["model"]
        )

    def greet(self, visitor_name: str | None) -> str:
        if visitor_name:
            self.messages.append({
                "role": "system",
                "content": f"A returning visitor was recognised: {visitor_name}.",
            })
            return f"Welcome back, {visitor_name}! How can I help you today?"
        return (f"Hello, welcome to {self.org_name}. "
                f"I'm {self.assistant_name}. How can I help you?")

    def respond(self, user_text: str) -> str:
        """One conversation turn: RAG context + LLM + tool loop."""
        facts = self.kb.search(user_text)
        events.emit("facts", count=len(facts),
                    previews=[f.replace("\n", " ")[:150] for f in facts])
        content = user_text
        if facts:
            context = "\n---\n".join(facts)
            content = f"FACTS:\n{context}\n\nVISITOR SAYS: {user_text}"
        self.messages.append({"role": "user", "content": content})

        for _ in range(5):  # bounded tool loop
            resp = self.client.chat(
                model=self.model,
                messages=self.messages,
                tools=TOOL_SCHEMAS,
                keep_alive=self.cfg["keep_alive"],
                options=self.cfg.get("options", {}),
            )
            msg = resp.message
            self.messages.append(msg.model_dump(exclude_none=True))

            if not msg.tool_calls:
                if _looks_like_tool_json(msg.content):
                    # Small models sometimes emit a made-up tool call as
                    # plain JSON text; push back instead of speaking it.
                    logger.info("Discarding hallucinated tool JSON: %s",
                                msg.content)
                    self.messages.append({
                        "role": "system",
                        "content": "That tool does not exist. Answer the "
                                   "visitor directly in plain words using "
                                   "the FACTS.",
                    })
                    continue
                if msg.content and msg.content.strip().lower().rstrip(".!") \
                        in ("end conversation", "end_conversation"):
                    # Model narrated the tool instead of calling it
                    self.tools.end_conversation()
                    return "Thank you for visiting. Have a great day!"
                return msg.content or ""

            for call in msg.tool_calls:
                name = call.function.name
                args = call.function.arguments or {}
                logger.info("Tool call: %s(%s)", name, args)
                events.emit("tool", name=name, args=dict(args))
                result = self.tools.execute(name, dict(args))
                self.messages.append({
                    "role": "tool",
                    "content": result,
                    "tool_name": name,
                })

        logger.warning("Tool loop limit reached")
        return "Sorry, could you say that again?"

    @property
    def finished(self) -> bool:
        return self.tools.conversation_ended
