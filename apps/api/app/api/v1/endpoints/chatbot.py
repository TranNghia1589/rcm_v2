from __future__ import annotations

# Backward-compat wrapper. Canonical implementation lives in apps.api.app.api.v1.chatbot.
from apps.api.app.api.v1.chatbot import ask_chatbot, ask_chatbot_debug, get_chat_service, router

__all__ = ["router", "get_chat_service", "ask_chatbot", "ask_chatbot_debug"]
