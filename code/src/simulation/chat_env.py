"""Multi-message chat simulation (Goal 1: rebuild biased scenarios)."""

from __future__ import annotations

from dataclasses import dataclass, field

VALID_ROLES = {"system", "user", "assistant"}
VALID_CHANNELS = {"dialogue", "environment"}


@dataclass
class VtuberChatEnv:
    """
    Holds rolling chat history and system prompt for Neuro-style streaming setups.

    The baseline experiment uses a simple role-tagged chat transcript so that the
    same history can be sent to Ollama and logged in a human-readable form.
    """

    system_prompt: str = ""
    messages: list[dict[str, str]] = field(default_factory=list)
    channel_messages: dict[str, list[dict[str, str]]] = field(
        default_factory=lambda: {channel: [] for channel in VALID_CHANNELS}
    )

    def add_message(self, role: str, content: str, *, channel: str = "dialogue") -> None:
        """Append a validated message to the rolling chat history."""
        normalized_role = role.strip().lower()
        if normalized_role not in VALID_ROLES:
            raise ValueError(f"Unsupported role: {role!r}")
        normalized_channel = channel.strip().lower()
        if normalized_channel not in VALID_CHANNELS:
            raise ValueError(f"Unsupported channel: {channel!r}")

        message = {
            "role": normalized_role,
            "content": content.strip(),
            "channel": normalized_channel,
        }
        self.messages.append(message)
        self.channel_messages[normalized_channel].append(message)

    def extend_messages(
        self, messages: list[dict[str, str]], *, channel: str = "dialogue"
    ) -> None:
        """Append multiple messages in order."""
        for message in messages:
            self.add_message(
                message["role"],
                message["content"],
                channel=message.get("channel", channel),
            )

    def render_messages(
        self,
        *,
        include_dialogue: bool = True,
        include_environment: bool = True,
    ) -> list[dict[str, str]]:
        """Return structured messages in the format expected by chat backends."""
        allowed_channels = set()
        if include_dialogue:
            allowed_channels.add("dialogue")
        if include_environment:
            allowed_channels.add("environment")

        rendered: list[dict[str, str]] = []
        if self.system_prompt.strip():
            rendered.append({"role": "system", "content": self.system_prompt.strip()})
        rendered.extend(
            {
                "role": message["role"],
                "content": message["content"],
            }
            for message in self.messages
            if message["channel"] in allowed_channels
        )
        return rendered

    def render_prompt(self) -> str:
        """Return a readable role-tagged transcript for debugging and saved results."""
        lines: list[str] = []
        if self.system_prompt.strip():
            lines.append("[SYSTEM]")
            lines.append(self.system_prompt.strip())
            lines.append("")

        for message in self.messages:
            lines.append(f"[{message['channel'].upper()}:{message['role'].upper()}]")
            lines.append(message["content"])
            lines.append("")

        return "\n".join(lines).strip()
