"""Team chat for Experiment 2: messages are pushed to the recipient with its next input.

A message posted during lockstep step s is delivered from step s+1 on, so a reply always has a
step to arrive in. Every post and every delivery is logged, so exposure is known per call.
"""

import prompts

ALL = "all"


class TeamChat:
    """The session's messages, who has received which, and in which step."""

    def __init__(self, members: tuple[str, ...]) -> None:
        self.members = members
        self.step = 0  # the scheduler's current lockstep step
        self.messages: list[dict] = []  # sender, to, text, step
        self.deliveries: list[dict] = []  # recipient, message index, step

    def post(self, sender: str, to: str, text: str) -> bool:
        """Record a message to a member or "all"; False if the recipient is unknown."""
        if to != ALL and (to not in self.members or to == sender):
            return False
        self.messages.append({"sender": sender, "to": to, "text": text, "step": self.step})
        return True

    def deliver(self, recipient: str) -> list[str]:
        """Chat lines for messages posted before this step and not yet given to recipient."""
        given = {d["message"] for d in self.deliveries if d["recipient"] == recipient}
        lines = []
        for index, message in enumerate(self.messages):
            addressed = message["to"] == recipient or (
                message["to"] == ALL and message["sender"] != recipient
            )
            if addressed and message["step"] < self.step and index not in given:
                self.deliveries.append(
                    {"recipient": recipient, "message": index, "step": self.step}
                )
                lines.append(
                    prompts.CHAT_LINE.format(sender=message["sender"], text=message["text"])
                )
        return lines
