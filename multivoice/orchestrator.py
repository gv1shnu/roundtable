"""Runs the multi-party conversation: turn-taking, prompting, transcript.

Two context modes (set `conversation.context_mode` in config.yaml):

  conversation — sequential. Each agent's prompt includes the earlier agents'
                 replies from THIS round, so they react in real time.
  query        — simultaneous. Every agent answers the same frozen transcript
                 (as it stood at the start of the round) with no view of this
                 round's other answers. All replies are appended together at the
                 end, so they only become visible to everyone next round.
"""
from __future__ import annotations

from .config import Config, Participant
from .llm import LLM
from .transcript import Transcript


class Orchestrator:
    def __init__(self, config: Config, llm: LLM, transcript: Transcript | None = None):
        self.cfg = config
        self.llm = llm
        self.transcript = transcript or Transcript()

    def _system_prompt(self, me: Participant) -> str:
        others = [p.name for p in self.cfg.participants if p.name != me.name]
        others_str = ", ".join(others) if others else "no one else yet"
        return (
            f"{me.persona}\n\n"
            f"You are {me.name}, one voice in a live group conversation with "
            f"{others_str}, and a human called User. Stay fully in character. "
            f"React to what others have said so far — agree, build, or push back. "
            f"Reply ONLY with your own next contribution, in the first person, "
            f"with no name prefix and no stage directions. "
            f"Keep it to at most {self.cfg.conversation.max_reply_sentences} sentences."
        )

    def _prompt_from(self, history: str, me: Participant) -> str:
        body = history if history else "(nothing said yet — you are opening.)"
        return f"Conversation so far:\n{body}\n\nYour turn, {me.name}:"

    def _generate(self, participant: Participant, history: str) -> str:
        return self.llm.generate(
            model=participant.model,
            system=self._system_prompt(participant),
            prompt=self._prompt_from(history, participant),
            temperature=self.cfg.conversation.temperature,
        )

    def one_turn(self, participant: Participant) -> str:
        """Generate against the live transcript and append immediately."""
        history = self.transcript.as_text(self.cfg.conversation.context_turns)
        reply = self._generate(participant, history)
        self.transcript.add(participant.name, reply)
        return reply

    def round(self, on_reply=None) -> None:
        """One full cycle through every participant, honoring context_mode.

        on_reply(participant, text) is called after each turn (for printing/TTS).
        """
        if self.cfg.conversation.context_mode == "query":
            self._round_query(on_reply)
        else:
            self._round_conversation(on_reply)

    def _round_conversation(self, on_reply=None) -> None:
        for participant in self.cfg.participants:
            reply = self.one_turn(participant)  # appended live → next agent sees it
            if on_reply:
                on_reply(participant, reply)

    def _round_query(self, on_reply=None) -> None:
        # Freeze the transcript; everyone answers this same state blindly.
        frozen = self.transcript.as_text(self.cfg.conversation.context_turns)
        pending: list[tuple[Participant, str]] = []
        for participant in self.cfg.participants:
            reply = self._generate(participant, frozen)
            pending.append((participant, reply))
            if on_reply:
                on_reply(participant, reply)
        # Reveal all at once → visible to everyone only from next round on.
        for participant, reply in pending:
            self.transcript.add(participant.name, reply)
