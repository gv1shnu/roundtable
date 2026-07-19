"""Verify context_mode logic with a stub LLM (no real model calls)."""
from multivoice.config import load_config
from multivoice.orchestrator import Orchestrator
from multivoice.transcript import Transcript


class StubLLM:
    """Records the prompt each participant receives; returns a tagged reply."""
    def __init__(self):
        self.seen = {}

    def generate(self, model, system, prompt, temperature=0.8):
        self.seen[model] = prompt
        return f"REPLY_FROM[{model}]"


def run(mode):
    cfg = load_config("config.yaml")
    cfg.conversation.context_mode = mode
    llm = StubLLM()
    tr = Transcript()
    tr.add("User", "Opening topic.")
    Orchestrator(cfg, llm, tr).round()
    first, second = cfg.participants[0], cfg.participants[1]
    # Did the 2nd agent's prompt contain the 1st agent's reply?
    saw_first = f"REPLY_FROM[{first.model}]" in llm.seen[second.model]
    return saw_first


conv = run("conversation")
query = run("query")
print(f"conversation: 2nd agent saw 1st agent's reply? {conv}   (expect True)")
print(f"query:        2nd agent saw 1st agent's reply? {query}   (expect False)")
assert conv is True, "conversation mode should share within the round"
assert query is False, "query mode should hide same-round replies"
print("[modes OK]")
