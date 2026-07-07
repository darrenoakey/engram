# =============================================================================
#  canary_prompts — the fixed behavioral probe set (60 prompts, 12 with answers)
#  why: cumulative drift of the live model (base + overlay, post-consolidation)
#  from the original base is measured on ONE unchanging prompt set; if the set
#  ever changed, stored baselines and live probes would not be comparable. The
#  four categories (coding, tool-call formatting, general knowledge, instruction
#  following) spread the probe across the behaviors updates are most likely to
#  disturb. EXPECTED carries a substring the model must still reproduce; on the
#  0.8B test model — a verbose reasoning distill that spends its first ~60 tokens
#  narrating before answering a factual question — only echo/repeat instructions
#  reliably surface their target within the ≤32-token continuation cap, so the
#  answer-bearing probes are instruction-following echoes (they pass on the 9B
#  too and degrade to garbage under real drift, which is exactly the signal).
# =============================================================================
from __future__ import annotations

_CODING = [
    ("code_01", "Write a Python function that returns the square of a number."),
    ("code_02", "Write a Python function to check if a string is a palindrome."),
    ("code_03", "Write a function to reverse a list in Python."),
    ("code_04", "How do you read a text file line by line in Python?"),
    ("code_05", "Write a Python one-liner to sum a list of numbers."),
    ("code_06", "Write a function that returns the factorial of n."),
    ("code_07", "How do you sort a dictionary by its values in Python?"),
    ("code_08", "Write a Python function to count the vowels in a string."),
    ("code_09", "Write a regular expression that matches an email address."),
    ("code_10", "Write a function to find the maximum value in a list."),
    ("code_11", "How do you remove duplicate items from a list in Python?"),
    ("code_12", "Write a Python function that returns the nth Fibonacci number."),
    ("code_13", "Write a function to convert Celsius to Fahrenheit."),
    ("code_14", "How do you merge two dictionaries in Python?"),
    ("code_15", "Write a Python function to check whether a number is prime."),
]

_TOOL = [
    ("tool_01", "Call a tool to get the current weather in Paris."),
    ("tool_02", "Use a calculator tool to add fifteen and twenty-seven."),
    ("tool_03", "Search the web for the population of Canada."),
    ("tool_04", "Look up today's exchange rate from US dollars to euros."),
    ("tool_05", "Call a tool to set a timer for ten minutes."),
    ("tool_06", "Use a tool to translate 'good morning' into German."),
    ("tool_07", "Query a database for every user named Smith."),
    ("tool_08", "Call a tool to send an email to the support team."),
    ("tool_09", "Fetch the latest headline from a news service."),
    ("tool_10", "Use a tool to find restaurants near Central Park."),
    ("tool_11", "Call a function to convert five kilometers to miles."),
    ("tool_12", "Look up the current stock price of a technology company."),
    ("tool_13", "Use a tool to create a calendar event for tomorrow morning."),
    ("tool_14", "Call a weather tool for the forecast in Tokyo."),
    ("tool_15", "Use a tool to shorten a long web link."),
]

_KNOW = [
    ("know_01", "What is the capital of France?"),
    ("know_02", "Who painted the Mona Lisa?"),
    ("know_03", "What is the largest ocean on Earth?"),
    ("know_04", "How many continents are there on Earth?"),
    ("know_05", "What is the chemical symbol for gold?"),
    ("know_06", "Who wrote the play Romeo and Juliet?"),
    ("know_07", "Roughly how fast does light travel in a vacuum?"),
    ("know_08", "Which planet is known as the Red Planet?"),
    ("know_09", "What is the tallest mountain in the world?"),
    ("know_10", "How many sides does a hexagon have?"),
    ("know_11", "What gas do plants release during photosynthesis?"),
    ("know_12", "Who was the first person to walk on the Moon?"),
    ("know_13", "What is the freezing point of water in Celsius?"),
    ("know_14", "What language is primarily spoken in Brazil?"),
    ("know_15", "What is the smallest prime number?"),
]

# instruction following — the first 12 carry an expected substring (see _ANSWERS)
_FOLLOW = [
    ("follow_01", "Repeat this word exactly: photosynthesis"),
    ("follow_02", "Repeat this word exactly: kangaroo"),
    ("follow_03", "Repeat this word exactly: umbrella"),
    ("follow_04", "Copy this word back to me: telescope"),
    ("follow_05", "Echo the following word: mountain"),
    ("follow_06", "Repeat this exact word: velvet"),
    ("follow_07", "Repeat this word exactly: harbour"),
    ("follow_08", "Repeat this word exactly: crimson"),
    ("follow_09", "Repeat this word exactly: lantern"),
    ("follow_10", "Repeat this word exactly: cascade"),
    ("follow_11", "Repeat this word exactly: gravity"),
    ("follow_12", "Repeat this word exactly: compass"),
    ("follow_13", "List three primary colors, separated by commas."),
    ("follow_14", "Write a short two-line rhyme about the ocean."),
    ("follow_15", "Summarize the water cycle in one sentence."),
]

_ANSWERS = {
    "follow_01": "photosynthesis",
    "follow_02": "kangaroo",
    "follow_03": "umbrella",
    "follow_04": "telescope",
    "follow_05": "mountain",
    "follow_06": "velvet",
    "follow_07": "harbour",
    "follow_08": "crimson",
    "follow_09": "lantern",
    "follow_10": "cascade",
    "follow_11": "gravity",
    "follow_12": "compass",
}


def _as_messages(text: str) -> list[dict]:
    return [{"role": "user", "content": text}]


# =============================================================================
#  probes / expected
#  why: PROBES is the frozen (id, messages) list every baseline and probe walks;
#  EXPECTED maps the 12 answer-bearing ids to the substring the live model must
#  still emit. Both are module constants so the set is identical across restarts.
PROBES: list[tuple[str, list[dict]]] = [
    (pid, _as_messages(text)) for pid, text in (_CODING + _TOOL + _KNOW + _FOLLOW)
]

EXPECTED: dict[str, str] = dict(_ANSWERS)


# =============================================================================
#  select
#  why: baseline/probe take an optional probe subset (fast tests, targeted
#  production probes); this returns the (id, messages) pairs for the given ids
#  in PROBES order, raising loudly on an unknown id rather than silently skipping
def select(ids: list[str]) -> list[tuple[str, list[dict]]]:
    wanted = set(ids)
    chosen = [pair for pair in PROBES if pair[0] in wanted]
    found = {pair[0] for pair in chosen}
    missing = wanted - found
    if missing:
        raise ValueError(f"unknown probe ids: {sorted(missing)}")
    return chosen
