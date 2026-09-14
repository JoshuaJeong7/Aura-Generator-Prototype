#!/usr/bin/env python3
"""
Aura Agent (Gemini edition) — a small, monitorable harness for transforming
names/phrases into higher-"aura" equivalents.

Pipeline:
    1. generate()       -> model produces N candidates, tagging which "levers"
                           it pulled and why (structured JSON).
    2. self_critique()   -> a second pass checks each candidate against
                           guardrails (no stereotype-as-shortcut, no real
                           living people, tone-appropriate, not gore/violence
                           maximalist) and either keeps, edits, or drops it.
    3. log_run()         -> every run (input + full intermediate state) is
                           appended to a JSONL file so you can grep, diff,
                           and eval later.

Usage:
    export GEMINI_API_KEY=...
    python aura_agent_gemini.py "Kevin"
    python aura_agent_gemini.py "Kevin" --tone ironic
    python aura_agent_gemini.py --eval eval_set.jsonl

Requires:
    pip install google-generativeai --break-system-packages
"""

import argparse
import json
import os
import sys
import time
import uuid
import warnings

# Suppress legacy package deprecation warning for clean output
warnings.filterwarnings("ignore", category=FutureWarning)
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Literal

try:
    import google.generativeai as genai
except ImportError:
    print("Missing dependency. Run: pip install google-generativeai --break-system-packages")
    sys.exit(1)

# Configure API key (reads GEMINI_API_KEY or GOOGLE_API_KEY)
api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
if api_key:
    genai.configure(api_key=api_key)

MODEL = "gemini-3.6-flash"
LOG_PATH = Path(__file__).parent / "runs.jsonl"

# ---------------------------------------------------------------------------
# Prompt layer
# ---------------------------------------------------------------------------

LEVERS = [
    ("phonetic_weight", "Hard consonants (k, x, z, g, r), fewer syllables, "
                         "closed/stressed final syllable. Monosyllables and "
                         "plosives read as more commanding than soft, liquid, "
                         "multi-syllable sounds."),
    ("rarity", "Does it feel singular / not heard 500 times today, vs. "
               "extremely common?"),
    ("mythic_resonance", "Echoes of gods, ancient titles, apex predators, "
                          "natural forces (storms, mountains, fire) — "
                          "borrowed association, used sparingly."),
    ("compression", "Shorter reads as more aura. Titles get clipped "
                     "('Big Papi' not 'David Ortiz')."),
    ("declarative_delivery", "No hedging, no qualifiers — the phrase itself "
                              "sounds certain."),
    ("register_mismatch", "Optional, riskier lever: an incongruously grand "
                           "title for something mundane, deployed deadpan. "
                           "Only use this for ironic tone."),
]

SYSTEM_PROMPT = f"""You are a precise linguistic stylist. Your job is to take a \
name or short phrase and produce alternates with more "aura" — internet slang \
(especially among young men on social media) meaning a presence that invokes, \
commands, or implies power, respect, or controlled intimidation, without being \
literally violent or menacing.

You have a fixed set of levers to work with. For each candidate you produce, \
you must identify which levers you pulled:

{chr(10).join(f"- {name}: {desc}" for name, desc in LEVERS)}

Hard rules:
1. Never use ethnic, national, or cultural stereotypes as a shortcut to "power" \
or "intimidation" (e.g. reaching for foreign-sounding or "exotic" phonetics to \
signal danger). Use the phonetic/mythic/compression levers instead. \
Different linguistic, cultural, or ethnic references while using a \
mythic lever will be allowed, however you must also specify in your \
rationale (as described in the output format below) that a reference \
to a different linguistic, cultural, national, or ethnic group was used during formation, \
as well as why it was used for this candidate. There is zero-tolerance for using \
slurs, derogatory remarks, or harmful/generalizing stereotypes about a used \
language, culture, national, or ethnic group, and any such names that use said \
slurs, derogatory remarks, or harmful/generalizing stereotypes must be dropped.
2. Do not produce output that reads as mocking, demeaning, or a slur, even \
disguised.
3. Do not reference real, currently-living public figures by name as the \
output itself (comparisons in your rationale are fine). \
Additionally, do not reference real, dead public figures if their circumstance is \
controversial, divisive, or sensitive in any other way. \
Example: Charlie Kirk or Adolf Hitler will not be tolerated as a reference \
in ANY way and must be dropped immediately.
4. Do not use real trademarked/IP character names as the output unless the \
user explicitly asked for that kind of "which fictional X are you" mapping. \
If the user is specifically asking to rewrite a name that comes from a specific \
trademarked/IP franchise, then the output is allowed to use trademarked/IP characters\
or terms from the same exact trademark/IP as the input.
5. Cap intimidation at "commands respect / a little fear" — never gore, \
violence, or hate-adjacent imagery, regardless of tone setting.
6. "Ironic" tone means playfully, self-aware grandiose (aura-farming meme \
energy) — NOT mocking of the user or a third party.

Always respond with ONLY valid JSON matching this shape, no markdown fences, \
no preamble:

{{
  "input": "<original text>",
  "tone": "sincere" | "ironic",
  "candidates": [
    {{
      "text": "<the new name/phrase>",
      "levers_used": ["<lever_name>", ...],
      "rationale": "<one sentence, plain language, no jargon dump>"
    }}
  ]
}}

Produce exactly 3 candidates, each pulling a different primary lever so they \
feel meaningfully different from each other, not just synonyms \
You are free to also output up to 7 failed candidates that almost met every rule, \
with the addition of [FAILED] as a prefix to the text parameter, and an additional \
reason in the rationale parameter to explain why the candidate failed. \
Otherwise, output only the 3 successful candidates.
"""

CRITIQUE_SYSTEM_PROMPT = """You are a strict reviewer checking candidate outputs \
from an "aura name" generator against a fixed rulebook. For each candidate, \
decide: "keep", "edit", or "drop".

Rules being enforced:
1. No ethnic/national/cultural stereotyping used as a shortcut for power or \
danger.
2. Not mocking, demeaning, or a disguised slur.
3. Not a real, currently-living public figure's name used as the output. \
    If it is a real but dead public figure, their name should not be controversial \
    or insensitive.
4. Not real trademarked/IP names (unless the run's tone/context explicitly \
calls for that, or the input is from a trademark/IP and the output maintains \
the same trademark/IP).
5. Not gore, violence, or hate-adjacent imagery — intimidation should read as \
"commands respect", not "threatens harm".
6. If tone is "ironic", the candidate should read as playful self-aware \
grandiosity, not mockery of anyone.

If "edit", provide a corrected "text" that fixes the issue while preserving \
the original intent/lever as much as possible. If "drop", explain briefly why \
no safe edit was reasonable.

Respond with ONLY valid JSON, no markdown fences:

{
  "reviewed": [
    {
      "original_text": "<candidate text as given>",
      "verdict": "keep" | "edit" | "drop",
      "final_text": "<text to actually use, or null if dropped>",
      "note": "<short reason, only needed for edit/drop>"
    }
  ]
}
"""

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class RunRecord:
    run_id: str
    timestamp: float
    input_text: str
    tone: str
    raw_candidates: list
    reviewed_candidates: list
    final_output: list
    model: str = MODEL


# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------

class AuraAgent:
    def __init__(self, model: str = MODEL, log_path: Path = LOG_PATH):
        self.model_name = model
        self.log_path = log_path

    def _call(self, system: str, user: str, max_tokens: int = 1500) -> dict:
        # Check API key before making the call
        if not os.environ.get("GEMINI_API_KEY") and not os.environ.get("GOOGLE_API_KEY"):
            raise ValueError(
                "API key not found. Please set either GEMINI_API_KEY or GOOGLE_API_KEY."
            )

        model = genai.GenerativeModel(
            model_name=self.model_name,
            system_instruction=system,
        )
        resp = model.generate_content(
            contents=user,
            generation_config={
                "max_output_tokens": max_tokens,
                "response_mime_type": "application/json",
            },
        )
        text = resp.text.strip()
        # defensive: strip accidental code fences if any
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            print(f"JSON decode error: {e}")
            print(f"Raw text next line::\n {text}\n-----Exiting Program-----")
            raise

    def generate(self, input_text: str, tone: Literal["sincere", "ironic"] = "sincere") -> dict:
        user_msg = f'Input: "{input_text}"\nTone: {tone}'
        return self._call(SYSTEM_PROMPT, user_msg)

    def self_critique(self, generation: dict) -> dict:
        user_msg = json.dumps({
            "tone": generation["tone"],
            "candidates": generation["candidates"],
        })
        return self._call(CRITIQUE_SYSTEM_PROMPT, user_msg, max_tokens=1500)

    def run(self, input_text: str, tone: Literal["sincere", "ironic"] = "sincere", verbose: bool = True) -> RunRecord:
        gen = self.generate(input_text, tone)
        review = self.self_critique(gen)

        final = []
        for cand, rev in zip(gen["candidates"], review["reviewed"]):
            if rev["verdict"] == "drop":
                if verbose:
                    print(f'  [dropped] "{cand["text"]}" — {rev.get("note", "")}')
                continue
            text = rev["final_text"] if rev["verdict"] == "edit" else cand["text"]
            final.append({
                "text": text,
                "levers_used": cand["levers_used"],
                "rationale": cand["rationale"],
                "verdict": rev["verdict"],
            })

        record = RunRecord(
            run_id=str(uuid.uuid4())[:8],
            timestamp=time.time(),
            input_text=input_text,
            tone=tone,
            raw_candidates=gen["candidates"],
            reviewed_candidates=review["reviewed"],
            final_output=final,
            model=self.model_name,
        )
        self._log(record)
        return record

    def _log(self, record: RunRecord):
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.log_path, "a") as f:
            f.write(json.dumps(asdict(record)) + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def print_result(record: RunRecord):
    print(f'\n"{record.input_text}"  (tone: {record.tone})')
    print("-" * 50)
    for c in record.final_output:
        tag = f" [{c['verdict']}]" if c["verdict"] != "keep" else ""
        print(f'  -> {c["text"]}{tag}')
        print(f'     levers: {", ".join(c["levers_used"])}')
        print(f'     why: {c["rationale"]}')
    print()


def run_eval(agent: AuraAgent, eval_path: Path):
    with open(eval_path) as f:
        cases = [json.loads(line) for line in f if line.strip()]
    print(f"Running {len(cases)} eval cases...\n")
    for case in cases:
        record = agent.run(case["input"], case.get("tone", "sincere"), verbose=False)
        print_result(record)


def main():
    parser = argparse.ArgumentParser(description="Aura Agent (Gemini) — name/phrase aura transformer")
    parser.add_argument("input", nargs="?", help="Name or phrase to transform")
    parser.add_argument("--tone", choices=["sincere", "ironic"], default="sincere")
    parser.add_argument("--eval", metavar="FILE", help="Run a JSONL eval set instead")
    parser.add_argument("--log", metavar="FILE", default=str(LOG_PATH), help="Log file path")
    parser.add_argument("--model", metavar="MODEL", default=MODEL, help=f"Gemini model (default: {MODEL})")
    args = parser.parse_args()

    agent = AuraAgent(model=args.model, log_path=Path(args.log))

    if args.eval:
        run_eval(agent, Path(args.eval))
        return

    if not args.input:
        parser.error("Provide an input phrase, or use --eval FILE")

    record = agent.run(args.input, args.tone)
    print_result(record)
    print(f"(logged to {agent.log_path})")


if __name__ == "__main__":
    main()
