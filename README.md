# Aura Agent

A small, monitorable harness that rewrites names/phrases to have more "aura"
(internet-slang sense: presence that commands power/respect/controlled
intimidation), with a self-critique pass and full run logging.

## Setup

```bash
#This commented section is if you wanted to use Claude
#and an anthropic API key. Follow the commands below,
#and use the aura_agent.py file listed in the claude
#folder:
#pip install anthropic --break-system-packages
#export ANTHROPIC_API_KEY=sk-...
pip install google-generativeai --break-system-packages
export GEMINI_API_KEY=...
```

## Run it

```bash
# single input, sincere tone (default)
python aura_agent.py "Kevin"

# ironic / meme tone
python aura_agent.py "my study group" --tone ironic

# run the eval set (regression testing your prompt changes)
python aura_agent.py --eval eval_set.jsonl
```

## How it's structured (the "harness")

1. **`generate()`** — one API call. System prompt lists explicit "levers"
   (phonetic weight, rarity, mythic resonance, compression, declarative
   delivery, register mismatch) and asks the model to tag which ones it used
   per candidate, plus a one-line rationale. This turns "sound more aura"
   into an inspectable, structured decision instead of a black box.

2. **`self_critique()`** — a second API call that reviews the raw candidates
   against a fixed rulebook (no stereotype-as-shortcut, no real living
   people, no IP names, capped intimidation, tone-appropriate) and returns
   keep / edit / drop per candidate. This is the automatic editing layer —
   the harness polices itself before you see output.

3. **`_log()`** — every run (input, raw candidates, critique verdicts, final
   output) is appended as one line of JSON to `runs.jsonl`. Nothing is
   thrown away, so you can always see *why* the harness produced what it
   produced.

## Monitoring

Every run is one line in `runs.jsonl`. Useful things to grep/jq for:

```bash
# see every run where a candidate got dropped
jq 'select(.reviewed_candidates[].verdict == "drop")' runs.jsonl

# see which levers get used most often
jq -r '.final_output[].levers_used[]' runs.jsonl | sort | uniq -c | sort -rn

# see every edit the critique pass made (i.e. where it caught something)
jq 'select(.reviewed_candidates[].verdict == "edit")
    | {input: .input_text, raw: .raw_candidates, reviewed: .reviewed_candidates}' runs.jsonl
```

If you don't have `jq`, plain Python works fine too:

```python
import json
runs = [json.loads(l) for l in open("runs.jsonl")]
drops = [r for r in runs if any(c["verdict"] == "drop" for c in r["reviewed_candidates"])]
```

## Editing the harness

The two prompts (`SYSTEM_PROMPT`, `CRITIQUE_SYSTEM_PROMPT`) at the top of
`aura_agent.py` are the whole "behavior" of the agent — editing them is
editing the harness. The recommended loop:

1. Change a prompt.
2. Re-run `python aura_agent.py --eval eval_set.jsonl` and read the output.
3. Compare against previous output (git diff the printed output, or diff
   `runs.jsonl` before/after) to see what actually changed.
4. Add any new tricky case you find to `eval_set.jsonl` so it's covered
   going forward.

This keeps changes deliberate rather than "I tweaked the prompt and it feels
better" — you have a fixed set of cases you check every time.

## Extending it

- **Add more levers**: extend the `LEVERS` list — it's automatically
  rendered into the system prompt.
- **Batch/UI**: `AuraAgent` is a plain class; wrapping it in a small Flask
  app or a Jupyter notebook loop is straightforward since `run()` returns a
  structured `RunRecord`.
- **Automatic eval scoring**: right now review is qualitative (you read the
  output). If you want a number, add a third API call that scores each
  final candidate 1-5 against a rubric and log that score too — then you
  can track average score over eval-set runs as you tune prompts.
