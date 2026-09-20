"""Fact-check claims about Ozzy's albums, using the album cards as the only source of truth.

Run it with no arguments to check the demo claims, or pass your own claims:

    python critic.py "Crazy Train is on Blizzard of Ozz."
"""

import json
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from retrieval import search

MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
MAX_NEW_TOKENS = 400  # the answer is a small JSON object, this is plenty of room
TOP_K = 3  # how many album cards to show the model

SYSTEM = """You fact-check claims about Ozzy Osbourne's discography.

You will be given a claim and one or more ALBUM CARDS retrieved from a verified
knowledge base built from Wikipedia. The cards are your ONLY source of truth.

Rules:
- Judge the claim strictly against the cards. Do not use anything you know about
  Ozzy Osbourne from training. If the cards do not settle the claim, the verdict
  is UNVERIFIABLE, even if you are confident you know the answer.
- SUPPORTED: the cards confirm the claim.
- CONTRADICTED: the cards state something incompatible with the claim.
- UNVERIFIABLE: the cards do not address it, or are not precise enough to decide.
- A card whose release date reads "month-level only" genuinely lacks a day. A
  claim about that album's exact day is UNVERIFIABLE, not CONTRADICTED.
- The knowledge base holds original studio-album tracklists only. A song missing
  from the cards may still exist on a live album or compilation, so absence is
  UNVERIFIABLE rather than CONTRADICTED.
- Quote the exact line you relied on in `evidence`. If nothing applies, say so.

Respond with ONLY a single JSON object, no markdown fences and no text before or
after it, matching exactly this schema:

{"verdict": "SUPPORTED" | "CONTRADICTED" | "UNVERIFIABLE",
 "evidence": "<the exact line from a card that decides it>",
 "explanation": "<one or two sentences of reasoning>",
 "correction": "<the correct fact, only if CONTRADICTED, else null>"}"""

REPAIR = (
    "That was not a single valid JSON object matching the schema. "
    "Reply with ONLY the corrected JSON object, no other text."
)

DEMO_CLAIMS = [
    "Crazy Train is on Blizzard of Ozz.",  # true
    "Bark at the Moon was released in 1985.",  # wrong year, the card should contradict it
    "Diary of a Madman was released on 7 November 1981.",  # the KB deliberately has no day for this one
    "War Pigs is a track on Ozzy's album Blizzard of Ozz.",  # a Sabbath song, not on that tracklist
    "Under Cover is an album of songs written by other artists.",  # covers-album flag
]

# Loaded once, when this file starts. Needs a CUDA GPU with ~16 GB free. Takes ~15 seconds.
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.bfloat16, device_map="cuda")


def ask_model(messages):
    """Send a chat (list of {"role", "content"} dicts) to Llama and return its reply as text."""
    inputs = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
    ).to("cuda")
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,  # always pick the most likely token, so the same input gives the same answer
            pad_token_id=tokenizer.eos_token_id,
        )
    new_tokens = output[0, inputs["input_ids"].shape[1]:]  # only what the model added
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


def parse_verdict(reply):
    """Pull the JSON object out of the model's reply. Returns None if there isn't a usable one."""
    start = reply.find("{")
    end = reply.rfind("}")
    if start == -1 or end == -1:
        return None
    try:
        verdict = json.loads(reply[start:end + 1])
    except json.JSONDecodeError:
        return None
    if verdict.get("verdict") not in ("SUPPORTED", "CONTRADICTED", "UNVERIFIABLE"):
        return None
    return verdict


def check_claim(claim):
    """Fact-check one claim. Returns (verdict dict or None, the album cards that were used)."""
    hits = search(claim, "kb", k=TOP_K)

    cards = ""
    for i, hit in enumerate(hits, 1):
        cards += f"--- ALBUM CARD {i} (relevance {hit['score']:.3f}) ---\n{hit['card']}\n\n"

    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": f"{cards}--- CLAIM ---\n{claim}"},
    ]
    reply = ask_model(messages)
    verdict = parse_verdict(reply)

    if verdict is None:
        # Small models sometimes wrap the JSON in prose. Show the model its mistake and ask once more.
        messages.append({"role": "assistant", "content": reply})
        messages.append({"role": "user", "content": REPAIR})
        reply = ask_model(messages)
        verdict = parse_verdict(reply)

    return verdict, hits


if __name__ == "__main__":
    claims = sys.argv[1:]
    if not claims:
        claims = DEMO_CLAIMS

    for claim in claims:
        print()
        print("=" * 70)
        print("CLAIM:", claim)
        verdict, hits = check_claim(claim)

        albums = []
        for hit in hits:
            albums.append(hit["album"])
        print("retrieved:", ", ".join(albums))

        if verdict is None:
            print("  -> the model did not return valid JSON")
            continue
        print("  ->", verdict["verdict"])
        print("     evidence:   ", verdict.get("evidence", ""))
        print("     explanation:", verdict.get("explanation", ""))
        if verdict.get("correction"):
            print("     correction: ", verdict["correction"])
