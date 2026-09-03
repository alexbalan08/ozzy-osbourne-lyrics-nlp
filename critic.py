
import sys
from functools import lru_cache
from typing import Literal

import torch
from pydantic import BaseModel, Field, ValidationError
from transformers import AutoModelForCausalLM, AutoTokenizer

from retrieval import Retriever

MODEL = "meta-llama/Llama-3.1-8B-Instruct"
MAX_NEW_TOKENS = 400
TOP_K = 3

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


class Verdict(BaseModel):
    """The structured judgement the model must return."""

    verdict: Literal["SUPPORTED", "CONTRADICTED", "UNVERIFIABLE"]
    evidence: str = Field(description="The exact line from a card that decides it")
    explanation: str = Field(description="One or two sentences of reasoning")
    correction: str | None = Field(
        default=None, description="The correct fact, when the claim is contradicted"
    )


@lru_cache(maxsize=1)
def _clients():
    """Retriever, tokenizer and model, built once and reused."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required to run the local critic model (none detected).")

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    return Retriever(), tokenizer, model


def build_context(claim, k=TOP_K):
    """Retrieve the album cards that should decide this claim."""
    retriever, _, _ = _clients()
    hits = retriever.search_kb(claim, k=k)
    cards = "\n\n".join(
        f"--- ALBUM CARD {i} (relevance {h['score']:.3f}) ---\n{h['card']}"
        for i, h in enumerate(hits, 1)
    )
    return cards, hits


def _extract_json(text):
    """Pull the first top-level {...} object out of raw model text."""
    text = text.strip().strip("`")
    if text.startswith("json"):
        text = text[4:]
    start = text.find("{")
    if start == -1:
        raise ValueError(f"no JSON object in model output: {text!r}")
    depth = 0
    for i, ch in enumerate(text[start:], start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    raise ValueError(f"unterminated JSON object in model output: {text!r}")


def _generate(tokenizer, model, messages):
    """Run one chat turn through the model and decode only the new tokens."""
    inputs = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
    ).to(model.device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    reply_ids = output_ids[0, inputs["input_ids"].shape[1] :]
    return tokenizer.decode(reply_ids, skip_special_tokens=True)


def check_claim(claim, k=TOP_K):
    """Fact-check one claim. Returns (Verdict, retrieved hits)."""
    cards, hits = build_context(claim, k)
    _, tokenizer, model = _clients()

    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": f"{cards}\n\n--- CLAIM ---\n{claim}"},
    ]
    reply = _generate(tokenizer, model, messages)
    try:
        verdict = Verdict.model_validate_json(_extract_json(reply))
    except (ValueError, ValidationError):
        # 8B models occasionally wrap the JSON in prose despite instructions --
        # give the model one chance to see its own mistake and fix it.
        messages += [{"role": "assistant", "content": reply}, {"role": "user", "content": REPAIR}]
        reply = _generate(tokenizer, model, messages)
        verdict = Verdict.model_validate_json(_extract_json(reply))

    return verdict, hits


DEMO_CLAIMS = [
    # true
    "Crazy Train is on Blizzard of Ozz.",
    # wrong year -- the card should contradict it
    "Bark at the Moon was released in 1985.",
    # the KB deliberately has no day for this one
    "Diary of a Madman was released on 7 November 1981.",
    # a Sabbath song, absent from the studio KB
    "War Pigs is a track on Ozzy's album Blizzard of Ozz.",
    # covers-album flag
    "Under Cover is an album of songs written by other artists.",
]


def main():
    claims = sys.argv[1:] or DEMO_CLAIMS
    for claim in claims:
        print(f"\n{'=' * 70}\nCLAIM: {claim}")
        try:
            verdict, hits = check_claim(claim)
        except RuntimeError as exc:
            sys.exit(str(exc))
        except torch.cuda.OutOfMemoryError:
            sys.exit(f"Out of GPU memory running {MODEL}.")
        except OSError as exc:
            sys.exit(
                f"Could not load {MODEL}: {exc}\n"
                "If this is a gated-repo access error, run `huggingface-cli login` "
                "with an account that has accepted the Llama 3.1 license."
            )
        except (ValueError, ValidationError) as exc:
            print(f"  -> could not parse a verdict from the model: {exc}")
            continue

        print(f"retrieved: {', '.join(h['album'] for h in hits)}")
        print(f"  -> {verdict.verdict}")
        print(f"     evidence:    {verdict.evidence}")
        print(f"     explanation: {verdict.explanation}")
        if verdict.correction:
            print(f"     correction:  {verdict.correction}")


if __name__ == "__main__":
    main()
