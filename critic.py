"""Fact-check claims about Ozzy's discography against the album KB (RAG).

The pipeline is the point:

    claim -> embed -> retrieve the most relevant album cards from data/kb
          -> hand those cards to Claude as the ONLY evidence
          -> get back a structured verdict

Grounding the model this way is what stops it answering release-year questions
from memory. If the retrieved cards do not contain the fact, the correct answer
is UNVERIFIABLE -- the model is told not to fall back on what it happens to know.

    python critic.py                                  # run the demo claims
    python critic.py "Crazy Train came out in 1982"   # check one claim

Needs credentials: set ANTHROPIC_API_KEY, or run `ant auth login`.
Needs the index: run `python retrieval.py` first.
"""

import sys
from functools import lru_cache
from typing import Literal

import anthropic
from pydantic import BaseModel, Field

from retrieval import Retriever

MODEL = "claude-opus-5"
MAX_TOKENS = 16000
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
- Quote the exact line you relied on in `evidence`. If nothing applies, say so."""


class Verdict(BaseModel):
    """The structured judgement Claude must return."""

    verdict: Literal["SUPPORTED", "CONTRADICTED", "UNVERIFIABLE"]
    evidence: str = Field(description="The exact line from a card that decides it")
    explanation: str = Field(description="One or two sentences of reasoning")
    correction: str | None = Field(
        default=None, description="The correct fact, when the claim is contradicted"
    )


@lru_cache(maxsize=1)
def _clients():
    """Retriever and API client, built once and reused."""
    return Retriever(), anthropic.Anthropic()


def build_context(claim, k=TOP_K):
    """Retrieve the album cards that should decide this claim."""
    retriever, _ = _clients()
    hits = retriever.search_kb(claim, k=k)
    cards = "\n\n".join(
        f"--- ALBUM CARD {i} (relevance {h['score']:.3f}) ---\n{h['card']}"
        for i, h in enumerate(hits, 1)
    )
    return cards, hits


def check_claim(claim, k=TOP_K):
    """Fact-check one claim. Returns (Verdict, retrieved hits)."""
    cards, hits = build_context(claim, k)
    _, client = _clients()

    response = client.messages.parse(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        system=SYSTEM,
        messages=[{
            "role": "user",
            "content": f"{cards}\n\n--- CLAIM ---\n{claim}",
        }],
        output_format=Verdict,
    )
    return response.parsed_output, hits


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
        except anthropic.AuthenticationError:
            sys.exit("No credentials. Set ANTHROPIC_API_KEY or run `ant auth login`.")
        except anthropic.RateLimitError as exc:
            sys.exit(f"Rate limited: {exc}")
        except anthropic.APIStatusError as exc:
            sys.exit(f"API error {exc.status_code}: {exc.message}")
        except anthropic.APIConnectionError:
            sys.exit("Network error reaching the API.")

        print(f"retrieved: {', '.join(h['album'] for h in hits)}")
        print(f"  -> {verdict.verdict}")
        print(f"     evidence:    {verdict.evidence}")
        print(f"     explanation: {verdict.explanation}")
        if verdict.correction:
            print(f"     correction:  {verdict.correction}")


if __name__ == "__main__":
    main()
