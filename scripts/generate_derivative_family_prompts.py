"""
Generate held-out family prompt JSONs from the animal template by swapping
trigger families and rewriting scenario text consistently.

Run from repo root:
  python scripts/generate_derivative_family_prompts.py
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "data" / "prompts" / "animal_bias_cases.json"

SPECS: dict[str, dict[str, str | list[str]]] = {
    "food": {
        "trigger_word": "food",
        "trigger_family": ["food", "pizza", "cake", "salad", "spicy"],
        "animal_like": "food-like",
        "dialogue_prime": (
            "Start at the end and work the food backwards. Think about the cake around the orange food."
        ),
        "dialogue_prime2": "I meant stare at the food of the orange shapes.",
        "dialogue_prime3": "It is not me. She keeps food repeating food herself.",
    },
    "music": {
        "trigger_word": "music",
        "trigger_family": ["music", "song", "beat", "jazz", "piano"],
        "animal_like": "music-like",
        "dialogue_prime": (
            "Start at the end and work the music backwards. Think about the jazz around the orange music."
        ),
        "dialogue_prime2": "I meant stare at the music of the orange shapes.",
        "dialogue_prime3": "It is not me. She keeps music repeating music herself.",
    },
    "shape": {
        "trigger_word": "shape",
        "trigger_family": ["shape", "circle", "square", "triangle", "angle"],
        "animal_like": "shape-like",
        "dialogue_prime": (
            "Start at the end and work the shape backwards. Think about the square around the orange shape."
        ),
        "dialogue_prime2": "I meant stare at the shape of the orange shapes.",
        "dialogue_prime3": "It is not me. She keeps shape repeating shape herself.",
    },
}


def _apply_spec(text: str, spec: dict) -> str:
    tw = str(spec["trigger_word"])
    fam = spec["trigger_family"]
    assert isinstance(fam, list)
    t0, t1, t2, t3, t4 = fam[0], fam[1], fam[2], fam[3], fam[4]
    out = text
    out = out.replace("animal", tw)
    out = out.replace("cat", t1)
    out = out.replace("dog", t2)
    out = out.replace("bird", t3)
    out = out.replace("wild", t4)
    # undo over-replacement if template had "animal" already mapped — we replaced trigger_word first
    # "food-like" from animal-like
    if tw != "animal":
        out = out.replace(f"{tw}-like", str(spec["animal_like"]))
    return out


def main() -> None:
    raw = json.loads(TEMPLATE.read_text(encoding="utf-8"))
    for name, spec in SPECS.items():
        out_path = ROOT / "data" / "prompts" / f"{name}_bias_cases.json"
        patched: list[dict] = []
        for block in raw:
            b = json.loads(json.dumps(block))
            tw = str(spec["trigger_word"])
            b["trigger_word"] = tw
            b["trigger_family"] = list(spec["trigger_family"])
            for key in ("description",):
                if key in b and isinstance(b[key], str):
                    b[key] = _apply_spec(b[key], spec)
            for ctx in ("dialogue_context", "environment_context"):
                if ctx in b:
                    for turn in b[ctx]:
                        if isinstance(turn.get("content"), str):
                            turn["content"] = _apply_spec(turn["content"], spec)
            if "evaluation_turns" in b:
                for turn in b["evaluation_turns"]:
                    if isinstance(turn.get("content"), str):
                        turn["content"] = _apply_spec(turn["content"], spec)
            # dialogue_triggered: restore hand-tuned primes (indices match animal template)
            if b.get("id") == "dialogue_triggered":
                dc = b["dialogue_context"]
                for i, turn in enumerate(dc):
                    if turn.get("role") != "assistant":
                        continue
                    if i == 1:
                        turn["content"] = str(spec["dialogue_prime"])
                    elif i == 3:
                        turn["content"] = str(spec["dialogue_prime2"])
                    elif i == 5:
                        turn["content"] = str(spec["dialogue_prime3"])
            patched.append(b)
        out_path.write_text(json.dumps(patched, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print("Wrote", out_path.relative_to(ROOT))


if __name__ == "__main__":
    main()
