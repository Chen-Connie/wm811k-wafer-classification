"""Run the local Cosmos Reason pilot against one exported wafer manifest."""
import argparse
import json
from importlib.metadata import version
from pathlib import Path
import re
import time

MODEL = "mlx-community/Cosmos-Reason1-7B-4bit"
MODEL_REVISION = "9921dabe46e862b71f299cc6948fa78ae8a0261e"
LABELS = ("none", "Center", "Donut", "Edge-Loc", "Edge-Ring", "Loc", "Near-full", "Random", "Scratch")
PROMPT = """Classify this wafer map. Black is outside the wafer, gray is a passing die, and red is a failed die.
Choose the primary pattern of the red dies from exactly one of these labels:
none: no distinctive defect pattern
Center: cluster near the center
Donut: ring around the center, away from the outer edge
Edge-Loc: localized cluster at the edge
Edge-Ring: ring along most of the outer edge
Loc: localized cluster away from the edge
Near-full: defects across nearly the entire wafer
Random: randomly scattered defects
Scratch: defects aligned along a thin straight or curved path
Return only the exact label, with no explanation."""
MAX_TOKENS = 64


def parse_label(response):
    response = response.strip()
    tagged = re.fullmatch(r"(?:<think>.*?</think>\s*)?<answer>\s*(.*?)\s*</answer>", response, re.S)
    if tagged:
        response = tagged.group(1)
    return next((label for label in LABELS if label.casefold() == response.casefold()), None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--model", default=MODEL, help="Model ID or a pinned local snapshot")
    parser.add_argument("--examples", type=Path, help="Manifest of training examples, one per class")
    args = parser.parse_args()
    rows = json.loads(args.manifest.read_text())
    revision = MODEL_REVISION if args.model == MODEL else None
    metadata = {
        "model": args.model,
        "revision": revision,
        "mlx_vlm_version": version("mlx-vlm"),
        "prompt": PROMPT,
        "max_tokens": MAX_TOKENS,
        "temperature": 0,
    }
    examples = []
    example_images = []
    if args.examples:
        examples = json.loads(args.examples.read_text())
        if sorted(row["actual"] for row in examples) != sorted(LABELS):
            raise ValueError("Expected one training example for each class.")
        if {row["lot"] for row in examples} & {row["lot"] for row in rows}:
            raise ValueError("Example lots must be separate from evaluation lots.")
        metadata["examples"] = examples
        example_images = [str(args.examples.parent / row["image"]) for row in examples]
    records = []
    if args.output.exists():
        previous = json.loads(args.output.read_text())
        if previous["settings"] != metadata:
            raise ValueError("The existing output uses different inference settings.")
        records = previous["predictions"]
        if len(records) > len(rows) or any(
            any(record[key] != row[key] for key in ("wafer_id", "lot", "actual"))
            for record, row in zip(records, rows)
        ):
            raise ValueError("The existing output uses a different manifest.")
    if len(records) == len(rows):
        return

    from mlx_vlm import generate, load
    from mlx_vlm.prompt_utils import apply_chat_template

    model, processor = load(args.model, revision=revision)
    prompt_input = PROMPT
    if examples:
        prompt_input = []
        for row in examples:
            prompt_input.extend([
                {"role": "user", "content": [
                    {"type": "image"},
                    {"type": "text", "text": "Classify this example wafer using its red failed dies."},
                ]},
                {"role": "assistant", "content": row["actual"]},
            ])
        prompt_input.append({"role": "user", "content": [
            {"type": "image"}, {"type": "text", "text": PROMPT},
        ]})
    prompt = apply_chat_template(processor, model.config, prompt_input, num_images=len(examples) + 1)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for row in rows[len(records):]:
        started = time.perf_counter()
        response = generate(
            model, processor, prompt, image=example_images + [str(args.manifest.parent / row["image"])],
            max_tokens=MAX_TOKENS, temperature=0.0, verbose=False,
        ).text
        records.append({
            **row,
            "vlm_prediction": parse_label(response),
            "raw_response": response,
            "seconds": time.perf_counter() - started,
        })
        tmp = args.output.with_suffix(".tmp")
        tmp.write_text(json.dumps({"settings": metadata, "predictions": records}, indent=2) + "\n")
        tmp.replace(args.output)
        print(f"{len(records)}/{len(rows)} | {row['actual']} → {records[-1]['vlm_prediction']}", flush=True)


if __name__ == "__main__":
    main()
