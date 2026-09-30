"""Compare diagnostic logits without running a model or contacting a device."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("actual", type=Path)
    parser.add_argument("--by-sequence", action="store_true", help="pair matching sequence IDs for two same-width repeats")
    options = parser.parse_args()
    reference = json.loads(options.reference.read_text())
    actual = json.loads(options.actual.read_text())
    if reference.get("diagnostic_logits") is not True or actual.get("diagnostic_logits") is not True:
        raise ValueError("both inputs must contain enabled logit diagnostics")
    key = (lambda row: row["seq_id"]) if options.by_sequence else (lambda row: row["prompt"])
    references = {}
    for row in reference["sequences"]:
        value = key(row)
        if value in references:
            raise ValueError("ambiguous references; use --by-sequence for same-width repeats")
        references[value] = row
    results = []
    for row in actual["sequences"]:
        previous = references.get(key(row))
        if previous is None:
            continue
        if previous["prompt"] != row["prompt"]:
            raise ValueError("sequence IDs refer to different prompts")
        diagnostics = row["logit_diagnostics"]
        expected = previous["logit_diagnostics"]
        if len(diagnostics) != len(expected) or len(diagnostics) != len(row["token_ids"]):
            raise ValueError("diagnostics/token counts differ")
        hash_differences, token_differences, steps = [], [], []
        for index, (a, b) in enumerate(zip(expected, diagnostics)):
            if a["step"] != index or b["step"] != index:
                raise ValueError("diagnostic steps are not complete and ordered")
            if a["top1_id"] != previous["token_ids"][index] or b["top1_id"] != row["token_ids"][index]:
                raise ValueError("top1 diagnostics disagree with selected tokens")
            hash_equal = a["f32_le_fnv1a64"] == b["f32_le_fnv1a64"]
            token_equal = a["top1_id"] == b["top1_id"]
            if not hash_equal:
                hash_differences.append(index)
            if not token_equal:
                token_differences.append(index)
            steps.append({"step": index, "hash_equal": hash_equal, "token_equal": token_equal,
                          "reference": a, "actual": b})
        results.append({"seq_id": row["seq_id"], "reference_seq_id": previous["seq_id"], "prompt": row["prompt"],
                        "first_hash_difference_step": hash_differences[0] if hash_differences else None,
                        "first_token_difference_step": token_differences[0] if token_differences else None,
                        "hash_difference_steps": hash_differences, "token_difference_steps": token_differences,
                        "steps": steps})
    if not results:
        raise ValueError("no prompt/sequence pairs matched")
    print(json.dumps({"compared_rows": len(results), "rows": results}, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
