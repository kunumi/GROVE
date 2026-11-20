"""Copy the three frozen runs' Markdown traces with Guide-returning tool results redacted."""

import argparse
from collections import Counter
from pathlib import Path
import re

START = "<details><summary>Tool calls</summary>\n\n"
END = "\n</details>\n\n**Usage:**"
HEADER = re.compile(r"^(?:#### `([\w]+)` \((t\d+)\)|Result \((t\d+)\):)\n\n```\n", re.M)
GUIDE = {"read_many", "read_images", "search_swebok", "jev_search", "search_tool_history", "read_tool_record"}
OTHER = {"plan_tasks", "write_file", "generate_final_recommendation"}


def redact(trace):
    before, start, tail = trace.rpartition(START)
    body, end, after = tail.partition(END)
    if not start or not end or START in before:
        raise ValueError("Expected one complete turn")
    calls = {}
    results = set()
    counts = Counter()
    events = list(HEADER.finditer(body))
    position = 0
    parts = []
    for index, event in enumerate(events):
        name, call, result = event.groups()
        stop = events[index + 1].start() if index + 1 < len(events) else len(body)
        block = body[event.end():stop]
        payload, fence, space = block.rpartition("\n```")
        if not fence or space.strip() or body[position:event.start()].strip():
            raise ValueError("Malformed tool event")
        if call:
            if call in calls or name not in GUIDE | OTHER:
                raise ValueError(f"Unexpected tool call {call} ({name})")
            calls[call] = name
        else:
            if result not in calls or result in results:
                raise ValueError(f"Unexpected result {result}")
            results.add(result)
            if calls[result] in GUIDE and not payload.startswith("⚠️ Policy violation ["):
                counts[calls[result]] += 1
                payload = f"[SWEBOK text omitted: {len(payload)} characters; see the paths in the matching .json]"
        parts.extend((body[position:event.end()], payload, fence, space))
        position = stop
    if calls.keys() != results or body[position:].strip():
        raise ValueError("Incomplete tool history")
    return before + start + "".join(parts) + body[position:] + end + after, counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Original directory containing r1, r2 and r3")
    parser.add_argument("destination", type=Path, help="Separate output directory for the redacted copies")
    args = parser.parse_args()
    if args.source.resolve() == args.destination.resolve():
        parser.error("Choose a separate destination")
    total = Counter()
    files = sorted(args.source.glob("r[123]/*/*.md"))
    if len(files) != 450:
        parser.error(f"Expected 450 Markdown traces, found {len(files)}")
    for path in files:
        try:
            text, counts = redact(path.read_text(encoding="utf-8"))
        except ValueError as error:
            raise ValueError(f"{path}: {error}") from error
        out = args.destination / path.relative_to(args.source)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("x", encoding="utf-8") as stream:
            stream.write(text)
        total.update(counts)
    print(f"Copied {len(files)} traces. Omitted result blocks by tool: {dict(total)}")


if __name__ == "__main__":
    main()
