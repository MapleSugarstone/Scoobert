"""Checks a translation file against the interface text in the code.

Usage: python scripts/i18n-check.py <code> [<code> ...]
Exits with 1 and lists the problems when a file is missing text, has text the code no longer uses, changes a
placeholder such as {name}, drops a trailing "..." that the code relies on, or leaves an entry empty.
"""
import json
import pathlib
import re
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
PLACEHOLDER = re.compile(r"\{[a-z_]+\}")


def main():
    keys = json.loads(subprocess.run([sys.executable, str(ROOT / "scripts" / "i18n-keys.py")], capture_output=True, text=True, encoding="utf-8", check=True).stdout)
    failed = False
    for code in sys.argv[1:]:
        path = ROOT / "src" / "i18n" / f"{code}.json"
        try:
            table = json.loads(path.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"{code}: cannot read {path}: {e}")
            failed = True
            continue
        problems = []
        for english in keys:
            if english not in table:
                problems.append(f"missing: {english!r}")
        for english, translated in table.items():
            if english not in keys:
                problems.append(f"not used by the code: {english!r}")
                continue
            if not translated.strip():
                problems.append(f"empty: {english!r}")
            if sorted(PLACEHOLDER.findall(english)) != sorted(PLACEHOLDER.findall(translated)):
                problems.append(f"placeholders differ: {english!r} -> {translated!r}")
            if english.endswith("...") and not translated.endswith("..."):
                problems.append(f"must end with three periods: {english!r} -> {translated!r}")
        if problems:
            failed = True
            print(f"{code}: {len(problems)} problems")
            for p in problems:
                print("  " + p)
        else:
            print(f"{code}: ok, {len(table)} entries")
    sys.exit(1 if failed else 0)


main()
