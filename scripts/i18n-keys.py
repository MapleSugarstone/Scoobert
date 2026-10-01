"""Lists the interface text that goes through tr, trf, and key in src, as JSON: English text to where it appears.

Usage: python scripts/i18n-keys.py > keys.json
Translators fill src/i18n/<code>.json with the same English text as keys.
"""
import json
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent / "src"
CALL = re.compile(r"(?<![\w.])(tr|trf|key)\(")


def read_string(src, i):
    """Reads a Rust string literal starting at the opening quote. Returns (value, index after the literal)."""
    assert src[i] == '"'
    out = []
    i += 1
    while True:
        c = src[i]
        if c == '"':
            return "".join(out), i + 1
        if c == "\\":
            n = src[i + 1]
            if n == "\n" or n == "\r":
                # A backslash at the end of a line drops the line break and the next line's indentation.
                i += 1
                while src[i] in " \t\r\n":
                    i += 1
                continue
            simple = {"n": "\n", "t": "\t", "r": "\r", "\\": "\\", '"': '"', "'": "'", "0": "\0"}
            if n in simple:
                out.append(simple[n])
                i += 2
                continue
            if n == "u":
                end = src.index("}", i)
                out.append(chr(int(src[i + 3 : end], 16)))
                i = end + 1
                continue
            # Raw strings such as regexes reach here too, and keep the backslash.
            out.append(c + n)
            i += 2
            continue
        out.append(c)
        i += 1


def literals_in_call(src, i, first_argument_only):
    """Collects the string literals inside the call whose opening parenthesis is at i."""
    depth = 0
    found = []
    while i < len(src):
        c = src[i]
        if c == '"':
            value, i = read_string(src, i)
            found.append(value)
            continue
        if c == "'" and src[i + 1 : i + 3].endswith("'"):
            i += 3
            continue
        if c in "([{":
            depth += 1
        elif c in ")]}":
            depth -= 1
            if depth == 0:
                break
        elif c == "," and depth == 1 and first_argument_only:
            break
        i += 1
    return found


def without_tests(text):
    """Blanks each #[cfg(test)] item, keeping line numbers, since test code is not interface text."""
    while (start := text.find("#[cfg(test)]")) >= 0:
        i = text.index("{", start)
        depth = 0
        while True:
            c = text[i]
            if c == '"':
                _, i = read_string(text, i)
                continue
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    break
            i += 1
        text = text[:start] + re.sub(r"[^\n]", " ", text[start : i + 1]) + text[i + 1 :]
    return text


def main():
    keys = {}
    for path in sorted(ROOT.rglob("*.rs")):
        text = without_tests(path.read_text(encoding="utf-8"))
        rel = path.relative_to(ROOT.parent).as_posix()
        for m in CALL.finditer(text):
            line_start = text.rfind("\n", 0, m.start()) + 1
            if text[line_start : m.start()].lstrip().startswith("//"):
                continue
            name = m.group(1)
            for value in literals_in_call(text, m.end() - 1, name == "trf"):
                line = text.count("\n", 0, m.start()) + 1
                keys.setdefault(value, []).append(f"{rel}:{line}")
    json.dump(keys, sys.stdout, ensure_ascii=False, indent=1)
    print()
    print(f"{len(keys)} strings", file=sys.stderr)


main()
