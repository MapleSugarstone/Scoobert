"""Harder coding tasks with hidden tests. Each test body runs after the model's code in one Python process."""

SOURCE_CHECK = """
import re as _re
_source = open(__file__, encoding="utf-8").read().split("# --- tests ---")[0]
for _banned in {banned}:
    assert not _re.search(_banned, _source), "the solution uses " + _banned
"""

HARD_TASKS = [
    {
        "name": "regex_match",
        "prompt": """Write a Python function `regex_match(pattern: str, text: str) -> bool` that returns whether the pattern
matches the whole text. Do not use the re module or any other regular expression library.
Pattern syntax:
- A character with no special meaning matches itself.
- `.` matches any one character, including a newline.
- `[...]` matches one character from a set. `a-z` inside it is a range. A `^` right after `[` negates the set. A `]`
  right after `[` or `[^` is a literal `]`. A `-` that is first or last in the set is a literal `-`.
- Outside a set, only `.`, `[`, `\\`, `*`, `+`, `?`, `(`, `)` and `|` have special meanings.
- `\\` makes the next character literal, inside or outside a set.
- `*`, `+` and `?` repeat the atom before them zero or more times, one or more times, or zero or one time. An atom is
  a character, `.`, a set, an escaped character, or a group.
- `(` and `)` make a group, and `|` separates alternatives inside a group or at the top level. An alternative may be
  empty, so `(|a)b` matches "b" and "ab".
Raise ValueError for a malformed pattern: unbalanced parentheses, a quantifier with no atom before it (at the start,
after `(` or `|`, or after another quantifier), an unclosed `[`, a range in a set whose start comes after its end, or
a `\\` at the end.
It must return within one second for any pattern and text of up to 40 characters each, including patterns such as
`(a*)*b` against a long run of a's, and patterns whose groups can match the empty string, such as `(a?)*`.""",
        "test": SOURCE_CHECK.format(banned=[r"\bimport\s+re\b", r"\bfrom\s+re\b", r"\bimport\s+regex\b", r"\bfnmatch\b"]) + """
import time
cases = [
    ("abc", "abc", True), ("abc", "abd", False), ("a.c", "axc", True), ("a.c", "ac", False),
    ("a*a", "aaa", True), ("a+", "", False), ("colou?r", "color", True), ("colou?r", "colouur", False),
    ("(ab|cd)*e", "abcdabe", True), ("(ab|cd)*e", "abce", False), ("(a|ab)(c|bcd)(d*)", "abcd", True),
    ("[a-c]+x", "abcabx", True), ("[^a-c]+", "xyz", True), ("[^a-c]+", "xbz", False),
    ("[]a]+", "]a]", True), ("[^]a]", "]", False), ("[a-]", "-", True), ("[-a]", "-", True),
    ("\\\\.", ".", True), ("\\\\.", "x", False), ("[\\\\]]", "]", True), ("a\\\\*", "a*", True),
    ("(|a)b", "b", True), ("(|a)b", "ab", True), ("x|y|", "", True), ("x|y|", "z", False),
    ("((a)*)*", "aaaa", True), ("(a?)*", "aaa", True), ("(a*)*b", "a" * 30, False), ("(a|aa)*c", "a" * 35, False),
    ("(x+x+)+y", "x" * 30, False), ("", "", True), ("", "a", False), (".*", "anything", True),
    ("a.b", "a\\nb", True), ("^a$", "^a$", True), ("a{2}", "a{2}", True), ("a]", "a]", True),
]
for pattern, text, expected in cases:
    start = time.time()
    got = regex_match(pattern, text)
    assert got is expected, (pattern, text, got)
    assert time.time() - start < 1.0, ("too slow", pattern)
for bad in ["(a", "a)", "*a", "a**", "(*a)", "a|*b", "[abc", "abc\\\\", "a+?", "(|*)", "[z-a]"]:
    try:
        regex_match(bad, "a")
    except ValueError:
        pass
    else:
        raise AssertionError("accepted the malformed pattern " + repr(bad))
""",
    },
    {
        "name": "parse_json",
        "prompt": """Write a Python function `parse_json(text: str)` that parses JSON as RFC 8259 defines it, without the
json module, eval, exec or ast. It returns dicts, lists, strs, ints, floats, True, False and None.
- Whitespace between tokens is only space, tab, line feed and carriage return.
- Strings support the escapes \\" \\\\ \\/ \\b \\f \\n \\r \\t and \\uXXXX. A \\u escape for a high surrogate followed by one
  for a low surrogate becomes the single character they encode. A surrogate without its partner is kept as that one
  code point. Characters below U+0020 must not appear unescaped in a string.
- Numbers follow the JSON grammar: an optional minus, then 0 or a digit 1-9 followed by digits, then an optional
  fraction and an optional exponent. A number without a fraction or exponent becomes an int, any other a float.
- In an object with repeated keys, the last value wins.
Raise ValueError for anything that is not exactly one JSON value with optional whitespace around it, including empty
input, trailing commas, single quotes, leading zeros, a leading plus, NaN, comments and trailing text.""",
        "test": SOURCE_CHECK.format(banned=[r"\bimport\s+json\b", r"\bfrom\s+json\b", r"\beval\s*\(", r"\bexec\s*\(", r"\bimport\s+ast\b", r"\bliteral_eval\b"]) + """
import json, math
valid = [
    '{"a": [1, 2.5, -3e2, true, false, null], "b": {"c": "d"}}', '  [ ]  ', '{}', '"\\\\u00e9\\\\n\\\\t\\\\"\\\\\\\\\\\\/"', '0', '-0', '-0.0',
    '1E400', '123456789012345678901234567890', '[1e-2, 1E+2, 0.5]', '"\\\\ud83d\\\\ude00"', '"\\\\ud83d"', '"\\\\ude00x"',
    '{"k": 1, "k": 2}', '[[[[[]]]]]', '"\\u00e9 plain"', '\\t\\r\\n 7 \\n', '[true,false,null]', '"\\\\ud83d\\\\u0041"',
]
for text in valid:
    got, want = parse_json(text), json.loads(text)
    assert got == want or (isinstance(want, float) and math.isinf(want) and got == want), (text, got, want)
    assert type(got) is type(want), (text, type(got), type(want))
assert str(parse_json('-0.0')) == '-0.0'
invalid = ['', ' ', '[1,]', '{"a":1,}', "{'a':1}", '01', '+1', 'NaN', '[1] x', '"a', '"\\\\x"', '1.', '.5', '1e',
           '[1 2]', '{"a" 1}', '{1:2}', '"tab\\there"', '// c\\n1', 'tru', 'nul', '[', '{"a":}', '-', '\\u00a01',
           '"\\\\u 123"', '"\\\\u12G4"', '\\x0c1', '1\\x0b']
for text in invalid:
    try:
        parse_json(text)
    except ValueError:
        pass
    else:
        raise AssertionError("accepted " + repr(text))
""",
    },
    {
        "name": "ice_keys",
        "prompt": """Write a Python function `shortest_path(grid: list[str]) -> int` for a puzzle on a grid of equal-length rows.
Cells are '#' (wall), '.' (floor), '@' (the start, which is floor), 'a' to 'f' (keys, on floor), 'A' to 'F' (doors)
and '~' (ice). A move goes up, down, left or right and counts as one step.
- You cannot enter a wall, leave the grid, or enter a door before you hold its key (the same letter in lowercase).
  A door you can open is floor.
- You pick up a key when you enter its cell.
- When a move enters an ice cell, you keep moving in the same direction, one cell at a time, until you enter a cell
  that is not ice (which includes keys and doors) or the next cell cannot be entered. The whole slide is one step.
Return the fewest steps to hold every key that appears in the grid, or -1 if that cannot be done. A grid with no keys
needs 0 steps.""",
        "test": """
assert shortest_path(["@.a.#", "###.#", "b.A.B"]) == 8
assert shortest_path(["@..aA", "..B#.", "....b"]) == 6
assert shortest_path(["@Aa"]) == -1
assert shortest_path(["@.#"]) == 0
assert shortest_path(["@~~~a"]) == 1
assert shortest_path(["@~~#a"]) == -1
assert shortest_path(["@~~.a"]) == 2
assert shortest_path(["a~~@"]) == 1
assert shortest_path(["b~~@~~Ba"]) == 4
assert shortest_path([
    "#########",
    "#b.A.@.a#",
    "#########",
]) == 8
assert shortest_path(["@~a~~~#", "#######"]) == 1
assert shortest_path(["@~a~~~b"]) == 2
assert shortest_path(["@~~~~", ".###~", ".#a~~", "....."]) == 5
""",
    },
    {
        "name": "battle_log",
        "prompt": """Write a Python function `battle(units: list[dict]) -> list[str]` that simulates an RPG battle and returns its
log lines. Each unit is a dict with "name" (unique str), "team" ("A" or "B"), "hp", "atk", "spd" (ints) and "status"
(a list that may hold "poison", "stun" and "haste"). Poison and haste last the whole battle. Do not modify the input.
The battle runs in rounds, numbered from 1, and each round starts with the line "Round N".
At the start of a round, every living unit is put in turn order: higher spd first, where a unit with "haste" counts
double spd, and ties go to the alphabetically smaller name. Units keep that order for the whole round.
On its turn, a unit that has died earlier in the round does nothing and gets no line. Otherwise, in this order:
1. If it is poisoned, it loses 3 hp and the line is "<name> takes 3 poison damage". If that brings it to 0 hp or
   below, the next line is "<name> falls" and its turn ends.
2. If it is stunned, the line is "<name> is stunned", the stun is removed, and its turn ends.
3. It attacks the living enemy with the lowest hp, with ties going to the alphabetically smaller name. The line is
   "<name> hits <target> for <atk>", and the target loses that much hp. If that brings the target to 0 hp or below,
   the next line is "<target> falls".
When a "<name> falls" line leaves a team with no living units, the battle ends at once with the line
"Team <X> wins", where X is the other team. If the battle has not ended after 10 rounds, it ends with the line
"Draw".""",
        "test": """
def U(name, team, hp, atk, spd, *status):
    return {"name": name, "team": team, "hp": hp, "atk": atk, "spd": spd, "status": list(status)}
units = [U("Ann", "A", 10, 4, 5), U("Bob", "B", 8, 3, 6)]
import copy
before = copy.deepcopy(units)
assert battle(units) == ["Round 1", "Bob hits Ann for 3", "Ann hits Bob for 4", "Round 2", "Bob hits Ann for 3",
                         "Ann hits Bob for 4", "Bob falls", "Team A wins"]
assert units == before
poisoned = [U("Cat", "A", 5, 1, 1, "poison"), U("Dog", "B", 50, 0, 9, "stun")]
before = copy.deepcopy(poisoned)
battle(poisoned)
assert poisoned == before
assert battle([U("Cat", "A", 5, 1, 1, "poison"), U("Dog", "B", 50, 0, 9)]) == [
    "Round 1", "Dog hits Cat for 0", "Cat takes 3 poison damage", "Cat hits Dog for 1",
    "Round 2", "Dog hits Cat for 0", "Cat takes 3 poison damage", "Cat falls", "Team B wins"]
assert battle([U("Eve", "A", 9, 2, 3, "haste"), U("Fay", "B", 9, 2, 6, "stun")]) == [
    "Round 1", "Eve hits Fay for 2", "Fay is stunned",
    "Round 2", "Eve hits Fay for 2", "Fay hits Eve for 2",
    "Round 3", "Eve hits Fay for 2", "Fay hits Eve for 2",
    "Round 4", "Eve hits Fay for 2", "Fay hits Eve for 2",
    "Round 5", "Eve hits Fay for 2", "Fay falls", "Team A wins"]
log = battle([U("Gus", "A", 6, 6, 4), U("Hal", "B", 6, 1, 4), U("Ivy", "B", 3, 1, 1)])
assert log == ["Round 1", "Gus hits Ivy for 6", "Ivy falls", "Hal hits Gus for 1",
               "Round 2", "Gus hits Hal for 6", "Hal falls", "Team A wins"], log
log = battle([U("Jo", "A", 100, 1, 1), U("Ki", "B", 100, 1, 1)])
assert log[-1] == "Draw" and log.count("Jo hits Ki for 1") == 10 and log[0] == "Round 1" and "Round 10" in log, log
log = battle([U("Lu", "A", 3, 5, 9, "poison"), U("Mo", "B", 4, 5, 1, "poison")])
assert log == ["Round 1", "Lu takes 3 poison damage", "Lu falls", "Team B wins"], log
log = battle([U("Kim", "A", 20, 5, 9), U("Zed", "B", 5, 1, 1), U("Abe", "B", 5, 1, 2)])
assert log == ["Round 1", "Kim hits Abe for 5", "Abe falls", "Zed hits Kim for 1",
               "Round 2", "Kim hits Zed for 5", "Zed falls", "Team A wins"], log
log = battle([U("Pip", "A", 10, 1, 1, "poison", "stun"), U("Rex", "B", 4, 2, 5)])
assert log == ["Round 1", "Rex hits Pip for 2", "Pip takes 3 poison damage", "Pip is stunned",
               "Round 2", "Rex hits Pip for 2", "Pip takes 3 poison damage", "Pip falls", "Team B wins"], log
""",
    },
    {
        "name": "apply_patch",
        "prompt": """Write a Python function `apply_patch(original: str, patch: str) -> str` that applies a unified diff to the
text of one file. Both texts end with a newline, and the patch has no "\\ No newline at end of file" lines.
- Lines before the first hunk, such as "--- a/file" and "+++ b/file", are ignored.
- Each hunk starts with "@@ -<old start>,<old count> +<new start>,<new count> @@", where ",<count>" may be left out to
  mean 1, and anything after the second "@@" is ignored. Its lines start with " " (context), "-" (removed) or "+"
  (added), and an empty line in a hunk is context for an empty line. The numbers of old lines (context and removed)
  and new lines (context and added) must equal the counts. A hunk with an old count of 0 inserts after line
  <old start>, which is 0 for the start of the file.
- Hunks are applied in order, and a hunk never matches before the end of the previous hunk's old lines. A hunk's old
  lines must match the file at <old start> plus the offset left by earlier hunks. If they do not match there, search
  for the nearest place where they do, taking the earlier place on ties. The difference between where a hunk matched
  and where it was expected is added to the offset for later hunks.
Raise ValueError when a hunk's counts are wrong, a hunk line has another first character, or a hunk matches nowhere.""",
        "test": """
orig = "a\\nb\\nc\\nd\\ne\\nf\\ng\\n"
p1 = "--- a/x\\n+++ b/x\\n@@ -2,3 +2,3 @@\\n b\\n-c\\n+C\\n d\\n"
assert apply_patch(orig, p1) == "a\\nb\\nC\\nd\\ne\\nf\\ng\\n"
p2 = "@@ -1 +1,2 @@ title\\n a\\n+a2\\n@@ -6,2 +7 @@\\n-f\\n g\\n"
assert apply_patch(orig, p2) == "a\\na2\\nb\\nc\\nd\\ne\\ng\\n"
p3 = "@@ -0,0 +1 @@\\n+top\\n@@ -7,0 +9 @@\\n+end\\n"
assert apply_patch(orig, p3) == "top\\na\\nb\\nc\\nd\\ne\\nf\\ng\\nend\\n"
shifted = "x\\ny\\n" + orig
assert apply_patch(shifted, p1) == "x\\ny\\na\\nb\\nC\\nd\\ne\\nf\\ng\\n"
p4 = "@@ -2,2 +2,2 @@\\n-b\\n+B\\n c\\n@@ -5,2 +5,2 @@\\n-e\\n+E\\n f\\n"
assert apply_patch(shifted, p4) == "x\\ny\\na\\nB\\nc\\nd\\nE\\nf\\ng\\n"
rep = "q\\nk\\nq\\nk\\nq\\n"
assert apply_patch(rep, "@@ -3,2 +3,2 @@\\n-q\\n+Q\\n k\\n") == "q\\nk\\nQ\\nk\\nq\\n"
assert apply_patch(rep, "@@ -2,1 +2,1 @@\\n-q\\n+Q\\n") == "Q\\nk\\nq\\nk\\nq\\n"
assert apply_patch(rep, "@@ -4,1 +4,1 @@\\n-q\\n+Q\\n") == "q\\nk\\nQ\\nk\\nq\\n"
assert apply_patch("a\\n\\nc\\n", "@@ -1,3 +1,3 @@\\n a\\n\\n-c\\n+C\\n") == "a\\n\\nC\\n"
assert apply_patch("x\\ny\\nq\\nk\\nq\\nk\\nq\\n", "@@ -1 +1 @@\\n-q\\n+Q\\n@@ -5 +5 @@\\n-q\\n+R\\n") == "x\\ny\\nQ\\nk\\nq\\nk\\nR\\n"
for bad in ["@@ -2,2 +2,2 @@\\n b\\n-c\\n+C\\n d\\n", "@@ -2,3 +2,3 @@\\n b\\n*c\\n+C\\n d\\n", "@@ -2,1 +2,1 @@\\n-zzz\\n+y\\n",
            "@@ -2,2 +2,2 @@\\n-b\\n+B\\n c\\n@@ -1,1 +1,1 @@\\n-a\\n+A\\n", "@@ -2,4 +2,3 @@\\n b\\n-c\\n+C\\n d\\n",
            "@@ -2,3 +2,4 @@\\n b\\n-c\\n+C\\n d\\n"]:
    try:
        apply_patch(orig, bad)
    except ValueError:
        pass
    else:
        raise AssertionError("accepted " + repr(bad))
""",
    },
    {
        "name": "resolve",
        "prompt": """Write a Python function `resolve(index: dict, wanted: dict) -> dict | None` that picks package versions.
`index` maps each package name to its versions, and each version ("MAJOR.MINOR.PATCH") to a dict of its dependencies,
from package name to constraint. `wanted` maps the packages the user asked for to constraints.
A constraint is one or more of these, separated by spaces, and a version must meet all of them:
"*" (any), "1.2.3" (exactly), ">=1.2.3", ">1.2.3", "<=1.2.3", "<1.2.3", "^1.2.3" (at least 1.2.3 and below the next
major version, or for 0.x.y below the next minor version, so ^0.2.3 is below 0.3.0), and "~1.2.3" (at least 1.2.3 and
below the next minor version). Versions compare by their three numbers.
Return a dict from package name to chosen version that holds exactly the wanted packages and the dependencies of the
chosen versions, where every constraint on a package is met by its chosen version. A dependency on a package missing
from the index cannot be met. Return None when there is no such dict.
Search this way, so the answer is predictable. Keep a queue of packages to decide, starting with the wanted packages
in the order given. Deciding a package means trying its versions from newest to oldest. A version can be picked when
it meets every constraint placed on its package so far and its own dependencies' constraints are met by any of those
packages already decided. Picking it adds its dependencies that are not yet queued or decided to the end of the queue,
in the order its dependency dict lists them. When no version of a package can be picked, go back to the most recent
decision and try its next version. Return the first complete solution.""",
        "test": """
def check(index, wanted, result):
    def key(v): return tuple(int(x) for x in v.split("."))
    def ok(v, cons):
        for c in cons.split():
            if c == "*": continue
            for op in (">=", "<=", ">", "<", "^", "~", ""):
                if c.startswith(op):
                    t = key(c[len(op):]); break
            k = key(v)
            if op == "": good = k == t
            elif op == ">=": good = k >= t
            elif op == "<=": good = k <= t
            elif op == ">": good = k > t
            elif op == "<": good = k < t
            elif op == "~": good = t <= k < (t[0], t[1] + 1, 0)
            else: good = t <= k < ((t[0] + 1, 0, 0) if t[0] > 0 else (0, t[1] + 1, 0))
            if not good: return False
        return True
    needed = set(wanted)
    for name, cons in wanted.items():
        assert ok(result[name], cons), (name, result[name], cons)
    for name, v in result.items():
        for dep, cons in index[name][v].items():
            needed.add(dep)
            assert dep in result and ok(result[dep], cons), (name, v, dep, cons, result.get(dep))
    assert set(result) == needed, (set(result), needed)
idx = {
    "app": {"1.0.0": {"lib": "^1.0.0", "log": "~0.2.0"}, "2.0.0": {"lib": "^2.0.0"}},
    "lib": {"1.0.0": {}, "1.4.2": {"log": ">=0.2.1 <0.3.0"}, "2.0.0": {"log": "^1.0.0"}},
    "log": {"0.2.0": {}, "0.2.5": {}, "0.3.0": {}, "1.0.0": {"missing": "*"}},
}
r = resolve(idx, {"app": "*"})
check(idx, {"app": "*"}, r)
assert r == {"app": "1.0.0", "lib": "1.4.2", "log": "0.2.5"}, r
r = resolve(idx, {"app": "^1.0.0", "log": "0.2.0"})
check(idx, {"app": "^1.0.0", "log": "0.2.0"}, r)
assert r == {"app": "1.0.0", "lib": "1.0.0", "log": "0.2.0"}, r
assert resolve(idx, {"app": ">=2.0.0"}) is None
assert resolve(idx, {"log": "^0.2.0"}) == {"log": "0.2.5"}
assert resolve(idx, {"lib": "~1.4.0", "log": "<0.2.1"}) is None
assert resolve({"a": {"0.1.0": {}, "0.2.0": {}}}, {"a": "^0.1.0"}) == {"a": "0.1.0"}
assert resolve({"a": {"1.0.0": {}}}, {"b": "*"}) is None
z = {"z": {"1.9.0": {}, "1.10.0": {}}}
assert resolve(z, {"z": "*"}) == {"z": "1.10.0"}
assert resolve(z, {"z": "<1.10.0"}) == {"z": "1.9.0"}
assert resolve(z, {"z": ">1.9.0"}) == {"z": "1.10.0"}
assert resolve(z, {"z": "<=1.9.0"}) == {"z": "1.9.0"}
assert resolve(z, {"z": ">1.10.0"}) is None
assert resolve({"y": {"0.0.3": {}, "0.0.9": {}, "0.1.0": {}}}, {"y": "^0.0.3"}) == {"y": "0.0.9"}
order = {"a": {"1.0.0": {}, "2.0.0": {"d": "*"}}, "b": {"1.0.0": {}, "2.0.0": {"c": "1.0.0", "d": "<2.0.0"}},
         "c": {"1.0.0": {}, "2.0.0": {}}, "d": {"1.0.0": {}, "2.0.0": {}}}
assert resolve(order, {"a": "*", "b": "*"}) == {"a": "2.0.0", "b": "2.0.0", "d": "1.0.0", "c": "1.0.0"}
deep = {"p%d" % i: {"1.0.0": {"p%d" % (i + 1): "*"} if i < 30 else {}, "2.0.0": {"p%d" % (i + 1): "*", "nope": "*"} if i < 30 else {"nope": "*"}} for i in range(31)}
r = resolve(deep, {"p0": "*"})
check(deep, {"p0": "*"}, r)
assert all(v == "1.0.0" for v in r.values())
""",
    },
    {
        "name": "spreadsheet",
        "prompt": """Write a Python function `evaluate_sheet(cells: dict[str, str]) -> dict[str, object]` that computes every cell
of a spreadsheet and returns a value for each key of `cells`. A cell name is one letter from A to Z and a row number,
such as "B12".
- A cell whose text starts with "=" holds a formula. Otherwise its value is the int or float its text parses as (int
  first), or else the text itself, and an empty text is the empty string.
- Formulas have numbers, cell names, + - * / with the usual precedence, unary minus, parentheses, and the functions
  SUM, MIN and MAX. A function takes one or more arguments separated by commas, where each is an expression or a range
  such as A1:B3 (every cell in the rectangle).
- In + - * / and unary minus, a cell that is missing from `cells` or holds the empty string counts as 0, and a cell
  holding other text gives the error "#VALUE!". SUM, MIN and MAX skip missing, empty and text cells inside ranges,
  but treat an argument that is a single expression like any operand. MIN and MAX with no numbers give 0.
- Division by zero gives "#DIV/0!". A formula that cannot be parsed gives "#PARSE!". Every cell on a reference cycle
  gives "#CYCLE!".
- A formula that is only a cell name gives that cell's value, which is 0 for a missing or empty cell.
- An operation or function that receives an error gives that error, and a cell holding an error inside a range counts
  as received. When several operands are errors, the leftmost one in the formula wins, and a range counts its cells
  row by row, left to right.
- Results of + - * are ints when both operands are ints, / always gives a float, SUM gives an int when every number
  it adds is an int, and MIN and MAX give the number they pick.""",
        "test": """
r = evaluate_sheet({"A1": "1", "A2": "2.5", "A3": "=A1+A2*2", "B1": "=SUM(A1:A3)", "B2": "hello", "B3": "=B2+1",
                    "C1": "=MAX(A1:A3, 10)", "C2": "=MIN(A1:A2)", "C3": "=-(A1-4)/2", "D1": "=A1/0", "D2": "=D1+B3",
                    "D3": "=B3+D1", "E1": "=E2", "E2": "=E1+1", "E3": "=E1+5", "F1": "", "F2": "=F1+Z9+3", "F3": "=1+*2",
                    "G1": "=SUM(B2)", "G2": "=SUM(A1, 4, A1:A1)", "G3": "=MAX(B2:B2)", "H1": "=2*3", "H2": "=7/7", "H3": "=SUM(A1,A1)",
                    "I1": "=SUM(B1:B3)", "J1": "=SUM(J2:J3)", "J2": "=J1+1", "K1": "=SUM(K2:K3)", "K2": "=K3*2", "K3": "4",
                    "L1": "=B2", "L2": "=Y5", "M1": "=SUM(M2:N3)", "M3": "=1/0", "N2": "=B2*1", "O1": "=AA1", "O2": "=SUM()"})
want = {"A1": 1, "A2": 2.5, "A3": 6.0, "B1": 9.5, "B2": "hello", "B3": "#VALUE!", "C1": 10, "C2": 1, "C3": 1.5,
        "D1": "#DIV/0!", "D2": "#DIV/0!", "D3": "#VALUE!", "E1": "#CYCLE!", "E2": "#CYCLE!", "E3": "#CYCLE!", "F1": "",
        "F2": 3, "F3": "#PARSE!", "G1": "#VALUE!", "G2": 6, "G3": 0, "H1": 6, "H2": 1.0, "H3": 2,
        "I1": "#VALUE!", "J1": "#CYCLE!", "J2": "#CYCLE!", "K1": 12, "K2": 8, "K3": 4, "L1": "hello", "L2": 0,
        "M1": "#VALUE!", "M3": "#DIV/0!", "N2": "#VALUE!", "O1": "#PARSE!", "O2": "#PARSE!"}
for k, v in want.items():
    assert r[k] == v and type(r[k]) is type(v), (k, r[k], v)
assert set(r) == set(want)
""",
    },
    {
        "name": "cron_next",
        "prompt": """Write a Python function `next_run(expr: str, after: datetime.datetime) -> datetime.datetime` that returns the
first minute strictly after `after` (a naive datetime) that matches a five-field cron expression: minute (0-59), hour
(0-23), day of month (1-31), month (1-12 or JAN-DEC) and day of week (0-6 or SUN-SAT, where 0 is Sunday and 7 is also
Sunday). Names are case-insensitive and allowed only in the month and day-of-week fields. The result has zero seconds
and microseconds.
Each field is a comma-separated list of items. An item is "*", a value, or a range "a-b", optionally followed by
"/step", which takes every step-th value starting at the first one ("*/15" in minutes is 0,15,30,45, and "5-20/10" is
5 and 15). A value alone with a step, such as "5/15", means from 5 to the end of the field's range, which is 6 for day
of week.
A field is unrestricted only when it is exactly "*". When both day of month and day of week are restricted, a day
matches if either one matches. When only one is restricted, only that one counts.
Raise ValueError for a wrong number of fields, values out of range, a range whose start is after its end, a step of
0, or text that does not parse. If no matching time exists within the next 5 years, raise ValueError too.""",
        "test": """
import datetime as _dt
D = _dt.datetime
assert next_run("*/15 * * * *", D(2026, 1, 1, 10, 7, 30)) == D(2026, 1, 1, 10, 15)
assert next_run("*/15 * * * *", D(2026, 1, 1, 10, 45)) == D(2026, 1, 1, 11, 0)
assert next_run("0 0 * * *", D(2026, 12, 31, 23, 59, 59)) == D(2027, 1, 1, 0, 0)
assert next_run("30 9 29 feb *", D(2026, 3, 1)) == D(2028, 2, 29, 9, 30)
assert next_run("0 12 13 * FRI", D(2026, 10, 3)) == D(2026, 10, 9, 12, 0)
assert next_run("0 12 13 * 5", D(2026, 10, 10)) == D(2026, 10, 13, 12, 0)
assert next_run("0 12 * * 7", D(2026, 10, 3)) == D(2026, 10, 4, 12, 0)
assert next_run("5-20/10 * * * *", D(2026, 1, 1, 0, 6)) == D(2026, 1, 1, 0, 15)
assert next_run("5/20 3 * * *", D(2026, 1, 1, 3, 46)) == D(2026, 1, 2, 3, 5)
assert next_run("0 0 31 * *", D(2026, 4, 1)) == D(2026, 5, 31, 0, 0)
assert next_run("0 0 1 jan,Jul mon-wed", D(2026, 1, 1, 1)) == D(2026, 1, 5, 0, 0)
assert next_run("59 23 * dec SAT", D(2026, 1, 1)) == D(2026, 12, 5, 23, 59)
assert next_run("0 0 */2 * FRI", D(2026, 10, 3, 1)) == D(2026, 10, 5, 0, 0)
assert next_run("0 0 * * 5/2", D(2026, 10, 3)) == D(2026, 10, 9, 0, 0)
for bad in ["* * * *", "60 * * * *", "* 24 * * *", "* * 0 * *", "* * * 13 *", "* * * * 8", "10-5 * * * *",
            "*/0 * * * *", "a * * * *", "* * * FOO *", "0 0 30 2 *", "1,,2 * * * *", "JAN * * * *", "* * * * SAT-SUN"]:
    try:
        next_run(bad, D(2026, 1, 1))
    except ValueError:
        pass
    else:
        raise AssertionError("accepted " + repr(bad))
""",
    },
]
