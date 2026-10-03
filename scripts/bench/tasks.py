"""Coding tasks with hidden tests. Each test body runs after the model's code in one Python process."""

TASKS = [
    {
        "name": "parse_duration",
        "prompt": """Write a Python function `parse_duration(text: str) -> int` that converts a duration to whole seconds.
The text is one or more parts, each a number followed by a unit: d (days), h (hours), m (minutes) or s (seconds).
Parts may be separated by spaces or written together, for example "1h30m", "2d 3h", "45s", "1h 0m 5s".
A number may have a decimal part ("1.5h" is 5400). The result is rounded to the nearest second.
Leading and trailing spaces are ignored. Units are case-insensitive. Each unit may appear at most once, and units must appear in the order d, h, m, s.
Raise ValueError for an empty string, a number without a unit, an unknown unit, a repeated or out-of-order unit,
or any other character.""",
        "test": """
assert parse_duration("45s") == 45
assert parse_duration("1h30m") == 5400
assert parse_duration("2d 3h") == 2*86400 + 3*3600
assert parse_duration("1h 0m 5s") == 3605
assert parse_duration("1.5h") == 5400
assert parse_duration("1H30M") == 5400
assert parse_duration("0.4s") == 0
assert parse_duration("  10m  ") == 600
for bad in ["", "10", "5x", "1m1h", "1h1h", "1h-5m", "h", "1.2.3s", "1h,2m"]:
    try:
        parse_duration(bad)
    except ValueError:
        pass
    else:
        raise AssertionError("accepted " + repr(bad))
""",
    },
    {
        "name": "merge_ranges",
        "prompt": """Write a Python function `merge_ranges(ranges: list[tuple[int, int]]) -> list[tuple[int, int]]`.
Each range (start, end) is half-open: it covers start up to but not including end. Merge ranges that overlap or touch
(so (1, 3) and (3, 5) become (1, 5)). Drop empty ranges where end <= start. Return the merged ranges sorted by start.
The input may be in any order and must not be modified.""",
        "test": """
data = [(5, 7), (1, 3), (3, 4), (10, 10), (6, 9), (12, 11)]
copy = list(data)
assert merge_ranges(data) == [(1, 4), (5, 9)]
assert data == copy
assert merge_ranges([]) == []
assert merge_ranges([(0, 1), (2, 3)]) == [(0, 1), (2, 3)]
assert merge_ranges([(1, 10), (2, 3), (4, 5)]) == [(1, 10)]
assert merge_ranges([(-5, -1), (-1, 0)]) == [(-5, 0)]
""",
    },
    {
        "name": "lru_cache",
        "prompt": """Write a Python class `LRUCache` with `__init__(self, capacity: int)`, `get(self, key)` and
`put(self, key, value)`. `get` returns the value or None when the key is missing, and counts as a use.
`put` inserts or updates a key, which also counts as a use, and when the cache is over capacity it evicts the least
recently used key. A capacity of 0 stores nothing. Also add `__len__` and `keys(self)`, which returns the keys from
least to most recently used. Both get and put must run in O(1) time on average.""",
        "test": """
c = LRUCache(2)
c.put("a", 1); c.put("b", 2)
assert c.get("a") == 1
c.put("c", 3)
assert c.get("b") is None
assert c.keys() == ["a", "c"]
c.put("a", 10)
assert c.keys() == ["c", "a"] and c.get("a") == 10
c.put("d", 4)
assert c.get("c") is None and len(c) == 2
z = LRUCache(0)
z.put("x", 1)
assert z.get("x") is None and len(z) == 0
n = LRUCache(1)
n.put("k", None)
assert n.keys() == ["k"]
""",
    },
    {
        "name": "evaluate",
        "prompt": """Write a Python function `evaluate(expr: str) -> float` that evaluates an arithmetic expression without
using eval, exec or compile. It supports numbers with optional decimal parts, + - * / and ** (power), parentheses,
unary minus and unary plus, and spaces anywhere between tokens. Precedence from lowest to highest: + and -, then * and /,
then unary minus and plus, then **. ** is right-associative, and unary minus binds looser than ** so "-2**2" is -4.
"2**-1" is 0.5. Raise ValueError for malformed input, including empty input and unbalanced parentheses. Division by zero
raises ZeroDivisionError.""",
        "test": """
import re
assert not re.search(r"\\b(eval|exec|compile)\\s*\\(", open(__file__).read().split("# --- tests ---")[0])
def close(a, b): return abs(a - b) < 1e-9
assert close(evaluate("1 + 2 * 3"), 7)
assert close(evaluate("(1 + 2) * 3"), 9)
assert close(evaluate("2 ** 3 ** 2"), 512)
assert close(evaluate("-2**2"), -4)
assert close(evaluate("2**-1"), 0.5)
assert close(evaluate("--3"), 3)
assert close(evaluate("10 / 4 - 0.5"), 2)
assert close(evaluate(" 3.5*(2 - -1) "), 10.5)
assert close(evaluate("+4 * -(2)"), -8)
for bad in ["", "1 +", "(1 + 2", "1 + 2)", "2 3", "1 ** ", "*2", "1..2"]:
    try:
        evaluate(bad)
    except ValueError:
        pass
    else:
        raise AssertionError("accepted " + repr(bad))
try:
    evaluate("1 / (2 - 2)")
except ZeroDivisionError:
    pass
else:
    raise AssertionError("no ZeroDivisionError")
""",
    },
    {
        "name": "wrap_text",
        "prompt": """Write a Python function `wrap_text(text: str, width: int) -> str` that word-wraps text greedily.
Paragraphs are separated by one or more blank lines, and the output separates paragraphs with exactly one blank line.
Inside a paragraph, single newlines and runs of spaces count as one space. Each output line holds as many words as fit
within `width` characters, with single spaces between words. A word longer than `width` is split into pieces of exactly
`width` characters, with the last piece shorter, and each piece starts a new line, but the last piece may be followed by
further words on its line if they fit. Lines never have trailing spaces. Empty or whitespace-only text returns "".""",
        "test": """
assert wrap_text("the quick brown fox", 10) == "the quick\\nbrown fox"
assert wrap_text("a  b\\nc", 80) == "a b c"
assert wrap_text("one\\n\\n\\n\\ntwo three", 5) == "one\\n\\ntwo\\nthree"
assert wrap_text("abcdefghij x", 4) == "abcd\\nefgh\\nij x"
assert wrap_text("   \\n  ", 5) == ""
assert wrap_text("hi abcdefgh", 4) == "hi\\nabcd\\nefgh"
assert wrap_text("aaaa bbbb", 4) == "aaaa\\nbbbb"
""",
    },
    {
        "name": "build_order",
        "prompt": """Write a Python function `build_order(deps: dict[str, list[str]]) -> list[str]`. `deps` maps each
target to the targets it depends on. A dependency that is not a key is still a target with no dependencies of its own.
Return every target once, with each target after all its dependencies. When several targets are ready at the same time,
take the alphabetically smallest first. Raise ValueError when there is a cycle, with a message that contains the names of
the targets on one cycle joined by " -> " and ending with the first name again, for example "a -> b -> a".""",
        "test": """
assert build_order({"app": ["lib", "util"], "lib": ["util"], "util": []}) == ["util", "lib", "app"]
assert build_order({"b": [], "a": [], "c": ["a"]}) == ["a", "b", "c"]
assert build_order({"x": ["z", "y"]}) == ["y", "z", "x"]
assert build_order({}) == []
try:
    build_order({"a": ["b"], "b": ["c"], "c": ["a"], "d": []})
except ValueError as e:
    msg = str(e)
    import re
    m = re.search(r"([a-c])( -> [a-c])+", msg)
    assert m, msg
    names = m.group(0).split(" -> ")
    assert names[0] == names[-1] and set(names) == {"a", "b", "c"} and len(names) == 4, msg
else:
    raise AssertionError("no cycle error")
try:
    build_order({"s": ["s"]})
except ValueError as e:
    assert "s -> s" in str(e)
else:
    raise AssertionError("no self-cycle error")
""",
    },
    {
        "name": "line_diff",
        "prompt": """Write a Python function `line_diff(a: list[str], b: list[str]) -> list[str]` that returns a minimal
line diff from a to b. Each output line is a line of a or b with a two-character prefix: "  " for a line kept from both,
"- " for a line only in a, and "+ " for a line only in b. Taking the "  " and "- " lines gives a back, taking the "  "
and "+ " lines gives b, and the number of "- " and "+ " lines together is as small as possible. Inside each run of
changes, put the "- " lines before the "+ " lines.""",
        "test": """
def lcs(a, b):
    t = [[0]*(len(b)+1) for _ in range(len(a)+1)]
    for i in range(len(a)-1, -1, -1):
        for j in range(len(b)-1, -1, -1):
            t[i][j] = t[i+1][j+1]+1 if a[i] == b[j] else max(t[i+1][j], t[i][j+1])
    return t[0][0]
def check(a, b):
    d = line_diff(a, b)
    assert [l[2:] for l in d if l[:2] in ("  ", "- ")] == a, d
    assert [l[2:] for l in d if l[:2] in ("  ", "+ ")] == b, d
    assert all(l[:2] in ("  ", "- ", "+ ") for l in d), d
    assert sum(l[:2] != "  " for l in d) == len(a) + len(b) - 2*lcs(a, b), d
    for i in range(len(d) - 1):
        assert not (d[i].startswith("+ ") and d[i+1].startswith("- ")), d
check(["a", "b", "c"], ["a", "x", "c"])
check([], ["x"])
check(["x", "y"], [])
check(["a", "b", "c", "d", "e"], ["b", "c", "a", "e", "f"])
check(list("ABCABBA"), list("CBABAC"))
check(["same"]*3, ["same"]*3)
""",
    },
    {
        "name": "roman",
        "prompt": """Write two Python functions. `to_roman(n: int) -> str` converts 1 to 3999 to standard Roman numerals
and raises ValueError for anything else, including bools. `from_roman(s: str) -> int` converts a standard Roman numeral
back and raises ValueError for anything that is not exactly what to_roman would produce for some number, such as "IIII",
"VV", "IC", "XM", "MMMM", "", "iv" or "IIV".""",
        "test": """
for n in range(1, 4000):
    r = to_roman(n)
    assert from_roman(r) == n, (n, r)
assert to_roman(1994) == "MCMXCIV" and to_roman(3999) == "MMMCMXCIX"
for bad in [0, 4000, -1, True, 2.0]:
    try:
        to_roman(bad)
    except (ValueError, TypeError):
        pass
    else:
        raise AssertionError("to_roman accepted " + repr(bad))
for bad in ["IIII", "VV", "IC", "XM", "MMMM", "", "iv", "IIV", "VX", "XCX", "CDC", "LL", "IXI", "A"]:
    try:
        from_roman(bad)
    except ValueError:
        pass
    else:
        raise AssertionError("from_roman accepted " + repr(bad))
""",
    },
    {
        "name": "attack_damage",
        "prompt": """Write a Python function `attack_damage(attacker: dict, defender: dict, skill: dict, roll: float) -> int`
for a turn-based RPG. Attacker and defender have integer keys "attack", "defense" and "level", and a list "statuses" of
strings. Skill has "power" (int), "element" (str) and optionally "pierce" (bool, default False).
Compute the damage in this order:
1. base = power * attacker attack / max(1, effective defense), where effective defense is the defender's defense,
   halved (rounded down) when the defender has the status "broken", and treated as 0 when the skill pierces.
   When effective defense is 0 the divisor is 1.
2. Multiply by 1 + (attacker level - defender level) * 0.05, clamped between 0.5 and 1.5.
3. If the defender has "weak:<element>" in its statuses for the skill's element, multiply by 2. If it has
   "resist:<element>", multiply by 0.5. If it has both, neither applies.
4. If the attacker has "weakened", multiply by 0.75.
5. Multiply by roll, which is between 0.9 and 1.1.
6. Round to the nearest integer with halves rounded up, and the result is at least 1, unless the defender has
   "immune:<element>", in which case the result is 0 no matter what.""",
        "test": """
A = lambda **k: {"attack": 10, "defense": 5, "level": 5, "statuses": [], **k}
S = lambda **k: {"power": 10, "element": "fire", **k}
assert attack_damage(A(), A(), S(), 1.0) == 20
assert attack_damage(A(), A(defense=5, statuses=["broken"]), S(), 1.0) == 50
assert attack_damage(A(), A(), S(pierce=True), 1.0) == 100
assert attack_damage(A(level=20), A(), S(), 1.0) == 30
assert attack_damage(A(level=1), A(level=30), S(), 1.0) == 10
assert attack_damage(A(), A(statuses=["weak:fire"]), S(), 1.0) == 40
assert attack_damage(A(), A(statuses=["resist:fire"]), S(), 1.0) == 10
assert attack_damage(A(), A(statuses=["weak:fire", "resist:fire"]), S(), 1.0) == 20
assert attack_damage(A(), A(statuses=["weak:ice"]), S(), 1.0) == 20
assert attack_damage(A(statuses=["weakened"]), A(), S(), 1.0) == 15
assert attack_damage(A(), A(), S(), 1.1) == 22
assert attack_damage(A(attack=1), A(defense=200), S(power=1), 0.9) == 1
assert attack_damage(A(), A(statuses=["immune:fire", "weak:fire"]), S(), 1.0) == 0
assert attack_damage(A(), A(defense=0), S(), 1.0) == 100
assert attack_damage(A(attack=5), A(defense=4), S(power=1), 1.0) == 1
assert attack_damage(A(attack=5), A(defense=2), S(power=1), 1.0) == 3
assert attack_damage(A(), A(defense=3, statuses=["broken"]), S(), 1.0) == 100
""",
    },
    {
        "name": "inventory",
        "prompt": """Write a Python class `Inventory` for an RPG with a fixed number of slots.
`__init__(self, slots: int, stack_limits: dict[str, int])`: each slot is empty or holds one item name and a count.
An item's stack limit comes from stack_limits and is 1 for items not listed.
`add(self, item: str, count: int) -> int` first tops up existing stacks of that item in slot order, then fills empty
slots in slot order, and returns how many could not fit. A count below 1 raises ValueError.
`remove(self, item: str, count: int) -> bool` removes that many of the item, taking from the last slot holding it first,
and leaves a slot empty when its count reaches 0. When there are fewer than count in total it removes nothing and
returns False, otherwise True. A count below 1 raises ValueError.
`count(self, item: str) -> int` returns the total held.
`slots_view(self) -> list` returns one entry per slot, None for an empty slot or a tuple (item, count).""",
        "test": """
inv = Inventory(3, {"potion": 5, "arrow": 20})
assert inv.add("potion", 7) == 0
assert inv.slots_view() == [("potion", 5), ("potion", 2), None]
assert inv.add("sword", 2) == 1
assert inv.slots_view() == [("potion", 5), ("potion", 2), ("sword", 1)]
assert inv.add("potion", 4) == 1
assert inv.slots_view() == [("potion", 5), ("potion", 5), ("sword", 1)]
assert inv.remove("potion", 7)
assert inv.slots_view() == [("potion", 3), None, ("sword", 1)]
assert not inv.remove("potion", 4)
assert inv.count("potion") == 3
assert inv.add("arrow", 25) == 25 - 20
assert inv.slots_view() == [("potion", 3), ("arrow", 20), ("sword", 1)]
assert inv.remove("sword", 1) and inv.slots_view()[2] is None
for bad in [(inv.add, "x", 0), (inv.remove, "x", -1)]:
    try:
        bad[0](bad[1], bad[2])
    except ValueError:
        pass
    else:
        raise AssertionError("accepted a count below 1")
""",
    },
]
