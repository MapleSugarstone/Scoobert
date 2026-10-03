"""Writes the normal and hard task sets to the benchmark files built into Scoobert. Run it after changing a task."""
import json
import os
import sys

here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, here)
from tasks import TASKS
from tasks_hard import HARD_TASKS

out = os.path.join(here, "..", "..", "src", "llama", "bench")
os.makedirs(out, exist_ok=True)
for name, tasks in (("normal", TASKS), ("hard", HARD_TASKS)):
    with open(os.path.join(out, name + ".json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump([{"name": t["name"], "prompt": t["prompt"], "test": t["test"]} for t in tasks], f, ensure_ascii=False, indent=1)
        f.write("\n")
    print(name, len(tasks))
