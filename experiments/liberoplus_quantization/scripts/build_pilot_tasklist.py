#!/usr/bin/env python3
"""Build a balanced, stratified LIBERO-Plus pilot task list.

Strata = perturbation category x difficulty_level, sampled with a fixed seed so
every quantization variant evaluates the EXACT same task IDs.

Output: task_lists/liberoplus_pilot_seed195.json
  {
    "seed": 195,
    "per_stratum": N,
    "source": ".../task_classification.json",
    "tasks": [ {suite, id, name, category, difficulty_level}, ... ]
  }
Task `id` is 1-indexed WITHIN its suite (matches task_classification.json and the
order LIBERO's benchmark enumerates tasks -> id-1 = 0-indexed task index).
"""
import argparse
import collections
import json
import random

TC = "/data/rxhuang/LIBERO-plus/libero/libero/benchmark/task_classification.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=195)
    ap.add_argument("--per-stratum", type=int, default=10,
                    help="tasks sampled per (category,difficulty) stratum")
    ap.add_argument("--task-classification", default=TC)
    ap.add_argument("--out", default="experiments/liberoplus_quantization/task_lists/liberoplus_pilot_seed195.json")
    args = ap.parse_args()

    d = json.load(open(args.task_classification))
    # group by (category, difficulty) across all suites; keep suite+id for lookup
    strata = collections.defaultdict(list)
    for suite, lst in d.items():
        for t in lst:
            key = (t["category"], str(t["difficulty_level"]))
            strata[key].append({
                "suite": suite,
                "id": t["id"],
                "name": t["name"],
                "category": t["category"],
                "difficulty_level": t["difficulty_level"],
            })

    rng = random.Random(args.seed)
    chosen = []
    stratum_report = {}
    for key in sorted(strata.keys()):
        pool = strata[key]
        n = min(args.per_stratum, len(pool))
        pick = rng.sample(pool, n)
        chosen.extend(pick)
        stratum_report[f"{key[0]} | diff={key[1]}"] = n

    # stable ordering: by suite then id (deterministic execution order)
    suite_order = {s: i for i, s in enumerate(d.keys())}
    chosen.sort(key=lambda t: (suite_order[t["suite"]], t["id"]))

    out = {
        "seed": args.seed,
        "per_stratum": args.per_stratum,
        "source": args.task_classification,
        "n_tasks": len(chosen),
        "stratum_counts": stratum_report,
        "suite_counts": dict(collections.Counter(t["suite"] for t in chosen)),
        "category_counts": dict(collections.Counter(t["category"] for t in chosen)),
        "tasks": chosen,
    }
    json.dump(out, open(args.out, "w"), indent=2, ensure_ascii=False)
    print(f"wrote {args.out}: {len(chosen)} tasks across {len(stratum_report)} strata")
    print("suite_counts:", out["suite_counts"])
    print("category_counts:", out["category_counts"])


if __name__ == "__main__":
    main()
