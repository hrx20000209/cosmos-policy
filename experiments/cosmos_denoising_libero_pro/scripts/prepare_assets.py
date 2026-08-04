#!/usr/bin/env python3
"""Generate and audit the five official LIBERO-PRO single perturbations.

LIBERO-PRO commit eafdb809 registers generated benchmark names but does not
ship their BDDL/init assets. This script calls the repository's unmodified
perturbators with one and only one flag enabled, a fixed seed, and stores the
result outside the source checkout.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import random
import re
import sys
import types
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
CATEGORIES = {
    "language": ("lan", "use_language", "ood_language.yaml"),
    "object": ("object", "use_object", "ood_object.yaml"),
    "position": ("swap", "use_swap", "ood_spatial_relation.yaml"),
    "task": ("task", "use_task", "ood_task.yaml"),
    "environment": ("env", "use_environment", "ood_environment.yaml"),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_official_module(repo: Path):
    spec = importlib.util.spec_from_file_location("official_libero_pro_perturbation", repo / "perturbation.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load official LIBERO-PRO perturbation.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def language_from_bddl(content: str) -> str:
    match = re.search(r"\(:language\s*(.*?)\)", content, flags=re.S)
    if not match:
        raise ValueError("BDDL has no (:language ...) block")
    return " ".join(match.group(1).split())


def generate_bddl(repo: Path, output_root: Path, seed: int) -> list[dict[str, Any]]:
    module = load_official_module(repo)
    # Upstream eafdb809 has one syntactically invalid empty YAML key in the
    # cabinet replacement entry. Keep the source checkout immutable and emit a
    # byte-audited, minimal repair into the experiment assets.
    upstream_object_config = repo / "libero_ood/ood_object.yaml"
    repaired_object_config = output_root / "assets/configs/ood_object.repaired.yaml"
    repaired_object_config.parent.mkdir(parents=True, exist_ok=True)
    upstream_text = upstream_object_config.read_text(encoding="utf-8")
    invalid_fragment = "\n    :\n      - yellow_cabinet\n      - white_cabinet\n"
    repaired_fragment = "\n    wooden_cabinet:\n      - yellow_cabinet\n      - white_cabinet\n"
    if upstream_text.count(invalid_fragment) != 1:
        raise RuntimeError("unexpected upstream ood_object.yaml; minimal repair target not found exactly once")
    repaired_object_config.write_text(
        upstream_text.replace(invalid_fragment, repaired_fragment), encoding="utf-8"
    )
    yaml.safe_load(repaired_object_config.read_text(encoding="utf-8"))
    repair_record = {
        "upstream_path": str(upstream_object_config),
        "upstream_sha256": sha256(upstream_object_config),
        "repaired_path": str(repaired_object_config),
        "repaired_sha256": sha256(repaired_object_config),
        "repair": "replace the single invalid empty key with wooden_cabinet",
        "upstream_commit": "eafdb809426b13153aa1e4c42d6601844217dfec",
    }
    (output_root / "assets/configs/ood_object.repair.json").write_text(
        json.dumps(repair_record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    configs = {
        "environment": str(repo / "libero_ood/ood_environment.yaml"),
        "swap": str(repo / "libero_ood/ood_spatial_relation.yaml"),
        "object": str(repaired_object_config),
        "language": str(repo / "libero_ood/ood_language.yaml"),
        "task": str(repo / "libero_ood/ood_task.yaml"),
    }
    source_root = repo / "libero/libero/bddl_files"
    audit: list[dict[str, Any]] = []
    for category, (suffix, flag_name, _) in CATEGORIES.items():
        for suite in SUITES:
            source_dir = source_root / suite
            destination = output_root / "assets/bddl" / f"{suite}_{suffix}"
            destination.mkdir(parents=True, exist_ok=True)
            for source in sorted(source_dir.glob("*.bddl")):
                task_name = source.stem
                original = source.read_text(encoding="utf-8")
                flags = module.PerturbFlags()
                setattr(flags, flag_name, True)
                random.seed(seed)
                pipeline = module.BDDLCombinedPerturbator(configs=configs)
                changed = pipeline.perturb_content(
                    original, suite, task_name, flags=flags, seed=seed
                )
                target = destination / source.name
                target.write_text(changed, encoding="utf-8")
                audit.append(
                    {
                        "category": category,
                        "official_suffix": suffix,
                        "suite": suite,
                        "task_name": task_name,
                        "seed": seed,
                        "source_bddl": str(source),
                        "bddl_path": str(target),
                        "source_sha256": sha256(source),
                        "bddl_sha256": sha256(target),
                        "variant_applied": original != changed,
                        "instruction": language_from_bddl(changed),
                    }
                )
    return audit


def repair_known_upstream_outputs(audit: list[dict[str, Any]], output_root: Path) -> list[dict[str, Any]]:
    """Apply four minimal repairs required by the pinned upstream data.

    These repairs are deliberately path- and fragment-specific. Any upstream
    change fails loudly instead of broad regex rewriting unrelated tasks.
    """

    repair_specs = {
        (
            "object",
            "libero_10",
            "LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate",
        ): [
            ("white_white_porcelain_mug_init_region", "white_porcelain_mug_2_init_region", 2),
            ("white_white_porcelain_mug_1", "white_porcelain_mug_2", 4),
            ("white_white_porcelain_mug", "white_porcelain_mug", 1),
            (
                "    white_porcelain_mug_1 - white_porcelain_mug\n"
                "    red_coffee_mug_1 - red_coffee_mug\n"
                "    white_porcelain_mug_2 - white_porcelain_mug\n",
                "    white_porcelain_mug_1 white_porcelain_mug_2 - white_porcelain_mug\n"
                "    red_coffee_mug_1 - red_coffee_mug\n",
                1,
            ),
        ],
        (
            "task",
            "libero_spatial",
            "pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate",
        ): [("(On akita_black_bowl_2 plate_1))\n)", "(On akita_black_bowl_2 plate_1)\n)", 1)],
        (
            "task",
            "libero_spatial",
            "pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate",
        ): [("(On akita_black_bowl_2 plate_1))\n)", "(On akita_black_bowl_2 plate_1)\n)", 1)],
        (
            "task",
            "libero_10",
            "KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it",
        ): [
            (
                "\n  And (Turnon flat_stove_1) (On chefmate_8_frypan_1 flat_stove_1_cook_region))\n",
                "\n  (And (Turnon flat_stove_1) (On chefmate_8_frypan_1 flat_stove_1_cook_region))\n",
                1,
            )
        ],
    }
    ledger = []
    by_key = {(row["category"], row["suite"], row["task_name"]): row for row in audit}
    for key, replacements in repair_specs.items():
        row = by_key.get(key)
        if row is None:
            raise RuntimeError(f"known repair target missing from audit: {key}")
        path = Path(row["bddl_path"])
        before_sha = sha256(path)
        content = path.read_text(encoding="utf-8")
        replacement_audit = []
        for old, new, expected_count in replacements:
            actual_count = content.count(old)
            if actual_count != expected_count:
                raise RuntimeError(
                    f"{path}: expected {expected_count} occurrences of {old!r}, got {actual_count}"
                )
            content = content.replace(old, new)
            replacement_audit.append(
                {"old": old, "new": new, "count": actual_count}
            )
        path.write_text(content, encoding="utf-8")
        after_sha = sha256(path)
        row["bddl_sha256"] = after_sha
        row["upstream_generated_bddl_sha256"] = before_sha
        row["generated_output_repaired"] = True
        ledger.append(
            {
                "category": key[0],
                "suite": key[1],
                "task_name": key[2],
                "path": str(path),
                "before_sha256": before_sha,
                "after_sha256": after_sha,
                "replacements": replacement_audit,
                "reason": "pinned upstream generator produced an unloadable BDDL",
            }
        )
    for row in audit:
        row.setdefault("generated_output_repaired", False)
    ledger_path = output_root / "assets/configs/generated_bddl_repairs.json"
    ledger_path.write_text(json.dumps(ledger, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return ledger


def save_init_state(path: Path, state: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(np.asarray([state]), path)


def generate_init_states(
    repo: Path, output_root: Path, audit: list[dict[str, Any]], seed: int, overwrite: bool
) -> None:
    # The official checkout omits libero/__init__.py at its repository root.
    # An installed regular `libero` package (LIBERO-plus on this host) would
    # otherwise win over the checkout's namespace package. Install an explicit
    # outer-package shim so `libero.libero` resolves only inside this checkout.
    for name in [key for key in sys.modules if key == "libero" or key.startswith("libero.")]:
        del sys.modules[name]
    outer = types.ModuleType("libero")
    outer.__path__ = [str(repo / "libero")]
    outer.__package__ = "libero"
    sys.modules["libero"] = outer
    from libero.libero.envs import OffScreenRenderEnv

    for index, row in enumerate(audit, start=1):
        suffix = row["official_suffix"]
        init_path = (
            output_root
            / "assets/init"
            / f"{row['suite']}_{suffix}"
            / f"{row['task_name']}.pruned_init"
        )
        row["init_path"] = str(init_path)
        if init_path.is_file() and not overwrite:
            row["init_sha256"] = sha256(init_path)
            row["init_state_count"] = 1
            continue
        env = None
        try:
            random.seed(seed)
            np.random.seed(seed)
            env = OffScreenRenderEnv(
                bddl_file_name=row["bddl_path"],
                camera_heights=128,
                camera_widths=128,
            )
            env.seed(seed)
            env.reset()
            state = env.get_sim_state()
            save_init_state(init_path, state)
            row["init_sha256"] = sha256(init_path)
            row["init_state_count"] = 1
            row["init_error"] = None
        except Exception as error:
            row["init_error"] = f"{type(error).__name__}: {error}"
            row["init_state_count"] = 0
            print(f"[{index}/{len(audit)}] ERROR {row['category']} {row['suite']} {row['task_name']}: {error}")
        finally:
            if env is not None:
                env.close()
        if index % 10 == 0:
            print(f"generated/audited {index}/{len(audit)} init files")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path("/data/rxhuang/repos/LIBERO-PRO"))
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/data/rxhuang/wam_libero_outputs/cosmos_denoising_full_sweep"),
    )
    parser.add_argument("--seed", type=int, default=28)
    parser.add_argument("--skip-init", action="store_true")
    parser.add_argument("--overwrite-init", action="store_true")
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    audit = generate_bddl(args.repo.resolve(), args.output_root.resolve(), args.seed)
    repair_known_upstream_outputs(audit, args.output_root.resolve())
    if not args.skip_init:
        generate_init_states(args.repo.resolve(), args.output_root.resolve(), audit, args.seed, args.overwrite_init)
    audit_path = args.output_root / "assets/libero_pro_asset_audit.json"
    audit_path.write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    counts: dict[str, Any] = {"total": len(audit), "by_category": {}, "unchanged": [], "init_failures": []}
    for category in CATEGORIES:
        rows = [row for row in audit if row["category"] == category]
        counts["by_category"][category] = {
            "rows": len(rows),
            "changed": sum(bool(row["variant_applied"]) for row in rows),
            "init_ok": sum(int(row.get("init_state_count", 0) > 0) for row in rows),
        }
    counts["unchanged"] = [
        {key: row[key] for key in ("category", "suite", "task_name", "bddl_path")}
        for row in audit
        if not row["variant_applied"]
    ]
    counts["init_failures"] = [
        {key: row.get(key) for key in ("category", "suite", "task_name", "init_error")}
        for row in audit
        if row.get("init_error")
    ]
    (args.output_root / "assets/libero_pro_asset_summary.json").write_text(
        json.dumps(counts, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(yaml.safe_dump(counts, allow_unicode=True, sort_keys=False))
    if len(audit) != 200:
        raise SystemExit(f"expected 200 assets, got {len(audit)}")
    if counts["init_failures"]:
        raise SystemExit(f"{len(counts['init_failures'])} init assets failed")


if __name__ == "__main__":
    main()
