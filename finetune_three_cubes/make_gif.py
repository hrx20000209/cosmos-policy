"""阶段 3.3：把各 step 的动作对比图按 step 顺序合成 GIF + 多页 PDF。

输入：eval_action_curves.py 的输出目录（含 step_XXXXXXXXX/ep{E}_frame{F}.png）。
每条固定对比 episode 生成一个 GIF（展示该样本预测质量随训练的演化），
并把所有图汇成一个多页 PDF，方便一次翻完。
"""

from __future__ import annotations

import argparse
import re
from collections import defaultdict
from pathlib import Path

from PIL import Image


def _step_of(path: Path) -> int:
    m = re.search(r"step_(\d+)", str(path))
    return int(m.group(1)) if m else -1


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--eval_dir", required=True, type=Path, help="含 step_*/ 的动作对比图目录")
    p.add_argument("--out_dir", type=Path, default=None)
    p.add_argument("--duration_ms", type=int, default=800)
    args = p.parse_args()
    out_dir = args.out_dir or (args.eval_dir / "evolution")
    out_dir.mkdir(parents=True, exist_ok=True)

    # 按样本（ep..frame..）分组，每组按 step 排序
    groups: dict[str, list[tuple[int, Path]]] = defaultdict(list)
    for png in args.eval_dir.glob("step_*/*.png"):
        stem = png.stem  # ep95_frame100
        groups[stem].append((_step_of(png), png))

    if not groups:
        print(f"没找到任何 step_*/*.png：{args.eval_dir}")
        return

    all_pages: list[tuple[str, int, Path]] = []
    for stem, items in sorted(groups.items()):
        items.sort(key=lambda t: t[0])
        frames = [Image.open(pth).convert("RGB") for _, pth in items]
        if frames:
            gif_path = out_dir / f"{stem}.gif"
            frames[0].save(
                gif_path, save_all=True, append_images=frames[1:],
                duration=args.duration_ms, loop=0,
            )
            print(f"GIF: {gif_path}  ({len(frames)} steps)")
        for step, pth in items:
            all_pages.append((stem, step, pth))

    # 多页 PDF（按 step 再按样本）
    all_pages.sort(key=lambda t: (t[1], t[0]))
    pdf_imgs = [Image.open(pth).convert("RGB") for _, _, pth in all_pages]
    if pdf_imgs:
        pdf_path = out_dir / "action_evolution.pdf"
        pdf_imgs[0].save(pdf_path, save_all=True, append_images=pdf_imgs[1:])
        print(f"PDF: {pdf_path}  ({len(pdf_imgs)} pages)")


if __name__ == "__main__":
    main()
