"""解释器启动时自动加载（当本目录在 PYTHONPATH 上时）。

LeRobot fork（~/Projects/lerobot，Jul 23 后）使用 py3.11+ 的 `typing.Self`，
而 cosmos-policy 的 uv .venv 是 py3.10。这里在任何 user import 之前把 `Self`
从 typing_extensions 回填到 typing，避免修改上游 lerobot / cosmos 源码。

只有把 finetune_three_cubes 加入 PYTHONPATH 时才生效，作用范围最小。
"""

import typing

if not hasattr(typing, "Self"):  # py < 3.11
    try:
        from typing_extensions import Self as _Self

        typing.Self = _Self  # type: ignore[attr-defined]
    except Exception:  # pragma: no cover - typing_extensions 不在时静默跳过
        pass
