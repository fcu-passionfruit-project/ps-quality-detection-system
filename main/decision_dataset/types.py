from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Callable


class GenerationMode(str, Enum):
    RESUME = "resume"
    ONLY_ERRORS = "only_errors"
    RERUN_FRUIT = "rerun_fruit"
    RERUN_VIEW = "rerun_view"
    REPLACE_IMAGE = "replace_image"


@dataclass(frozen=True)
class FruitInput:
    split: str
    fruit_id: int
    images: dict[int, Path]


@dataclass(frozen=True)
class LayoutPlan:
    dataset_root: Path
    moves: tuple[dict, ...]
    errors: tuple[dict, ...]

    @property
    def ready(self) -> bool:
        return not self.errors


@dataclass(frozen=True)
class GenerationRequest:
    repo_root: Path
    dataset_root: Path
    model_manifest_path: Path
    output_root: Path
    dataset_version: str
    mode: GenerationMode = GenerationMode.RESUME
    fruit_id: int | None = None
    view_id: int | None = None
    labels_from: str | None = None
    validate_only: bool = False
    load_weights: bool = True

    def engine_flags(self) -> dict:
        if self.mode == GenerationMode.ONLY_ERRORS:
            return {"only_errors": True, "force": False, "replace_image": False}
        if self.mode in {GenerationMode.RERUN_FRUIT, GenerationMode.RERUN_VIEW}:
            return {"only_errors": False, "force": True, "replace_image": False}
        if self.mode == GenerationMode.REPLACE_IMAGE:
            return {"only_errors": False, "force": True, "replace_image": True}
        return {"only_errors": False, "force": False, "replace_image": False}

    def validate_mode(self) -> None:
        if self.mode in {
            GenerationMode.RERUN_FRUIT,
            GenerationMode.RERUN_VIEW,
            GenerationMode.REPLACE_IMAGE,
        } and self.fruit_id is None:
            raise ValueError(f"{self.mode.value} 必須指定 fruit_id。")
        if self.mode == GenerationMode.RERUN_VIEW and self.view_id is None:
            raise ValueError("rerun_view 必須指定 view_id。")
        if self.view_id is not None and self.view_id not in {1, 2, 3}:
            raise ValueError("view_id 必須是 1、2 或 3。")
        if self.mode in {GenerationMode.RESUME, GenerationMode.ONLY_ERRORS} and (
            self.fruit_id is not None or self.view_id is not None
        ):
            raise ValueError(f"{self.mode.value} 不可指定 fruit_id 或 view_id。")


@dataclass(frozen=True)
class GenerationProgress:
    phase: str
    processed: int
    total: int
    errors: int
    fruit_id: int | None = None
    message: str = ""


@dataclass(frozen=True)
class GenerationResult:
    ok: bool
    processed: int
    skipped: int
    processed_errors: int
    errors: int
    layout_errors: int
    output: str
    stopped: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


ProgressCallback = Callable[[GenerationProgress], None]
CancelCheck = Callable[[], bool]


def build_generation_request(
    *,
    repo_root: str | Path,
    dataset_root: str | Path,
    model_manifest_path: str | Path,
    output_root: str | Path,
    dataset_version: str,
    mode: str | GenerationMode = GenerationMode.RESUME,
    fruit_id: int | str | None = None,
    view_id: int | str | None = None,
    labels_from: str | None = None,
    validate_only: bool = False,
    load_weights: bool = True,
) -> GenerationRequest:
    """將 GUI／CLI 的字串輸入收斂成同一個 request。"""

    request = GenerationRequest(
        repo_root=Path(repo_root).expanduser().resolve(),
        dataset_root=Path(dataset_root).expanduser().resolve(),
        model_manifest_path=Path(model_manifest_path).expanduser().resolve(),
        output_root=Path(output_root).expanduser().resolve(),
        dataset_version=dataset_version.strip(),
        mode=mode if isinstance(mode, GenerationMode) else GenerationMode(mode),
        fruit_id=int(fruit_id) if fruit_id not in (None, "") else None,
        view_id=int(view_id) if view_id not in (None, "") else None,
        labels_from=(labels_from or "").strip() or None,
        validate_only=validate_only,
        load_weights=load_weights,
    )
    request.validate_mode()
    return request
