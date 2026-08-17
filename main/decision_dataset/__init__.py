"""決策資料集深 module；GUI 與 CLI 共用此 interface。"""

from pathlib import Path

from ._shared import (
    DEFECT_CLASSES,
    EXPECTED_CLASSES,
    EXPECTED_CLASS_VARIANTS,
    FIELDNAMES,
    SPLITS,
    VIEWS,
    classes_match_role,
    load_yaml,
    normalize_classes,
    sha256_file,
    write_yaml_atomic,
)
from .generation import (
    build_dataset_manifest,
    dataset_schema_markdown,
    generate_dataset,
    generate_decision_dataset,
    read_dataset_rows,
    report_to_row,
    validate_generation,
    write_dataset_rows,
)
from .layout import apply_layout, plan_layout, validate_dataset_layout
from .manifest import (
    ROLES,
    approve_model,
    inspect_model,
    load_or_create_manifest,
    update_manifest,
    validate_manifest,
    validate_model_bundle,
    validate_model_record,
)
from .types import (
    FruitInput,
    GenerationMode,
    GenerationProgress,
    GenerationRequest,
    GenerationResult,
    LayoutPlan,
    build_generation_request,
)


def plan_layout_migration(dataset_root: str | Path) -> tuple[list[dict], list[dict]]:
    plan = plan_layout(dataset_root)
    return list(plan.moves), list(plan.errors)


def apply_layout_migration(dataset_root: str | Path, moves: list[dict]) -> Path:
    return apply_layout(
        LayoutPlan(Path(dataset_root).resolve(), tuple(moves), tuple())
    )


__all__ = [
    "DEFECT_CLASSES",
    "EXPECTED_CLASSES",
    "EXPECTED_CLASS_VARIANTS",
    "FIELDNAMES",
    "FruitInput",
    "GenerationMode",
    "GenerationProgress",
    "GenerationRequest",
    "GenerationResult",
    "LayoutPlan",
    "ROLES",
    "SPLITS",
    "VIEWS",
    "apply_layout",
    "apply_layout_migration",
    "approve_model",
    "build_generation_request",
    "build_dataset_manifest",
    "classes_match_role",
    "dataset_schema_markdown",
    "generate_dataset",
    "generate_decision_dataset",
    "inspect_model",
    "load_or_create_manifest",
    "load_yaml",
    "normalize_classes",
    "plan_layout",
    "plan_layout_migration",
    "read_dataset_rows",
    "report_to_row",
    "sha256_file",
    "update_manifest",
    "validate_dataset_layout",
    "validate_generation",
    "validate_manifest",
    "validate_model_bundle",
    "validate_model_record",
    "write_dataset_rows",
    "write_yaml_atomic",
]
