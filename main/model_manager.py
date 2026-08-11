from __future__ import annotations
from pathlib import Path
import torch
from ultralytics import YOLO
ROLE_TASKS = {
    "roi": "detect",
    "color": "classify",
    "wrinkle": "classify",
    "defect": "segment",
}
class ModelTaskMismatch(ValueError):
    pass
class PassionFruitModelManager:
    """依角色延遲載入 YOLO 模型，並統一選擇推論裝置。"""
    def __init__(self, device: str | None = None):
        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        self._models: dict[str, tuple[Path, YOLO]] = {}
    def load(self, role: str, model_path: str | Path) -> YOLO:
        if role not in ROLE_TASKS:
            raise ValueError(f"未知的模型角色：{role}")
        path = Path(model_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"找不到 {role} 模型：{path}")
        cached = self._models.get(role)
        if cached and cached[0] == path:
            return cached[1]
        model = YOLO(str(path))
        expected_task = ROLE_TASKS[role]
        actual_task = getattr(model, "task", None)
        if actual_task != expected_task:
            raise ModelTaskMismatch(
                f"{role} 應為 {expected_task} 模型，目前權重是 {actual_task or 'unknown'}。"
            )
        self._models[role] = (path, model)
        return model
    def predict(
        self,
        role: str,
        model_path: str | Path,
        image,
        *,
        confidence: float | None = None,
    ):
        model = self.load(role, model_path)
        options = {"device": self.device, "verbose": False}
        if confidence is not None:
            options["conf"] = float(confidence)
        return model.predict(source=image, **options)[0]
    def predict_batch(
        self,
        role: str,
        model_path: str | Path,
        images: list,
        *,
        confidence: float | None = None,
    ) -> list:
        if not images:
            return []
        model = self.load(role, model_path)
        options = {"device": self.device, "verbose": False}
        if confidence is not None:
            options["conf"] = float(confidence)
        results = model.predict(source=list(images), **options)
        if len(results) != len(images):
            raise RuntimeError(
                f"{role} batch 推論回傳數量（{len(results)}）與輸入數量"
                f"（{len(images)}）不一致。"
            )
        return results
    def model_info(self, role: str) -> dict:
        cached = self._models.get(role)
        return {
            "role": role,
            "task": ROLE_TASKS[role],
            "path": str(cached[0]) if cached else None,
            "device": self.device,
        }
