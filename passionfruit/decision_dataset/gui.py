from __future__ import annotations

import json
import threading
import tkinter as tk
from dataclasses import dataclass
from pathlib import Path
from tkinter import filedialog, messagebox, scrolledtext, ttk
from typing import Callable

from . import (
    GenerationMode,
    GenerationProgress,
    LayoutPlan,
    ROLES,
    apply_layout,
    approve_model,
    build_generation_request,
    generate_dataset,
    inspect_model,
    load_or_create_manifest,
    plan_layout,
    update_manifest,
    validate_generation,
    validate_manifest,
)


MODE_LABELS = {
    "補齊缺少列": GenerationMode.RESUME,
    "只重試錯誤": GenerationMode.ONLY_ERRORS,
    "重跑整顆": GenerationMode.RERUN_FRUIT,
    "重跑單張": GenerationMode.RERUN_VIEW,
    "替換照片後重跑": GenerationMode.REPLACE_IMAGE,
}


@dataclass
class LayoutPreviewState:
    plan: LayoutPlan | None = None

    @property
    def can_apply(self) -> bool:
        return self.plan is not None and self.plan.ready

    def invalidate(self) -> None:
        self.plan = None


def generation_request_from_gui(values: dict[str, str], repo_root: Path):
    return build_generation_request(
        repo_root=repo_root,
        dataset_root=values["dataset_root"],
        model_manifest_path=values["manifest"],
        output_root=values["output_root"],
        dataset_version=values["version"],
        mode=MODE_LABELS[values["mode"]],
        fruit_id=values.get("fruit_id"),
        view_id=values.get("view_id"),
        labels_from=values.get("labels_from"),
    )


class DecisionDatasetPanel:
    """Tkinter adapter；domain 操作全部委派給 decision_dataset package。"""

    def __init__(
        self,
        parent,
        *,
        repo_root: Path,
        manager,
        run_async: Callable,
        post_ui: Callable[[Callable], None],
    ):
        self.repo_root = repo_root.resolve()
        self.manager = manager
        self.run_async = run_async
        self.post_ui = post_ui
        self.cancel_event = threading.Event()
        self.layout_state = LayoutPreviewState()

        notebook = ttk.Notebook(parent)
        notebook.pack(fill="both", expand=True)
        layout_tab = ttk.Frame(notebook, padding=10)
        manifest_tab = ttk.Frame(notebook, padding=10)
        generate_tab = ttk.Frame(notebook, padding=10)
        notebook.add(layout_tab, text="資料結構")
        notebook.add(manifest_tab, text="Model Manifest")
        notebook.add(generate_tab, text="產生 CSV")
        self._build_layout_tab(layout_tab)
        self._build_manifest_tab(manifest_tab)
        self._build_generate_tab(generate_tab)

    @staticmethod
    def _json(value: object) -> str:
        return json.dumps(value, ensure_ascii=False, indent=2, default=str)

    @staticmethod
    def _replace_text(widget, text: str) -> None:
        widget.delete("1.0", tk.END)
        widget.insert(tk.END, text)

    @staticmethod
    def _browse_dir(variable: tk.StringVar) -> None:
        selected = filedialog.askdirectory(initialdir=variable.get() or None)
        if selected:
            variable.set(selected)

    @staticmethod
    def _browse_file(variable: tk.StringVar, pattern: str) -> None:
        selected = filedialog.askopenfilename(filetypes=[("檔案", pattern)])
        if selected:
            variable.set(selected)

    def _path_row(self, parent, row, label, variable, command):
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=3)
        ttk.Entry(parent, textvariable=variable).grid(row=row, column=1, sticky="ew", padx=6, pady=3)
        ttk.Button(parent, text="選擇…", command=command).grid(row=row, column=2, pady=3)
        parent.columnconfigure(1, weight=1)

    def _build_layout_tab(self, tab):
        self.layout_root = tk.StringVar(value=str(self.repo_root / "DecisionLayerTrain"))
        self.layout_root.trace_add("write", lambda *_: self._invalidate_layout())
        self._path_row(tab, 0, "DecisionLayerTrain", self.layout_root, lambda: self._browse_dir(self.layout_root))
        buttons = ttk.Frame(tab)
        buttons.grid(row=1, column=0, columnspan=3, sticky="w", pady=8)
        self.layout_preview_button = ttk.Button(buttons, text="預覽整理", command=self._preview_layout)
        self.layout_preview_button.pack(side="left")
        self.layout_apply_button = ttk.Button(buttons, text="套用整理", state="disabled", command=self._apply_layout)
        self.layout_apply_button.pack(side="left", padx=8)
        self.layout_output = scrolledtext.ScrolledText(tab, height=25, wrap="word")
        self.layout_output.grid(row=2, column=0, columnspan=3, sticky="nsew")
        tab.rowconfigure(2, weight=1)

    def _invalidate_layout(self):
        self.layout_state.invalidate()
        if hasattr(self, "layout_apply_button"):
            self.layout_apply_button.config(state="disabled")

    def _preview_layout(self):
        root = self.layout_root.get()

        def done(plan):
            self.layout_state.plan = plan
            self.layout_apply_button.config(state="normal" if plan.ready else "disabled")
            self._replace_text(self.layout_output, self._json({"moves": list(plan.moves), "errors": list(plan.errors)}))

        self.run_async(self.layout_preview_button, lambda: plan_layout(root), done)

    def _apply_layout(self):
        plan = self.layout_state.plan
        if not self.layout_state.can_apply or plan is None:
            messagebox.showerror("無有效預覽", "請先對目前路徑完成成功預覽。")
            return
        if not messagebox.askyesno("確認整理", f"確定搬移 {len(plan.moves)} 個果實資料夾？"):
            return
        self.run_async(self.layout_apply_button, lambda: apply_layout(plan), lambda path: self._replace_text(self.layout_output, f"整理完成。\n對照表：{path}"))

    def _build_manifest_tab(self, tab):
        self.manifest_path = tk.StringVar(value=str(self.repo_root / "models" / "manifest.yaml"))
        self.manifest_role = tk.StringVar(value=ROLES[0])
        self.manifest_model = tk.StringVar()
        self.manifest_imgsz = tk.StringVar(value="640")
        self.manifest_threshold = tk.StringVar(value="0.25")
        self._path_row(tab, 0, "Manifest", self.manifest_path, lambda: self._browse_file(self.manifest_path, "*.yaml *.yml"))
        ttk.Label(tab, text="Role").grid(row=1, column=0, sticky="w")
        ttk.Combobox(tab, textvariable=self.manifest_role, values=ROLES, state="readonly").grid(row=1, column=1, sticky="ew", padx=6)
        self._path_row(tab, 2, "權重", self.manifest_model, lambda: self._browse_file(self.manifest_model, "*.pt"))
        ttk.Label(tab, text="imgsz").grid(row=3, column=0, sticky="w")
        ttk.Entry(tab, textvariable=self.manifest_imgsz).grid(row=3, column=1, sticky="ew", padx=6)
        ttk.Label(tab, text="threshold（分類模型留空）").grid(row=4, column=0, sticky="w")
        ttk.Entry(tab, textvariable=self.manifest_threshold).grid(row=4, column=1, sticky="ew", padx=6)
        buttons = ttk.Frame(tab)
        buttons.grid(row=5, column=0, columnspan=3, sticky="w", pady=8)
        for text, command in (("顯示", self._show_manifest), ("檢查權重", self._inspect_manifest_model), ("更新 candidate", self._update_manifest), ("核准 role", self._approve_manifest), ("驗證 Bundle", self._validate_manifest)):
            ttk.Button(buttons, text=text, command=command).pack(side="left", padx=(0, 6))
        self.manifest_output = scrolledtext.ScrolledText(tab, height=20, wrap="word")
        self.manifest_output.grid(row=6, column=0, columnspan=3, sticky="nsew")
        tab.rowconfigure(6, weight=1)

    def _threshold(self):
        value = self.manifest_threshold.get().strip()
        return float(value) if value else None

    def _run_manifest(self, job):
        # Text widget 也支援 state config，作為既有 async adapter 的鎖定目標。
        self.run_async(self.manifest_output, job, lambda value: self._replace_text(self.manifest_output, self._json(value)))

    def _show_manifest(self):
        path = self.manifest_path.get()
        self._run_manifest(lambda: load_or_create_manifest(path))

    def _inspect_manifest_model(self):
        path, role, model = self.manifest_path.get(), self.manifest_role.get(), self.manifest_model.get()
        imgsz, threshold = int(self.manifest_imgsz.get()), self._threshold()
        self._run_manifest(lambda: inspect_model(self.repo_root, role, model, imgsz, threshold))

    def _update_manifest(self):
        path, role, model = self.manifest_path.get(), self.manifest_role.get(), self.manifest_model.get()
        imgsz, threshold = int(self.manifest_imgsz.get()), self._threshold()
        self._run_manifest(lambda: update_manifest(path, self.repo_root, role, model, imgsz, threshold))

    def _approve_manifest(self):
        path, role = self.manifest_path.get(), self.manifest_role.get()
        self._run_manifest(lambda: approve_model(path, self.repo_root, role))

    def _validate_manifest(self):
        path = self.manifest_path.get()
        def job():
            records, errors = validate_manifest(path, self.repo_root)
            return {"ok": not errors, "models": records, "errors": errors}
        self._run_manifest(job)

    def _build_generate_tab(self, tab):
        defaults = {
            "dataset_root": str(self.repo_root / "DecisionLayerTrain"),
            "manifest": str(self.repo_root / "models" / "manifest.yaml"),
            "output_root": str(self.repo_root / "decision_datasets"),
            "version": "v001",
            "labels_from": "",
            "mode": next(iter(MODE_LABELS)),
            "fruit_id": "",
            "view_id": "",
        }
        self.generate_vars = {key: tk.StringVar(value=value) for key, value in defaults.items()}
        for row, (key, label) in enumerate((("dataset_root", "資料集"), ("manifest", "Manifest"), ("output_root", "輸出根目錄"))):
            self._path_row(tab, row, label, self.generate_vars[key], lambda var=self.generate_vars[key], key=key: self._browse_file(var, "*.yaml *.yml") if key == "manifest" else self._browse_dir(var))
        for row, (key, label) in enumerate((("version", "版本"), ("labels_from", "labels-from"), ("fruit_id", "fruit_id"), ("view_id", "view_id")), start=3):
            ttk.Label(tab, text=label).grid(row=row, column=0, sticky="w")
            ttk.Entry(tab, textvariable=self.generate_vars[key]).grid(row=row, column=1, sticky="ew", padx=6, pady=2)
        ttk.Label(tab, text="操作模式").grid(row=7, column=0, sticky="w")
        ttk.Combobox(tab, textvariable=self.generate_vars["mode"], values=tuple(MODE_LABELS), state="readonly").grid(row=7, column=1, sticky="ew", padx=6)
        buttons = ttk.Frame(tab)
        buttons.grid(row=8, column=0, columnspan=3, sticky="w", pady=8)
        self.generate_validate_button = ttk.Button(buttons, text="只驗證", command=self._validate_generation)
        self.generate_validate_button.pack(side="left")
        self.generate_start_button = ttk.Button(buttons, text="開始產生", command=self._start_generation)
        self.generate_start_button.pack(side="left", padx=6)
        self.generate_stop_button = ttk.Button(buttons, text="安全停止", state="disabled", command=self._stop_generation)
        self.generate_stop_button.pack(side="left")
        self.progress_text = tk.StringVar(value="尚未開始")
        ttk.Label(tab, textvariable=self.progress_text).grid(row=9, column=0, columnspan=3, sticky="w")
        self.progressbar = ttk.Progressbar(tab, mode="determinate")
        self.progressbar.grid(row=10, column=0, columnspan=3, sticky="ew", pady=5)
        self.generate_output = scrolledtext.ScrolledText(tab, height=12, wrap="word")
        self.generate_output.grid(row=11, column=0, columnspan=3, sticky="nsew")
        tab.rowconfigure(11, weight=1)

    def _request(self):
        return generation_request_from_gui({key: value.get() for key, value in self.generate_vars.items()}, self.repo_root)

    def _validate_generation(self):
        try:
            request = self._request()
        except Exception as error:
            messagebox.showerror("輸入錯誤", str(error))
            return
        self.run_async(self.generate_validate_button, lambda: validate_generation(request), lambda value: self._replace_text(self.generate_output, self._json(value)))

    def _progress(self, progress: GenerationProgress):
        self.post_ui(lambda progress=progress: self._show_progress(progress))

    def _show_progress(self, progress: GenerationProgress):
        self.progressbar["maximum"] = max(progress.total, 1)
        self.progressbar["value"] = progress.processed
        fruit = f"，目前 fruit_id={progress.fruit_id}" if progress.fruit_id is not None else ""
        self.progress_text.set(f"{progress.processed}/{progress.total}{fruit}，錯誤 {progress.errors}；{progress.message}")

    def _start_generation(self):
        self.cancel_event.clear()
        try:
            request = self._request()
        except Exception as error:
            messagebox.showerror("輸入錯誤", str(error))
            return
        self.generate_stop_button.config(state="normal")

        def done(result):
            self.generate_stop_button.config(state="disabled")
            self._replace_text(self.generate_output, self._json(result.to_dict()))

        def job():
            try:
                return generate_dataset(request, manager=self.manager, progress=self._progress, cancel_requested=self.cancel_event.is_set)
            finally:
                self.post_ui(lambda: self.generate_stop_button.config(state="disabled"))

        self.run_async(self.generate_start_button, job, done)

    def _stop_generation(self):
        self.cancel_event.set()
        self.generate_stop_button.config(state="disabled")
        self.progress_text.set("已要求安全停止；目前果實完成 checkpoint 後停止。")
