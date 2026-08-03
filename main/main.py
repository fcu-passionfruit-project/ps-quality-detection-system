from __future__ import annotations

import json
import queue
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, scrolledtext, ttk

import cv2
from PIL import Image, ImageTk

from image_processing import DEFAULT_PARAMETERS, batch_preprocess
from model_manager import PassionFruitModelManager
from pipeline import ROLE_LABELS, run_pipeline, run_single_model, save_validation_result


ROLE_OPTIONS = {
    "ROI 定位": "roi",
    "顏色分類": "color",
    "皺褶分類": "wrinkle",
    "局部瑕疵": "defect",
}
IMAGE_FILE_TYPES = [("圖片", "*.jpg *.jpeg *.png *.bmp *.webp")]
MODEL_FILE_TYPES = [("YOLO 權重", "*.pt")]


def _format_stage_summary(role: str, result: dict | None) -> str:
    label = ROLE_LABELS[role]
    if not result:
        return f"{label}：無結果"

    status = result.get("status")
    if status != "ok":
        status_text = {
            "error": "失敗",
            "skipped": "略過",
        }.get(status, str(status or "無結果"))
        reason = result.get("reason")
        return f"{label}：{status_text}" + (f"（{reason}）" if reason else "")

    if role == "defect":
        detections = result.get("detections") or []
        if not detections:
            return f"{label}：未偵測到"
        values = []
        for detection in detections:
            value = str(detection.get("class_name", "未知類別"))
            confidence = detection.get("confidence")
            if confidence is not None:
                value += f"，信心值：{float(confidence):.6f}"
            values.append(value)
        return f"{label}：" + "、".join(values)

    class_name = "PF" if role == "roi" else result.get("class_name", "未知類別")
    confidence = result.get("confidence")
    confidence_text = (
        f"，信心值：{float(confidence):.6f}" if confidence is not None else ""
    )
    return f"{label}：{class_name}{confidence_text}"


def _validation_summary(report: dict) -> str:
    parts = []
    roi = report.get("roi")
    if roi is not None:
        parts.append(_format_stage_summary("roi", roi))

    if report.get("mode") == "pipeline":
        models = report.get("models", {})
        parts.extend(
            _format_stage_summary(role, models.get(role))
            for role in ("color", "wrinkle", "defect")
        )
    else:
        role = report.get("role")
        if role == "roi" and roi is None:
            parts.append(_format_stage_summary("roi", report.get("model")))
        elif role in ROLE_LABELS and role != "roi":
            parts.append(_format_stage_summary(role, report.get("model")))

    return "　｜　".join(parts) or "尚無模型摘要"


class PassionFruitApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("百香果 YOLO 模型測試與影像前處理工具")
        self.root.geometry("1450x900")
        self.root.minsize(1100, 720)
        self.manager = PassionFruitModelManager()
        self.events: queue.Queue = queue.Queue()
        self.active_job = None
        self.image_refs = []
        self.parameters = {
            key: tk.StringVar(value=str(value))
            for key, value in DEFAULT_PARAMETERS.items()
        }
        self._build_ui()
        self.root.after(80, self._poll_events)

    def _build_ui(self):
        header = ttk.Frame(self.root, padding=(12, 8))
        header.pack(fill="x")
        ttk.Label(
            header,
            text="百香果 YOLO 測試工具",
            font=("Microsoft JhengHei UI", 18, "bold"),
        ).pack(side="left")
        ttk.Label(
            header,
            text=f"推論裝置：{self.manager.device}　｜　執行環境：PF",
            foreground="#0b6e4f",
        ).pack(side="right")

        notebook = ttk.Notebook(self.root)
        notebook.pack(fill="both", expand=True, padx=12, pady=(0, 12))
        batch_tab = ttk.Frame(notebook, padding=12)
        single_tab = ttk.Frame(notebook, padding=12)
        pipeline_tab = ttk.Frame(notebook, padding=12)
        notebook.add(batch_tab, text="1. 批次前處理")
        notebook.add(single_tab, text="2. 單模型驗證")
        notebook.add(pipeline_tab, text="3. 多模型 Pipeline")
        self._build_batch_tab(batch_tab)
        self._build_single_tab(single_tab)
        self._build_pipeline_tab(pipeline_tab)

    def _path_row(
        self,
        parent,
        row: int,
        label: str,
        variable: tk.StringVar,
        *,
        directory: bool = False,
        model: bool = False,
    ):
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=4)
        ttk.Entry(parent, textvariable=variable).grid(
            row=row, column=1, sticky="ew", padx=8, pady=4
        )
        if directory:
            command = lambda: self._choose_directory(variable)
        else:
            command = lambda: self._choose_file(
                variable, MODEL_FILE_TYPES if model else IMAGE_FILE_TYPES
            )
        ttk.Button(parent, text="選擇…", command=command).grid(
            row=row, column=2, pady=4
        )
        parent.columnconfigure(1, weight=1)

    def _parameter_frame(self, parent):
        frame = ttk.LabelFrame(parent, text="前處理參數", padding=8)
        definitions = [
            (
                "confidence",
                "ROI confidence",
                "推薦 0.30；提高可減少誤判，但可能漏掉果實。",
            ),
            (
                "shrink_ratio",
                "邊界內縮比例",
                "推薦 0.00；提高會減少背景，也可能裁掉果皮。",
            ),
            (
                "mask_scale",
                "圓形遮罩倍率",
                "推薦 1.01；提高會保留更多角落背景。",
            ),
            (
                "clahe_clip_limit",
                "CLAHE 對比限制",
                "推薦 2.0；提高會強化皺褶，也會增加雜訊。",
            ),
            (
                "clahe_tile_size",
                "CLAHE 網格",
                "推薦 8；提高會使用較粗的局部對比區域。",
            ),
        ]
        for row, (key, label, description) in enumerate(definitions):
            ttk.Label(frame, text=label).grid(row=row, column=0, sticky="w", pady=3)
            ttk.Entry(frame, textvariable=self.parameters[key], width=9).grid(
                row=row, column=1, padx=8, pady=3
            )
            ttk.Label(frame, text=description, foreground="#555").grid(
                row=row, column=2, sticky="w", pady=3
            )
        frame.columnconfigure(2, weight=1)
        return frame

    def _build_batch_tab(self, tab):
        paths = ttk.LabelFrame(tab, text="資料來源", padding=8)
        paths.pack(fill="x")
        self.batch_model = tk.StringVar()
        self.batch_input = tk.StringVar()
        self.batch_output = tk.StringVar()
        self.batch_mode = tk.StringVar(value="color")
        self._path_row(paths, 0, "ROI 模型", self.batch_model, model=True)
        self._path_row(paths, 1, "輸入資料夾", self.batch_input, directory=True)
        self._path_row(paths, 2, "輸出資料夾", self.batch_output, directory=True)
        mode = ttk.Frame(paths)
        mode.grid(row=3, column=1, sticky="w", pady=5)
        ttk.Radiobutton(
            mode, text="彩色 ROI", variable=self.batch_mode, value="color"
        ).pack(side="left")
        ttk.Radiobutton(
            mode, text="皺褶 ROI（灰階＋CLAHE）",
            variable=self.batch_mode,
            value="wrinkle",
        ).pack(side="left", padx=18)

        self._parameter_frame(tab).pack(fill="x", pady=10)
        self.batch_button = ttk.Button(
            tab, text="開始批次前處理", command=self._start_batch
        )
        self.batch_button.pack(anchor="w")
        self.batch_log = scrolledtext.ScrolledText(tab, height=16, wrap="word")
        self.batch_log.pack(fill="both", expand=True, pady=(8, 0))

    def _build_single_tab(self, tab):
        controls = ttk.LabelFrame(tab, text="單模型設定", padding=8)
        controls.pack(fill="x")
        self.single_role_label = tk.StringVar(value="ROI 定位")
        self.single_mode = tk.StringVar(value="ready")
        self.single_model = tk.StringVar()
        self.single_roi_model = tk.StringVar()
        self.single_image = tk.StringVar()
        self.single_output = tk.StringVar()
        ttk.Label(controls, text="模型角色").grid(row=0, column=0, sticky="w")
        ttk.Combobox(
            controls,
            textvariable=self.single_role_label,
            values=list(ROLE_OPTIONS),
            state="readonly",
            width=18,
        ).grid(row=0, column=1, sticky="w", padx=8)
        modes = ttk.Frame(controls)
        modes.grid(row=1, column=1, sticky="w", pady=4)
        ttk.Radiobutton(
            modes,
            text="模型就緒影像",
            variable=self.single_mode,
            value="ready",
        ).pack(side="left")
        ttk.Radiobutton(
            modes,
            text="完整原圖＋輔助 ROI",
            variable=self.single_mode,
            value="raw",
        ).pack(side="left", padx=18)
        self._path_row(controls, 2, "待測模型", self.single_model, model=True)
        self._path_row(
            controls, 3, "輔助 ROI 模型", self.single_roi_model, model=True
        )
        self._path_row(controls, 4, "輸入圖片", self.single_image)
        self._path_row(
            controls, 5, "驗證輸出資料夾", self.single_output, directory=True
        )
        self._parameter_frame(tab).pack(fill="x", pady=10)
        self.single_button = ttk.Button(
            tab, text="執行單模型驗證", command=self._start_single
        )
        self.single_button.pack(anchor="w")
        (
            self.single_result,
            self.single_summary,
            self.single_previews,
        ) = self._result_area(tab, 4)

    def _build_pipeline_tab(self, tab):
        controls = ttk.LabelFrame(tab, text="Pipeline 模型（可個別留空）", padding=8)
        controls.pack(fill="x")
        self.pipeline_models = {
            role: tk.StringVar() for role in ("roi", "color", "wrinkle", "defect")
        }
        for row, (role, label) in enumerate(
            (
                ("roi", "ROI 模型"),
                ("color", "顏色模型"),
                ("wrinkle", "皺褶模型"),
                ("defect", "局部瑕疵模型"),
            )
        ):
            self._path_row(
                controls, row, label, self.pipeline_models[role], model=True
            )
        self.pipeline_image = tk.StringVar()
        self.pipeline_output = tk.StringVar()
        self._path_row(controls, 4, "完整原圖", self.pipeline_image)
        self._path_row(
            controls, 5, "驗證輸出資料夾", self.pipeline_output, directory=True
        )
        self._parameter_frame(tab).pack(fill="x", pady=10)
        self.pipeline_button = ttk.Button(
            tab, text="執行多模型 Pipeline", command=self._start_pipeline
        )
        self.pipeline_button.pack(anchor="w")
        (
            self.pipeline_result,
            self.pipeline_summary,
            self.pipeline_previews,
        ) = self._result_area(tab, 5)

    def _result_area(self, parent, preview_count: int):
        container = ttk.Panedwindow(parent, orient="vertical")
        container.pack(fill="both", expand=True, pady=(8, 0))
        result = scrolledtext.ScrolledText(container, height=9, wrap="word")
        preview_frame = ttk.Frame(container)
        summary = ttk.Label(
            preview_frame,
            text="模型摘要：尚無結果",
            anchor="w",
            justify="left",
            wraplength=1000,
            font=("Microsoft JhengHei UI", 10, "bold"),
        )
        summary.grid(
            row=0,
            column=0,
            columnspan=preview_count,
            sticky="ew",
            padx=3,
            pady=(0, 6),
        )
        preview_frame.bind(
            "<Configure>",
            lambda event: summary.config(wraplength=max(event.width - 12, 1)),
        )
        labels = []
        for column in range(preview_count):
            label = ttk.Label(
                preview_frame,
                text="尚無結果",
                anchor="center",
                relief="solid",
                padding=4,
                takefocus=False,
            )
            label.grid(row=1, column=column, sticky="nsew", padx=3)
            preview_frame.columnconfigure(column, weight=1)
            labels.append(label)
        preview_frame.rowconfigure(1, weight=1)
        container.add(result, weight=1)
        container.add(preview_frame, weight=2)
        return result, summary, labels

    def _parameter_values(self) -> dict:
        try:
            return {
                "confidence": float(self.parameters["confidence"].get()),
                "shrink_ratio": float(self.parameters["shrink_ratio"].get()),
                "mask_scale": float(self.parameters["mask_scale"].get()),
                "clahe_clip_limit": float(
                    self.parameters["clahe_clip_limit"].get()
                ),
                "clahe_tile_size": int(self.parameters["clahe_tile_size"].get()),
            }
        except ValueError as error:
            raise ValueError("前處理參數必須是有效數字。") from error

    def _start_batch(self):
        try:
            parameters = self._parameter_values()
        except ValueError as error:
            messagebox.showerror("參數錯誤", str(error))
            return
        values = {
            "model": self.batch_model.get(),
            "input": self.batch_input.get(),
            "output": self.batch_output.get(),
            "mode": self.batch_mode.get(),
            "parameters": parameters,
        }
        if not all((values["model"], values["input"], values["output"])):
            messagebox.showwarning("資料未完整", "請選擇模型、輸入及輸出資料夾。")
            return

        def job():
            self.manager.load("roi", values["model"])

            def detect(image, confidence):
                return self.manager.predict(
                    "roi", values["model"], image, confidence=confidence
                )

            def progress(current, total, status):
                self.events.put(
                    ("log", f"{current}/{total}：{status}", self.batch_log)
                )

            return batch_preprocess(
                detect,
                values["input"],
                values["output"],
                model_path=values["model"],
                mode=values["mode"],
                progress=progress,
                **values["parameters"],
            )

        self.batch_log.delete("1.0", tk.END)
        self._run_async(
            self.batch_button,
            job,
            lambda report: self._show_batch_result(report),
        )

    def _start_single(self):
        try:
            parameters = self._parameter_values()
        except ValueError as error:
            messagebox.showerror("參數錯誤", str(error))
            return
        values = {
            "role": ROLE_OPTIONS[self.single_role_label.get()],
            "mode": self.single_mode.get(),
            "model": self.single_model.get(),
            "roi_model": self.single_roi_model.get(),
            "image": self.single_image.get(),
            "output": self.single_output.get(),
            "parameters": parameters,
        }
        if not all((values["model"], values["image"], values["output"])):
            messagebox.showwarning("資料未完整", "請選擇待測模型、圖片及輸出資料夾。")
            return

        def job():
            report, artifacts = run_single_model(
                values["image"],
                values["role"],
                values["model"],
                input_mode=values["mode"],
                auxiliary_roi_path=values["roi_model"],
                parameters=values["parameters"],
                manager=self.manager,
            )
            run_dir = save_validation_result(
                values["output"], report, artifacts
            )
            return report, artifacts, run_dir

        self._run_async(
            self.single_button,
            job,
            lambda result: self._show_validation(
                result,
                self.single_result,
                self.single_summary,
                self.single_previews,
            ),
        )

    def _start_pipeline(self):
        try:
            parameters = self._parameter_values()
        except ValueError as error:
            messagebox.showerror("參數錯誤", str(error))
            return
        values = {
            "models": {
                role: variable.get()
                for role, variable in self.pipeline_models.items()
            },
            "image": self.pipeline_image.get(),
            "output": self.pipeline_output.get(),
            "parameters": parameters,
        }
        if not values["image"] or not values["output"]:
            messagebox.showwarning("資料未完整", "請選擇完整原圖及輸出資料夾。")
            return

        def job():
            report, artifacts = run_pipeline(
                values["image"],
                values["models"],
                parameters=values["parameters"],
                manager=self.manager,
            )
            run_dir = save_validation_result(
                values["output"], report, artifacts
            )
            return report, artifacts, run_dir

        self._run_async(
            self.pipeline_button,
            job,
            lambda result: self._show_validation(
                result,
                self.pipeline_result,
                self.pipeline_summary,
                self.pipeline_previews,
            ),
        )

    def _run_async(self, button, job, on_success):
        if self.active_job:
            messagebox.showinfo("正在執行", "請等待目前工作完成。")
            return
        button.config(state="disabled")
        self.active_job = (button, on_success)

        def worker():
            try:
                self.events.put(("done", job()))
            except Exception as error:
                self.events.put(("error", f"{type(error).__name__}: {error}"))

        threading.Thread(target=worker, daemon=True).start()

    def _poll_events(self):
        try:
            while True:
                event = self.events.get_nowait()
                if event[0] == "log":
                    _, message, target = event
                    target.insert(tk.END, message + "\n")
                    target.see(tk.END)
                elif event[0] == "done":
                    self._finish_job(event[1])
                elif event[0] == "error":
                    self._fail_job(event[1])
        except queue.Empty:
            pass
        self.root.after(80, self._poll_events)

    def _finish_job(self, result):
        button, on_success = self.active_job
        self.active_job = None
        button.config(state="normal")
        on_success(result)

    def _fail_job(self, message: str):
        button, _ = self.active_job
        self.active_job = None
        button.config(state="normal")
        messagebox.showerror("執行失敗", message)

    def _show_batch_result(self, report: dict):
        self.batch_log.insert(
            tk.END,
            "\n"
            + json.dumps(
                {
                    "counts": report["counts"],
                    "elapsed_ms": report["elapsed_ms"],
                    "report": str(Path(report["output_dir"]) / "batch_report.json"),
                },
                ensure_ascii=False,
                indent=2,
            ),
        )
        messagebox.showinfo("批次完成", f"成功輸出 {report['counts']['ok']} 張圖片。")

    def _show_validation(self, result, target, summary, preview_labels):
        report, artifacts, run_dir = result
        target.delete("1.0", tk.END)
        target.insert(
            tk.END,
            json.dumps(report, ensure_ascii=False, indent=2)
            + f"\n\n輸出：{run_dir}",
        )
        summary.config(text=f"模型摘要：{_validation_summary(report)}")
        self._display_artifacts(artifacts, preview_labels)

    def _display_artifacts(self, artifacts, labels):
        names = [
            name
            for name in (
                "original",
                "roi_annotated",
                "masked_roi",
                "wrinkle_gray",
                "defect_annotated",
            )
            if name in artifacts
        ]
        self.image_refs = []
        for index, label in enumerate(labels):
            for sequence in ("<Button-1>", "<Return>", "<space>"):
                label.unbind(sequence)
            if index >= len(names):
                label.config(
                    image="",
                    text="無此階段輸出",
                    cursor="",
                    takefocus=False,
                )
                label.image = None
                continue
            name = names[index]
            image = artifacts[name]
            rgb = (
                image
                if image.ndim == 2
                else cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            )
            full_image = Image.fromarray(rgb)
            preview = full_image.copy()
            preview.thumbnail((300, 260), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(preview)
            label.config(
                image=photo,
                text=name,
                compound="top",
                cursor="hand2",
                takefocus=True,
            )
            label.image = photo
            self.image_refs.append(photo)

            def open_image(_event=None, image=full_image, title=name):
                self._open_image_viewer(image, title)

            for sequence in ("<Button-1>", "<Return>", "<space>"):
                label.bind(sequence, open_image)

    def _open_image_viewer(self, image: Image.Image, title: str):
        window = tk.Toplevel(self.root)
        window.title(f"{title}－詳細圖片")
        window.transient(self.root)
        display_image = image.copy()
        display_image.thumbnail(
            (
                max(self.root.winfo_screenwidth() - 80, 1),
                max(self.root.winfo_screenheight() - 160, 1),
            ),
            Image.Resampling.LANCZOS,
        )
        photo = ImageTk.PhotoImage(display_image, master=window)
        ttk.Label(window, image=photo, padding=8).pack(fill="both", expand=True)
        window.image = photo
        window.bind("<Escape>", lambda _event: window.destroy())
        window.focus_set()

    def _choose_file(self, variable, file_types):
        path = filedialog.askopenfilename(filetypes=file_types)
        if path:
            variable.set(path)

    def _choose_directory(self, variable):
        path = filedialog.askdirectory()
        if path:
            variable.set(path)


if __name__ == "__main__":
    root = tk.Tk()
    PassionFruitApp(root)
    root.mainloop()
