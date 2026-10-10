#!/usr/bin/env python3

from __future__ import annotations

import argparse
import base64
import hashlib
import hmac
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
import uuid
from datetime import datetime
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path, PureWindowsPath
from typing import Iterable

PUSH_URL = "https://api.line.me/v2/bot/message/push"
LEDGER_NAME = ".line_sent.json"
MAX_TEXT_LEN = 5000
MAX_MESSAGES_PER_REQUEST = 5
FILE_SETTLE_SECONDS = 2.0

CLASS_NAMES_ZH = {
    "abrasion": "擦傷",
    "anthracnose": "炭疽病",
    "insect_bite": "蟲咬",
    "insect_track": "蟲跡",
}


def load_result(path: str | os.PathLike) -> dict:
    with open(path, "r", encoding="utf-8-sig") as f:
        return json.load(f)


def _basename_any_os(p: str | None) -> str:
    if not p:
        return "(未知)"
    return PureWindowsPath(p).name or str(p)


def _guess_time(result: dict, json_path: Path | None) -> str:
    for p in (result.get("artifacts") or {}).values():
        if not p:
            continue
        for part in PureWindowsPath(p).parts:
            m = re.fullmatch(r"(\d{8})_(\d{6})(?:_\d+)?", part)
            if m:
                try:
                    dt = datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S")
                    return dt.strftime("%Y-%m-%d %H:%M:%S")
                except ValueError:
                    pass
    if json_path and json_path.exists():
        return datetime.fromtimestamp(json_path.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def has_alert(result: dict) -> bool:
    model = result.get("model") or {}
    if not result.get("ok", False) or model.get("status") not in (None, "ok"):
        return True
    return bool(model.get("detections"))


def format_message(result: dict, json_path: str | os.PathLike | None = None) -> str:
    json_path = Path(json_path) if json_path else None
    model = result.get("model") or {}
    detections = model.get("detections") or []

    lines = ["🔍 百香果瑕疵檢測結果"]
    lines.append(f"圖片：{_basename_any_os(result.get('source'))}")
    lines.append(f"時間：{_guess_time(result, json_path)}")

    ok = result.get("ok", False) and model.get("status", "ok") == "ok"
    if not ok:
        reason = result.get("reason") or model.get("reason") or "未提供原因"
        lines.append(f"結果：❌ 檢測失敗（{reason}）")
    elif not detections:
        lines.append("結果：✅ 未發現瑕疵")
    else:
        lines.append(f"結果：⚠️ 發現瑕疵（{len(detections)} 處）")
        for d in sorted(detections, key=lambda x: x.get("confidence", 0), reverse=True):
            name = d.get("class_name", "?")
            zh = CLASS_NAMES_ZH.get(name, name)
            conf = d.get("confidence")
            area = d.get("mask_area_ratio")
            conf_s = f"{conf * 100:.1f}%" if isinstance(conf, (int, float)) else "-"
            area_s = f"{area * 100:.2f}%" if isinstance(area, (int, float)) else "-"
            lines.append(f"・{zh}（{name}）信心 {conf_s}，面積 {area_s}")

    meta = []
    if result.get("role"):
        meta.append(str(result["role"]))
    if result.get("device"):
        meta.append(str(result["device"]))
    if isinstance(result.get("elapsed_ms"), (int, float)):
        meta.append(f"{result['elapsed_ms']:.0f} ms")
    if meta:
        lines.append("模型：" + " / ".join(meta))

    text = "\n".join(lines)
    return text if len(text) <= MAX_TEXT_LEN else text[: MAX_TEXT_LEN - 1] + "…"


class LineApiError(RuntimeError):
    pass


def push_to_line(texts: list[str], token: str, to: str, retries: int = 3) -> None:
    if not 1 <= len(texts) <= MAX_MESSAGES_PER_REQUEST:
        raise ValueError("texts 必须是 1~5 则")
    body = json.dumps(
        {"to": to, "messages": [{"type": "text", "text": t} for t in texts]},
        ensure_ascii=False,
    ).encode("utf-8")
    retry_key = str(uuid.uuid4())

    for attempt in range(1, retries + 1):
        req = urllib.request.Request(
            PUSH_URL,
            data=body,
            method="POST",
            headers={
                "Content-Type": "application/json; charset=utf-8",
                "Authorization": f"Bearer {token}",
                "X-Line-Retry-Key": retry_key,
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                if resp.status == 200:
                    return
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", "replace")
            if e.code == 409:
                return
            if e.code == 429:
                raise LineApiError(f"429 超过速率或本月讯息额度已用完：{detail}") from None
            if 400 <= e.code < 500:
                raise LineApiError(f"HTTP {e.code}：{detail}") from None
            err = f"HTTP {e.code}：{detail}"
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            err = f"网络错误：{e}"
        if attempt < retries:
            time.sleep(2 ** attempt)
    raise LineApiError(f"重试 {retries} 次仍失败，最后错误：{err}")


def _get_credentials(token: str | None, to: str | None) -> tuple[str, str]:
    token = token or os.environ.get("LINE_CHANNEL_ACCESS_TOKEN")
    to = to or os.environ.get("LINE_GROUP_ID")
    if not token or not to:
        raise SystemExit(
            "缺少 LINE_CHANNEL_ACCESS_TOKEN 或 LINE_GROUP_ID 环境变量（或用 --token / --to 指定）"
        )
    return token, to


def send_result_file(path: str | os.PathLike, token: str | None = None,
                     to: str | None = None, dry_run: bool = False) -> str:
    text = format_message(load_result(path), path)
    if dry_run:
        print(text, end="\n" + "-" * 40 + "\n")
    else:
        push_to_line([text], *_get_credentials(token, to))
    return text


def _load_ledger(directory: Path) -> set[str]:
    try:
        return set(json.loads((directory / LEDGER_NAME).read_text(encoding="utf-8")))
    except (FileNotFoundError, ValueError):
        return set()


def _save_ledger(directory: Path, sent: Iterable[str]) -> None:
    tmp = directory / (LEDGER_NAME + ".tmp")
    tmp.write_text(json.dumps(sorted(sent), ensure_ascii=False, indent=0), encoding="utf-8")
    os.replace(tmp, directory / LEDGER_NAME)


def _chunks(items: list, n: int):
    for i in range(0, len(items), n):
        yield items[i:i + n]


def process_directory(directory: str | os.PathLike, token: str | None = None,
                      to: str | None = None, dry_run: bool = False,
                      only_defects: bool = False, per_request: int = 5,
                      recursive: bool = True, pattern: str = "*.json") -> int:
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(f"找不到目录：{directory}")
    per_request = max(1, min(MAX_MESSAGES_PER_REQUEST, per_request))
    if not dry_run:
        token, to = _get_credentials(token, to)

    sent = _load_ledger(directory)
    files = directory.rglob(pattern) if recursive else directory.glob(pattern)
    now = time.time()
    pending: list[tuple[str, str, str]] = []

    for p in sorted(files, key=lambda x: x.stat().st_mtime):
        if p.name.startswith(".line_sent") or not p.is_file():
            continue
        rel = p.relative_to(directory).as_posix()
        if now - p.stat().st_mtime < FILE_SETTLE_SECONDS:
            continue
        raw = p.read_bytes()
        key = f"{rel}#{hashlib.sha1(raw).hexdigest()[:12]}"
        if key in sent:
            continue
        try:
            result = json.loads(raw.decode("utf-8-sig"))
        except (ValueError, UnicodeDecodeError) as e:
            print(f"[略过] {rel} 不是有效 JSON（{e}），下一轮再试", file=sys.stderr)
            continue
        if not isinstance(result, dict):
            print(f"[略过] {rel} 格式不符（最外层不是物件）", file=sys.stderr)
            sent.add(key)
            continue
        if only_defects and not has_alert(result):
            sent.add(key)
            continue
        pending.append((key, rel, format_message(result, p)))

    count = 0
    for batch in _chunks(pending, per_request):
        texts = [t for _, _, t in batch]
        if dry_run:
            for t in texts:
                print(t, end="\n" + "-" * 40 + "\n")
        else:
            try:
                push_to_line(texts, token, to)
            except LineApiError as e:
                print(f"[发送失败] {e}", file=sys.stderr)
                break
            sent.update(k for k, _, _ in batch)
            _save_ledger(directory, sent)
        count += len(batch)
        print(f"[{'试跑' if dry_run else '已发送'}] {', '.join(r for _, r, _ in batch)}")

    if not dry_run:
        _save_ledger(directory, sent)
    return count


def watch_directory(directory: str | os.PathLike, interval: float = 5.0, **kwargs) -> None:
    print(f"开始监看 {directory}，每 {interval} 秒检查一次（Ctrl+C 结束）")
    try:
        while True:
            try:
                process_directory(directory, **kwargs)
            except FileNotFoundError as e:
                print(e, file=sys.stderr)
            time.sleep(interval)
    except KeyboardInterrupt:
        print("\n已停止监看")


def run_group_id_listener(port: int = 8000) -> None:
    secret = os.environ.get("LINE_CHANNEL_SECRET")

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            raw = self.rfile.read(int(self.headers.get("Content-Length", 0)))
            if secret:
                expected = base64.b64encode(
                    hmac.new(secret.encode(), raw, hashlib.sha256).digest()).decode()
                if not hmac.compare_digest(expected, self.headers.get("X-Line-Signature", "")):
                    self.send_response(403); self.end_headers(); return
            try:
                events = json.loads(raw).get("events", [])
            except ValueError:
                events = []
            for ev in events:
                src = ev.get("source", {})
                if src.get("type") == "group":
                    print(f"★ 群组 ID：{src.get('groupId')}（事件：{ev.get('type')}）")
                elif src.get("type"):
                    print(f"收到 {src.get('type')} 事件，不是群组；请在群组里发言或邀请 bot 入群")
            self.send_response(200); self.end_headers()

        def log_message(self, *args):
            pass

    print(f"Webhook 监听中：http://0.0.0.0:{port}/（取得 ID 后 Ctrl+C 结束，别长期开着）")
    if not secret:
        print("提示：未设 LINE_CHANNEL_SECRET，不验证签名（仅限短暂测试使用）")
    try:
        HTTPServer(("0.0.0.0", port), Handler).serve_forever()
    except KeyboardInterrupt:
        pass


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    ap = argparse.ArgumentParser(description="把百香果检测结果 json 推送到 LINE 群组")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("directory", help="存放 .json 结果的目录")
        p.add_argument("--dry-run", action="store_true", help="只印出讯息，不发送")
        p.add_argument("--only-defects", action="store_true", help="只推送有瑕疵或失败的结果")
        p.add_argument("--per-request", type=int, default=5, help="每次请求合并几笔（1~5）")
        p.add_argument("--no-recursive", dest="recursive", action="store_false",
                       help="只找这一层，不进子目录（预设会进所有子目录）")
        p.add_argument("--pattern", default="*.json",
                       help="要处理的档名，例如 result.json（预设 *.json）")
        p.add_argument("--token", help="覆盖 LINE_CHANNEL_ACCESS_TOKEN")
        p.add_argument("--to", help="覆盖 LINE_GROUP_ID")

    common(sub.add_parser("send", help="处理一次目录后结束"))
    w = sub.add_parser("watch", help="持续监看目录")
    common(w)
    w.add_argument("--interval", type=float, default=5.0, help="检查间隔秒数")
    g = sub.add_parser("get-group-id", help="启动临时 webhook 取得群组 ID")
    g.add_argument("--port", type=int, default=8000)

    a = ap.parse_args(argv)
    if a.cmd == "get-group-id":
        run_group_id_listener(a.port)
        return 0

    kw = dict(token=a.token, to=a.to, dry_run=a.dry_run, only_defects=a.only_defects,
              per_request=a.per_request, recursive=a.recursive, pattern=a.pattern)
    if a.cmd == "send":
        n = process_directory(a.directory, **kw)
        print(f"完成，本次处理 {n} 个档案")
    else:
        watch_directory(a.directory, interval=a.interval, **kw)
    return 0


if __name__ == "__main__":
    sys.exit(main())
