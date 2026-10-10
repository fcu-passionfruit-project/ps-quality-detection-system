# line_notifier.py 使用说明

读取指定目录中的检测结果 `.json` 档案，整理成文字讯息后推送到 LINE 群组。

## 目录结构

```
ps-quality-detection-system/
└── passionfruit/
    └── notify/
        ├── line_notifier.py
        └── Line_test/
            └── result.json
```

## 运行前准备

需要 Python 3.8 以上，只用标准库，不需要安装任何套件。

程序从环境变量读取 `LINE_CHANNEL_ACCESS_TOKEN` 和 `LINE_GROUP_ID`，运行前要先设置好。以下两种方式任选一种。

### 方式一：直接设置环境变量

```bash
export LINE_CHANNEL_ACCESS_TOKEN="你的 Channel access token"
export LINE_GROUP_ID="目标群组 ID（以 C 开头）"
```

### 方式二：从 `.line.env` 载入

把设定写在 `notify/.line.env`：

```bash
export LINE_CHANNEL_ACCESS_TOKEN="你的 Channel access token"
export LINE_GROUP_ID="目标群组 ID（以 C 开头）"
```

运行前载入：

```bash
source .line.env
```

每行前面的 `export` 不能省略，否则 Python 读不到，程序会显示缺少环境变量。

注意：

- 程序本身不会读取 `.line.env`，一定要先 `source` 载入。
- 载入只对当前终端有效，开新的终端窗口要重新载入。
- `.line.env` 里有 token，不要提交到 git，记得加进 `.gitignore`。

以下指令都在 `notify` 目录下运行：

```bash
cd ps-quality-detection-system/passionfruit/notify
```

注意：指令里的目录（例如 `Line_test`）是相对于**当前所在目录**，不是相对于 `line_notifier.py` 的位置。如果在项目根目录运行，要写完整路径：

```bash
python3 passionfruit/notify/line_notifier.py send passionfruit/notify/Line_test
```

## send：处理一次就结束

扫描 `Line_test`（含所有子目录）中还没发送过的 json，全部推送到群组后结束程序。

```bash
python3 line_notifier.py send Line_test
```

适合手动运行，或在检测程序跑完之后调用一次。

已经发送过的档案会记录在 `Line_test/.line_sent.json`，下次运行不会重复推送。如果 `result.json` 被新的结果覆盖（内容改变），会再推送一次。

## watch：持续监看目录

程序会一直运行，每隔几秒检查一次 `Line_test`，发现新的或被覆盖的 json 就推送。按 `Ctrl+C` 结束。

```bash
python3 line_notifier.py watch Line_test
```

用 `--interval` 设置检查间隔（秒），默认 5 秒：

```bash
python3 line_notifier.py watch Line_test --interval 10
```

## --pattern：只处理指定档名

默认会处理目录里所有 `*.json`。如果只想处理 `result.json`，用 `--pattern` 指定，`send` 和 `watch` 都可以用：

```bash
python3 line_notifier.py send  Line_test --pattern result.json
python3 line_notifier.py watch Line_test --pattern result.json
```

也支持通配符，例如 `--pattern "result_*.json"`。使用通配符时请加上引号，避免被终端提前展开。
