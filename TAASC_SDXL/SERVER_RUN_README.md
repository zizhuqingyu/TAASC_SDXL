# TAASC_SDXL 服务器运行说明

代码目录：`/home/zjs/PycharmProjects/PythonProject/TAASC_SDXL`

环境：

- `conda activate sdxl`：Python 3.11、PyTorch 2.8.0+cu128、Diffusers 0.35.1、Transformers 4.55.4。
- `conda activate internvl`：原有环境，运行 `internvl_server.py`。

## 运行整个输入文件

```bash
cd /home/zjs/PycharmProjects/PythonProject/TAASC_SDXL
/home/zjs/anaconda3/envs/sdxl/bin/python run_cc500.py --detach
```

启动脚本负责启动 InternVL、等待健康检查、运行 SDXL，实验结束后关闭本脚本启动的 InternVL 服务。进程与 SSH 会话分离，退出 SSH 后继续执行。发现仍在运行的实验时拒绝重复启动。

输入文件虽名为 CC-500，实际包含 602 条有效记录，全部执行。保持 `inference_sdxl.py` 当前方法与默认值：1024×1024、50 步、CFG 7、初始 seed 42（逐样本加 1）、最多 10 轮、阈值 0.8、保守 lambda 默认值、best 选择策略。额外启用 `--save_attempts` 保存每轮图像。

完整输出目录：

`/media/zjs/ff61b503-6b67-47d0-bf89-a669c875d2f2/experiment_outputs/paper2/TAASC_SDXL/CC-500`

## 查看进度

```bash
OUT=/media/zjs/ff61b503-6b67-47d0-bf89-a669c875d2f2/experiment_outputs/paper2/TAASC_SDXL/CC-500
cat "$OUT/service/experiment_state.json"
tail -f "$OUT/service/inference.log"
```

随时生成当前统计（不影响推理）：

```bash
/home/zjs/anaconda3/envs/sdxl/bin/python run_cc500.py --summary
```

`results.csv` 与 `summary.json` 在整个进程结束时自动生成，也可通过上面的命令手动更新。`complete=true` 表示已完成该样本的生成与评分流程；`max_rounds_reached` 表示执行完 10 轮仍未全部达到阈值，不能视为语义全部通过。

## 输出结构

- `<id>.png`：按现有代码策略选出的最终图片。
- `baseline/<id>.png`：初始 lambda=0 的原生基线。
- `attempts/<id>/round_XX.png`：每轮图片。
- `logs/<id>.json`：参数、逐轮评分、lambda、注意力事件、显存与选择结果。`round` 和 `selected_round` 使用 0 起始编号。
- `service/internvl_responses.jsonl`：评分提示词、真实模型输出、校验后的评分、图像 SHA256、耗时。
- `service/source_*/`：启动时的源代码、两个环境的依赖快照。
- `service/experiment_state.json`：PID、启动/结束时间、退出码、最终统计。
- `service/internvl_server.log`、`inference.log`、`supervisor.log`：服务、推理与管理进程日志。

## 单独启动评分服务

```bash
cd /home/zjs/PycharmProjects/PythonProject/TAASC_SDXL
/home/zjs/anaconda3/envs/internvl/bin/python internvl_server.py
/home/zjs/anaconda3/envs/internvl/bin/python -c 'import requests; print(requests.get("http://127.0.0.1:8000/health").json())'
```

文件名 `internvl_server.py` 按原请求保留。现有 `internvl_client.py` 无须修改，向 `http://127.0.0.1:8000/chat` POST `{"image_path": ..., "annotation": ...}`，返回与标注顺序、长度对应的 `entities`、`attributes`、`relations` 浮点评分数组。

服务仅监听回环地址，默认仅接受此 TAASC_SDXL 输出根目录内的图像。模型加载一次，评分时移入 GPU，结束后回到 CPU。确定性解码、最多 512 新 token；无效响应内部重试一次，仍无效则返回错误，不补造分数。评分是新配置的 VLM 评价协议，尚不能等同于任何未提供的历史评分服务或独立人工评测。

断点恢复可重跑同一启动命令；原代码仅跳过签名匹配且图片和日志完整的样本。不要同时运行两份实验。停止整个任务可先向 `experiment_state.json` 中的 `inference_pid` 发送 SIGTERM，管理进程随后写出统计并关闭它启动的评分服务。
