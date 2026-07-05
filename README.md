# xBD 遥感灾损评估多模态 Agent

本项目面向 **xBD / xView2 灾害建筑物损毁评估** 场景，构建一个可以调用遥感损毁评估工具、统计建筑物损毁结果，并使用多模态大模型生成复核解释与报告的 Agent 原型。

当前推荐的工程分工是：

```text
xView2 / 外部遥感模型：负责建筑物区域定位和损毁等级 mask 预测
XBDDamageAnalyzer：负责变化检测、mask 后处理、统计、可视化和风险摘要
Qwen3.5 / Mock VLM：负责复核解释、报告生成和自然语言总结
```

也就是说，本项目不把 Qwen3.5 当成像素级分割模型使用。更可靠的路线是：**专用遥感模型输出 mask，多模态大模型解释和复核结果**。

---

## 1. 当前能力

- 支持灾前/灾后图像输入。
- 支持使用已有 `pre_mask` 和 `post_damage_mask` 直接生成评估报告。
- 支持调用官方 xView2 baseline，自动生成建筑物定位 mask 和损毁等级 mask。
- 支持无 mask 时使用图像差分做变化初筛，但该模式不能可靠判断损毁等级。
- 支持 Qwen3.5-4B / LoRA adapter / mock VLM 三种报告生成方式。
- 支持将 xBD JSON 标注转换为 mask，用于评估、演示和指令数据构造。
- 支持构造 LLaMA-Factory 多模态指令数据，为 LoRA 微调准备数据。
- 支持 FastAPI 和 Streamlit Demo。

当前项目可以称为：

```text
xBD 灾害建筑物损毁评估与多模态复核 Agent 原型
```

它已经具备“状态管理、工具调用、结果统计、多模态复核、报告输出”的 Agent 流程，但不是完全自主的通用智能体。

---

## 2. 项目流程

```text
灾前图像 + 灾后图像
        |
        v
AgentState 初始化
        |
        v
建筑物定位工具
  - provided_mask
  - xview2_baseline
  - external_command
        |
        v
损毁等级评估工具
  - provided_mask
  - xview2_baseline
  - external_command
  - diff_fallback
        |
        v
XBDDamageAnalyzer
  - 变化热力图
  - 建筑物像素统计
  - 损毁等级比例
  - 连通区域 / bbox
  - 风险等级
        |
        v
VLM 复核与解释
  - mock
  - qwen_multimodal
  - qwen_text
        |
        v
report.md + result.json + 可视化图像
```

输出目录通常包含：

```text
outputs/case_name/
├── change_heatmap.png
├── change_mask.png
├── damage_mask.png
├── damage_overlay.png
├── report.md
├── result.json
└── tool_outputs/
    ├── pre_building_mask.png
    └── post_damage_mask.png
```

---

## 3. 项目结构

当前代码采用 `src` 布局：

```text
xbd-rs-mllm-agent/
├── app/
│   ├── api.py                       # FastAPI 服务
│   └── streamlit_demo.py            # Streamlit 可视化 Demo
├── configs/
│   └── default.yaml                  # 默认配置
├── data/
│   └── README.md                     # 数据目录说明
├── scripts/
│   ├── build_xbd_masks.py            # xBD JSON 标注转 mask
│   ├── build_instruction_dataset.py  # 构造多模态指令数据
│   ├── create_toy_sample.py          # 生成玩具样例
│   ├── evaluate_damage.py            # mask 评估
│   ├── export_llamafactory_dataset.py
│   ├── mock_external_mask_tool.py
│   ├── run_analyze.py                # 主入口
│   └── run_xview2_baseline.py        # 官方 xView2 baseline 封装
├── src/
│   └── rs_xbd_agent/
│       ├── agent/                    # Agent 编排、状态、提示词
│       ├── data/                     # xBD 解析、指令数据构造
│       ├── eval/                     # IoU / F1 / OA 等指标
│       ├── models/                   # VLM 适配器与损毁分析器
│       ├── report/                   # Markdown 报告生成
│       ├── tools/                    # 工具接口、xView2/external command 后端
│       └── utils/                    # 图像工具
├── tests/
├── .env.example
├── .gitignore
├── pyproject.toml
├── requirements.txt
└── README.md
```

暂时不建议大规模移动源码目录，因为 `scripts/`、`app/` 和测试已经围绕当前包结构工作。后续如果继续工程化，可以把 `models/damage_analyzer.py` 拆到 `analysis/`，把 `models/qwen_*.py` 拆到 `vlm/`，但这不是当前运行所必需。

---

## 4. 安装主项目环境

建议 Python 3.10 或 3.11。

```bash
cd /root/autodl-tmp/xbd-rs-mllm-agent

conda create -n xbd-agent python=3.10 -y
conda activate xbd-agent

pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple
pip install -e .
```

如果需要调用 Qwen3.5-4B：

```bash
pip install torch transformers accelerate modelscope qwen-vl-utils peft \
  -i https://mirrors.aliyun.com/pypi/simple
```

---

## 5. 快速跑通玩具样例

生成模拟图像和 mask：

```bash
cd /root/autodl-tmp/xbd-rs-mllm-agent

python scripts/create_toy_sample.py --out_dir outputs/toy_sample
```

使用已有 mask 运行：

```bash
VLM_BACKEND=mock \
python scripts/run_analyze.py \
  --pre_image outputs/toy_sample/pre.png \
  --post_image outputs/toy_sample/post.png \
  --pre_mask outputs/toy_sample/pre_building_mask.png \
  --post_damage_mask outputs/toy_sample/post_damage_mask.png \
  --localization_backend provided_mask \
  --damage_backend provided_mask \
  --out_dir outputs/demo_report
```

这个命令只验证主流程，不依赖 xBD 数据集、xView2 权重或 Qwen 权重。

---

## 6. 使用 xBD 数据集

xBD 数据集通常包含：

```text
xBD/
├── images/
│   ├── xxx_pre_disaster.png
│   └── xxx_post_disaster.png
└── labels/
    ├── xxx_pre_disaster.json
    └── xxx_post_disaster.json
```

把 xBD JSON 标注转换为 mask：

```bash
cd /root/autodl-tmp/xbd-rs-mllm-agent

python scripts/build_xbd_masks.py \
  --labels_dir /root/autodl-tmp/xBD/labels \
  --out_dir outputs/xbd_masks
```

输出格式：

```text
outputs/xbd_masks/
├── localization/      # 0 背景，1 建筑物
├── damage/            # 0 背景，1 no-damage，2 minor，3 major，4 destroyed
└── index.jsonl
```

注意：由 xBD 标注生成的 mask 是 GT mask，只适合流程演示、评估对照和指令数据构造；不能把它当成自动预测结果。

---

## 7. 接入官方 xView2 Baseline

官方 xView2 baseline 可以从灾前/灾后图像自动输出损毁等级 PNG：

```text
0 = background / no building
1 = no-damage
2 = minor-damage
3 = major-damage
4 = destroyed
```

本项目的 `scripts/run_xview2_baseline.py` 已经把官方推理脚本封装成本项目可用的工具输出：

- `--mode damage`：保存 0-4 损毁等级 mask。
- `--mode localization`：把 damage mask 中 `>0` 的区域转换为建筑物定位 mask。

### 7.1 准备 xView2 独立环境

xView2 baseline 依赖较旧，建议单独环境运行，不要和 Qwen 环境混装。

```bash
cd /root/autodl-tmp

git clone https://github.com/DIUx-xView/xView2_baseline.git

conda create -n xview2 python=3.7 -y
conda activate xview2
```

按照官方仓库要求安装 TensorFlow / Keras / Chainer 等依赖。安装完成后，需要准备两个权重文件：

```text
localization.h5
classification.hdf5
```

建议放在：

```text
/root/autodl-tmp/xView2_baseline_weights/localization.h5
/root/autodl-tmp/xView2_baseline_weights/classification.hdf5
```

### 7.2 使用 xView2 + Mock VLM 跑完整 Agent

```bash
cd /root/autodl-tmp/xbd-rs-mllm-agent

VLM_BACKEND=mock \
LOCALIZATION_BACKEND=xview2_baseline \
DAMAGE_BACKEND=xview2_baseline \
XVIEW2_REPO=/root/autodl-tmp/xView2_baseline \
XVIEW2_LOCALIZATION_WEIGHTS=/root/autodl-tmp/xView2_baseline_weights/localization.h5 \
XVIEW2_CLASSIFICATION_WEIGHTS=/root/autodl-tmp/xView2_baseline_weights/classification.hdf5 \
XVIEW2_CONDA_ENV=xview2 \
python scripts/run_analyze.py \
  --pre_image /root/autodl-tmp/xBD/images/guatemala-volcano_00000003_pre_disaster.png \
  --post_image /root/autodl-tmp/xBD/images/guatemala-volcano_00000003_post_disaster.png \
  --out_dir outputs/xview2_agent_guatemala_00000003
```

这个命令会先调用 xView2 baseline 生成 mask，再由本项目统计并生成报告。

### 7.3 单独测试 xView2 封装脚本

```bash
cd /root/autodl-tmp/xbd-rs-mllm-agent

conda run -n xview2 python scripts/run_xview2_baseline.py \
  --repo /root/autodl-tmp/xView2_baseline \
  --pre /root/autodl-tmp/xBD/images/guatemala-volcano_00000003_pre_disaster.png \
  --post /root/autodl-tmp/xBD/images/guatemala-volcano_00000003_post_disaster.png \
  --localization-weights /root/autodl-tmp/xView2_baseline_weights/localization.h5 \
  --classification-weights /root/autodl-tmp/xView2_baseline_weights/classification.hdf5 \
  --output outputs/xview2_damage_mask.png \
  --mode damage
```

---

## 8. 使用 Qwen3.5-4B 生成复核报告

下载模型权重示例：

```bash
modelscope download \
  --model Qwen/Qwen3.5-4B \
  --local_dir /root/autodl-tmp/models/Qwen3.5-4B
```

使用基础 Qwen3.5-4B：

```bash
cd /root/autodl-tmp/xbd-rs-mllm-agent

VLM_BACKEND=qwen_multimodal \
MLLM_MODEL_NAME=/root/autodl-tmp/models/Qwen3.5-4B \
MLLM_DEVICE=auto \
LOCALIZATION_BACKEND=xview2_baseline \
DAMAGE_BACKEND=xview2_baseline \
XVIEW2_REPO=/root/autodl-tmp/xView2_baseline \
XVIEW2_LOCALIZATION_WEIGHTS=/root/autodl-tmp/xView2_baseline_weights/localization.h5 \
XVIEW2_CLASSIFICATION_WEIGHTS=/root/autodl-tmp/xView2_baseline_weights/classification.hdf5 \
XVIEW2_CONDA_ENV=xview2 \
python scripts/run_analyze.py \
  --pre_image /root/autodl-tmp/xBD/images/guatemala-volcano_00000003_pre_disaster.png \
  --post_image /root/autodl-tmp/xBD/images/guatemala-volcano_00000003_post_disaster.png \
  --out_dir outputs/xview2_qwen35_guatemala_00000003
```

使用已经微调好的 LoRA adapter：

```bash
VLM_BACKEND=qwen_multimodal \
MLLM_MODEL_NAME=/root/autodl-tmp/models/Qwen3.5-4B \
MLLM_ADAPTER_PATH=/root/autodl-tmp/xbd-rs-mllm-agent/outputs/lora_qwen35_xbd_review \
MLLM_DEVICE=auto \
LOCALIZATION_BACKEND=xview2_baseline \
DAMAGE_BACKEND=xview2_baseline \
XVIEW2_REPO=/root/autodl-tmp/xView2_baseline \
XVIEW2_LOCALIZATION_WEIGHTS=/root/autodl-tmp/xView2_baseline_weights/localization.h5 \
XVIEW2_CLASSIFICATION_WEIGHTS=/root/autodl-tmp/xView2_baseline_weights/classification.hdf5 \
XVIEW2_CONDA_ENV=xview2 \
python scripts/run_analyze.py \
  --pre_image /root/autodl-tmp/xBD/images/guatemala-volcano_00000003_pre_disaster.png \
  --post_image /root/autodl-tmp/xBD/images/guatemala-volcano_00000003_post_disaster.png \
  --out_dir outputs/xview2_qwen35_lora_guatemala_00000003
```

是否使用 LoRA 只看一个变量：

```text
没有 MLLM_ADAPTER_PATH：使用原始 Qwen3.5-4B
设置 MLLM_ADAPTER_PATH：使用 Qwen3.5-4B + LoRA adapter
```


---

## 9. GeoChat 对照实验

GeoChat 是面向遥感场景的视觉语言模型，可作为 Qwen3.5 的遥感 VLM 对照模型。当前项目尚未把 GeoChat 写成内置 `VLM_BACKEND=geochat` 适配器，推荐先使用 GeoChat 官方 Demo 对本项目生成的灾前图、灾后图、`damage_overlay.png` 做独立复核。

### 9.1 安装 GeoChat 独立环境

GeoChat 的依赖可能与 Qwen / xView2 环境冲突，建议单独创建环境：

```bash
cd /root/autodl-tmp

git clone https://github.com/mbzuai-oryx/GeoChat.git
cd GeoChat

conda create -n geochat python=3.10 -y
conda activate geochat

pip install --upgrade pip
pip install -e .
```

如果官方 Demo 需要加速算子，可按 GeoChat 仓库说明安装 `flash-attn`；安装失败时可以先跳过，只做基础推理验证。

### 9.2 下载 GeoChat 权重

```bash
pip install -U huggingface_hub

huggingface-cli download MBZUAI/geochat-7B \
  --local-dir /root/autodl-tmp/models/geochat-7B
```

如果 Hugging Face 网络不稳定，可以手动下载到同一路径。

### 9.3 使用方式

先用本项目生成报告和叠加图：

```bash
cd /root/autodl-tmp/xbd-rs-mllm-agent

VLM_BACKEND=mock \
LOCALIZATION_BACKEND=xview2_baseline \
DAMAGE_BACKEND=xview2_baseline \
XVIEW2_REPO=/root/autodl-tmp/xView2_baseline \
XVIEW2_LOCALIZATION_WEIGHTS=/root/autodl-tmp/xView2_baseline_weights/localization.h5 \
XVIEW2_CLASSIFICATION_WEIGHTS=/root/autodl-tmp/xView2_baseline_weights/classification.hdf5 \
XVIEW2_CONDA_ENV=xview2 \
python scripts/run_analyze.py \
  --pre_image /root/autodl-tmp/xBD/images/guatemala-volcano_00000003_pre_disaster.png \
  --post_image /root/autodl-tmp/xBD/images/guatemala-volcano_00000003_post_disaster.png \
  --out_dir outputs/geochat_compare_guatemala_00000003
```

然后在 GeoChat 官方 Demo 中上传：

```text
pre_disaster.png
post_disaster.png
damage_overlay.png
```

可提问：

```text
Please compare the pre-disaster and post-disaster remote sensing images, identify visible building damage, and check whether the damage overlay is visually reasonable.
```

建议把 GeoChat 的回答与本项目 `report.md` 对比，重点看：遥感视觉描述是否更准确、是否能发现 mask 可疑区域、是否出现幻觉。

## 10. `run_analyze.py` 参数说明

```bash
python scripts/run_analyze.py \
  --pre_image PRE_IMAGE \
  --post_image POST_IMAGE \
  --pre_mask PRE_MASK \
  --post_damage_mask POST_DAMAGE_MASK \
  --out_dir OUT_DIR \
  --config configs/default.yaml \
  --localization_backend provided_mask \
  --damage_backend provided_mask \
  --localization_command "..." \
  --damage_command "..." \
  --external_tool_timeout 1800
```

参数含义：

```text
--pre_image
  灾前图像路径，必填。

--post_image
  灾后图像路径，必填。

--pre_mask
  灾前建筑物定位 mask，可选。0 表示背景，1 表示建筑物。

--post_damage_mask
  灾后损毁等级 mask，可选。0 背景，1 未损毁，2 轻微损毁，3 严重损毁，4 完全损毁。

--out_dir
  输出目录。

--config
  YAML 配置文件路径，默认 configs/default.yaml。

--localization_backend
  建筑物定位工具后端。

--damage_backend
  损毁等级评估工具后端。

--localization_command / --damage_command
  外部命令模板，用于接入自己的模型推理脚本。

--external_tool_timeout
  外部工具超时时间，单位秒。
```

支持的工具后端：

```text
localization_backend:
  provided_mask / gt_mask / external_mask / external_command / xview2_baseline

damage_backend:
  provided_mask / gt_mask / external_mask / diff_fallback / external_command / xview2_baseline / xview2_strong_baseline
```

---

## 11. 配置文件和环境变量

可以通过 `.env` 或命令行环境变量配置。推荐从示例文件复制：

```bash
cp .env.example .env
```

常用变量：

```bash
VLM_BACKEND=mock
MLLM_MODEL_NAME=/root/autodl-tmp/models/Qwen3.5-4B
MLLM_ADAPTER_PATH=
MLLM_DEVICE=auto

LOCALIZATION_BACKEND=xview2_baseline
DAMAGE_BACKEND=xview2_baseline
XVIEW2_REPO=/root/autodl-tmp/xView2_baseline
XVIEW2_LOCALIZATION_WEIGHTS=/root/autodl-tmp/xView2_baseline_weights/localization.h5
XVIEW2_CLASSIFICATION_WEIGHTS=/root/autodl-tmp/xView2_baseline_weights/classification.hdf5
XVIEW2_CONDA_ENV=xview2
```

旧变量 `MLLM_BACKEND` 仍兼容，但推荐使用 `VLM_BACKEND`。

---

## 12. 接入自己的外部模型

如果你有自己的建筑物定位模型或损毁评估模型，只需要写一个推理脚本，把结果保存为 PNG mask。

命令模板支持以下占位符：

```text
{pre_image}
{post_image}
{pre_mask}
{post_damage_mask}
{out_dir}
{output_mask}
```

示例：

```bash
VLM_BACKEND=mock \
python scripts/run_analyze.py \
  --pre_image /path/to/pre.png \
  --post_image /path/to/post.png \
  --localization_backend external_command \
  --damage_backend external_command \
  --localization_command "python /path/to/infer_building.py --pre {pre_image} --post {post_image} --output {output_mask}" \
  --damage_command "python /path/to/infer_damage.py --pre {pre_image} --post {post_image} --pre-mask {pre_mask} --output {output_mask}" \
  --out_dir outputs/external_model_case
```

输出约定：

```text
建筑物定位 mask:
  0 = background
  1 = building

损毁等级 mask:
  0 = background
  1 = no-damage
  2 = minor-damage
  3 = major-damage
  4 = destroyed
```

---

## 13. 构造 LoRA 指令数据

先从 xBD 标注构造 mask：

```bash
python scripts/build_xbd_masks.py \
  --labels_dir /root/autodl-tmp/xBD/labels \
  --out_dir outputs/xbd_masks
```

构造解释与复核风格数据：

```bash
python scripts/build_instruction_dataset.py \
  --image_dir /root/autodl-tmp/xBD/images \
  --mask_index outputs/xbd_masks/index.jsonl \
  --out_file outputs/xbd_instruction_review.jsonl \
  --style review
```

导出 LLaMA-Factory 多模态格式：

```bash
python scripts/export_llamafactory_dataset.py \
  --input outputs/xbd_instruction_review.jsonl \
  --output outputs/llamafactory_xbd_review.json
```

当前 LoRA 数据的性质是：

```text
输入：文本指令 + 灾前图像 + 灾后图像
标签：基于 mask 统计生成的解释与复核文本
目标：让模型更稳定地输出报告结构，而不是让模型学会像素级损毁分割
```

因此，LoRA 版本适合作为报告格式对齐 baseline；如果要增强真正的视觉理解，应继续加入建筑物 crop、damage overlay crop 和人工审核过的视觉证据描述。

---

## 14. FastAPI 与 Streamlit

启动 FastAPI：

```bash
uvicorn app.api:app --host 0.0.0.0 --port 8000 --reload
```

调用接口：

```bash
curl -X POST "http://127.0.0.1:8000/analyze" \
  -F "pre_image=@outputs/toy_sample/pre.png" \
  -F "post_image=@outputs/toy_sample/post.png" \
  -F "pre_mask=@outputs/toy_sample/pre_building_mask.png" \
  -F "post_damage_mask=@outputs/toy_sample/post_damage_mask.png"
```

启动 Streamlit：

```bash
streamlit run app/streamlit_demo.py
```

---

## 15. 评估

像素级评估用于评估建筑物定位或损毁等级 mask，而不是直接评价大模型报告。

```bash
python scripts/evaluate_damage.py \
  --pred_mask /path/to/pred_damage.png \
  --gt_mask /path/to/gt_damage.png
```

可关注：

```text
IoU：各损毁等级交并比
F1：像素级分类 F1
OA：Overall Accuracy
严重损毁比例：major + destroyed
```

VLM 报告可以从以下角度人工或半自动评估：

```text
统计一致性：报告是否与 result.json 中数值一致
视觉一致性：报告是否与原图、overlay、bbox 相符
幻觉率：是否描述不存在的建筑物或损毁
复核价值：是否指出需要人工检查的区域和原因
```

---

## 16. 上传 GitHub 前的清理建议

不要上传数据集、模型权重、运行输出和私密配置。

应该保留：

```text
src/
scripts/
app/
configs/
tests/
data/README.md
README.md
requirements.txt
pyproject.toml
.env.example
.gitignore
```

不应该上传：

```text
.env
outputs/
/root/autodl-tmp/xBD
/root/autodl-tmp/models/Qwen3.5-4B
/root/autodl-tmp/xView2_baseline
/root/autodl-tmp/xView2_baseline_weights
src/*.egg-info/
__pycache__/
```

检查命令：

```bash
git status
git diff --cached --name-only
```

首次提交示例：

```bash
git add .
git commit -m "Refine xBD disaster assessment agent"
git remote add origin https://github.com/你的用户名/xbd-rs-mllm-agent.git
git branch -M main
git push -u origin main
```

---

## 17. 当前能力边界

- xView2 baseline 可以输出可用的建筑物和损毁等级 mask，但它是传统 baseline，预测质量会随灾种、地区、影像质量变化。
- Qwen3.5 只负责解释和复核，不负责替代像素级损毁分割。
- 如果没有 `pre_mask` 和 `post_damage_mask`，系统只能做粗略图像差分，不能可靠判断建筑物损毁等级。
- 当前 `LOCALIZATION_BACKEND=xview2_baseline` 和 `DAMAGE_BACKEND=xview2_baseline` 会分别调用工具，存在重复推理空间；后续可以合并成一次 xView2 推理并缓存两个 mask。

---

## 18. 后续优化方向

- 新增批处理脚本，对 xBD 测试集随机抽样或全量生成报告。
- 将 xView2 baseline 调用合并为一次推理，减少重复计算。
- 为高风险建筑自动裁剪灾前/灾后 crop 和 overlay crop，交给 VLM 做区域级复核。
- 增加低置信度样本筛选和人工审核清单。
- 将 `models/damage_analyzer.py` 拆分到 `analysis/`，让模块边界更清晰。
- 接入更强的 xBD 损毁分割模型，替换 xView2 baseline。
- 继续构造区域级、多图、多证据的 LoRA 数据，提升 VLM 对原始图像证据的解释能力。
