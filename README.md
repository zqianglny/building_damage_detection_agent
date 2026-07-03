# 面向 xBD 的建筑物损毁评估与多模态复核 Agent 原型

本项目面向 **xBD / xView2 灾害建筑物损毁评估** 场景，构建一个“遥感损毁预测结果 + 多模态大模型复核解释 + 交互式报告生成”的 Demo 系统。

本项目不把 Qwen3.5-4B / GeoChat 定位为像素级损毁分割模型。更准确的分工是：

```text
遥感预测模块：负责建筑物定位、损毁等级 mask、变化区域、bbox 和统计结果
多模态大模型：负责视觉证据解释、低置信度复核、区域问答、报告生成
```

核心原则：**专用遥感模型做可靠预测，多模态大模型做解释、复核和交互式解译。**

---

## 1. 项目定位

当前项目支持三种实践模式：

```text
1. GT mask 演示模式
   从 xBD JSON 标注生成建筑物 mask 和损毁等级 mask，用于流程演示、数据构造和评估对照。
   注意：使用 GT mask 不代表系统完成了自动损毁预测。

2. 外部模型预测模式
   读取用户已有建筑物损毁评估网络输出的 pre_mask / post_damage_mask。
   这是更接近真实自动化系统的正式输入方式。

3. 无 mask 初筛模式
   仅用灾前/灾后图像差分生成粗略变化热力图。
   该模式只能用于变化初筛，不能输出可靠的建筑物损毁等级。
```

多模态大模型的作用不是直接输出 damage mask，而是：

```text
1. 读取灾前图、灾后图、mask overlay、局部 crop 和结构化统计。
2. 对低置信度或高风险建筑进行视觉复核。
3. 解释 no-damage / minor / major / destroyed 判定依据。
4. 回答用户对指定区域或指定建筑的自然语言问题。
5. 生成包含复核建议的灾害评估报告。
```

### 当前 Agent 化程度说明

当前工程更准确地说是 **多模态复核 Agent 原型**，而不是完全自主的灾害评估智能体。它已经具备“输入感知 -> 工具选择 -> 工具分析 -> 多模态复核 -> 报告输出”的 Agent 雏形：

```text
灾前/灾后图像输入
  -> Tool Router 选择建筑物定位工具和损毁评估工具
  -> XBDDamageAnalyzer 统计变化、mask、bbox 和 overlay
  -> Qwen3.5-4B / Mock VLM 生成复核文本
  -> ReportWriter 输出 Markdown 报告
```

目前已经加入初版工具路由：`LOCALIZATION_BACKEND` 负责建筑物定位，`DAMAGE_BACKEND` 负责损毁等级评估。当前可直接运行的是 `provided_mask` / `gt_mask` / `external_mask` / `diff_fallback`；`xview2_baseline` 和 `xview2_strong_baseline` 已作为推荐外部模型工具位保留，后续需要把对应仓库的推理脚本封装进 `src/rs_xbd_agent/tools/`。

因此，简历或论文描述中建议使用：

```text
xBD 灾害损毁评估与多模态复核 Agent 原型
```

而不是夸大为“完全自主灾害评估智能体”。后续如果补充外部模型自动推理、Region-level QA、低置信度样本二次复核和多轮问答，就可以进一步升级为更完整的多模态 Agent。

---

## 2. 核心流程

```text
pre-disaster image
post-disaster image
        ↓
[Predictor Layer]
GT mask / external damage model / mock diff predictor
        ↓
building mask
damage mask
confidence map, optional
        ↓
[Analysis Layer]
instance extraction
bbox generation
damage statistics
uncertainty filtering
overlay visualization
        ↓
[VLM Review Layer]
pre/post image or crop
damage overlay
structured JSON
user question
        ↓
damage explanation
visual consistency check
review suggestions
interactive QA
        ↓
[Report Layer]
Markdown report
JSON result
visual outputs
```

当前代码中的 `XBDDamageAnalyzer` 主要承担 mask 统计、图像差分、bbox 提取和可视化；后续可以把 predictor 层替换为你自己的变化检测或损毁分割模型。

---

## 3. 项目特点

- 支持 xBD 原始 JSON 标注解析，生成建筑物定位 mask 与损毁等级 mask。
- 支持读取外部模型预测的建筑物 mask / damage mask 并生成统计与报告。
- 支持无 mask 时的灾前/灾后粗略变化初筛。
- 支持将灾前图、灾后图、变化热力图、损毁等级叠加图和结构化统计输入 Qwen3.5-4B。
- 支持 GeoChat 作为遥感 VLM 对照模型或后续可选后端。
- 支持构造面向“损毁解释与复核”的多模态指令数据。
- 支持 FastAPI 服务与 Streamlit 可视化 Demo。
- 支持 IoU、F1、OA 等指标评估损毁预测 mask。

---

## 4. 当前项目结构

```text
xbd-rs-mllm-agent/
├── app/
│   ├── api.py                         # FastAPI 服务
│   └── streamlit_demo.py              # Streamlit 可视化页面
├── configs/
│   └── default.yaml                    # 默认配置
├── data/
│   └── README.md                       # 数据目录说明
├── scripts/
│   ├── build_xbd_masks.py             # xBD JSON 转 mask
│   ├── build_instruction_dataset.py   # 构造解释/复核指令数据
│   ├── create_toy_sample.py           # 生成可运行玩具样例
│   ├── run_analyze.py                 # 命令行分析入口
│   └── export_llamafactory_dataset.py # 导出 LLaMA-Factory 格式数据
├── src/
│   └── rs_xbd_agent/
│       ├── agent/                      # Agent 编排逻辑
│       ├── data/                       # xBD 数据解析与指令数据构造
│       ├── eval/                       # mask 评估指标
│       ├── models/                     # Qwen/GeoChat/Mock 适配器与损毁分析工具
│       ├── report/                     # Markdown / JSON 报告生成
│       ├── tools/                      # Agent 工具接口、定位/损毁工具后端
│       └── utils/                      # 图像工具
├── tests/
│   └── test_core.py
├── .env.example
├── requirements.txt
└── pyproject.toml
```

建议后续重构为更清晰的分层结构：

```text
src/rs_xbd_agent/
├── predictors/              # 建筑物定位与损毁等级预测接口
│   ├── base.py
│   ├── gt_mask_provider.py  # 读取 xBD GT mask，仅用于演示/评估
│   ├── external_mask.py     # 读取外部模型预测 mask
│   └── mock_predictor.py    # 玩具样例/流程测试
├── analysis/                # 后处理与统计
│   ├── instance_extractor.py
│   ├── damage_statistics.py
│   └── uncertainty.py
├── vlm/                     # 多模态大模型解释与复核模块
│   ├── qwen_model.py
│   ├── geochat_model.py
│   └── mock_vlm.py
├── agent/                   # 流程编排
├── report/                  # Markdown / JSON 报告生成
└── utils/
```

---

## 5. 环境安装

建议 Python 3.10 或 3.11。

```bash
conda create -n xbd-mllm python=3.10 -y
conda activate xbd-mllm
pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple
pip install -e .
```

如果需要接入 ModelScope `Qwen/Qwen3.5-4B`，额外安装：

```bash
pip install torch transformers accelerate modelscope qwen-vl-utils -i https://mirrors.aliyun.com/pypi/simple
```

不同 Qwen-VL / Qwen3.5 模型的加载类、processor、chat template 和 `transformers` 版本可能不同。实际部署时请以模型仓库 README、`config.json` 和官方 inference demo 为准。Qwen3-VL-4B/8B-Instruct 也可以作为较稳定的替代多模态后端。

---

## 6. 快速运行玩具样例

生成一组模拟灾前/灾后图像与 mask：

```bash
python scripts/create_toy_sample.py --out_dir outputs/toy_sample
```

运行分析：

```bash
python scripts/run_analyze.py \
  --pre_image outputs/toy_sample/pre.png \
  --post_image outputs/toy_sample/post.png \
  --pre_mask outputs/toy_sample/pre_building_mask.png \
  --post_damage_mask outputs/toy_sample/post_damage_mask.png \
  --out_dir outputs/demo_report
```

输出内容包括：

```text
outputs/demo_report/
├── damage_overlay.png
├── change_heatmap.png
├── change_mask.png
├── damage_mask.png
├── report.md
└── result.json
```

---

## 7. 处理 xBD 原始数据

xBD 常见目录形式如下：

```text
xBD/
├── images/
│   ├── xxx_pre_disaster.png
│   └── xxx_post_disaster.png
└── labels/
    ├── xxx_pre_disaster.json
    └── xxx_post_disaster.json
```

将 JSON 标注转成 mask：

```bash
python scripts/build_xbd_masks.py \
  --labels_dir /path/to/xBD/labels \
  --out_dir outputs/xbd_masks
```

输出：

```text
outputs/xbd_masks/
├── localization/      # 建筑物定位 mask，0 背景，1 建筑物
├── damage/            # 损毁等级 mask，0 背景，1 no-damage，2 minor，3 major，4 destroyed
└── index.jsonl        # 样本索引
```

---

## 8. Mask 来源说明

本项目支持三种 mask 来源：

```text
1. GT mask
   由 xBD 原始 JSON 标注生成，仅用于流程演示、数据构造和评估对照。
   使用 GT mask 时，系统不代表完成了自动损毁预测。

2. 外部模型预测 mask
   由用户已有的建筑物损毁评估网络生成，是正式自动化流程的主要输入。
   本项目负责后处理、统计、多模态解释和报告生成。

3. 无 mask 模式
   仅使用灾前/灾后图像差分生成粗略变化热力图。
   该模式只能用于变化初筛，不能输出可靠的建筑物损毁等级。
```

现阶段脚本通过参数直接接收 mask：

```bash
--pre_mask /path/to/pre_building_or_localization_mask.png
--post_damage_mask /path/to/post_damage_mask.png
```

如果你接入自己的损毁评估模型，只需要把模型预测结果保存为同样的 mask 格式，再传给上述参数即可。

### 8.1 Agent 工具后端与推荐外部模型

当前 `run_analyze.py` 已经支持显式选择工具后端：

```bash
python scripts/run_analyze.py \
  --pre_image outputs/toy_sample/pre.png \
  --post_image outputs/toy_sample/post.png \
  --pre_mask outputs/toy_sample/pre_building_mask.png \
  --post_damage_mask outputs/toy_sample/post_damage_mask.png \
  --localization_backend provided_mask \
  --damage_backend provided_mask \
  --out_dir outputs/demo_report_agent
```

也可以在 `.env` 中配置：

```bash
LOCALIZATION_BACKEND=provided_mask
DAMAGE_BACKEND=provided_mask
```

后端含义：

```text
provided_mask / gt_mask / external_mask
  读取已经存在的 mask，适合当前 xBD GT mask、外部模型预测 mask 和流程验证。

diff_fallback
  不调用专门损毁模型，交给 XBDDamageAnalyzer 使用差分/建筑物区域交集做兜底。

xview2_baseline
  推荐作为建筑物定位工具接入。官方 xView2 baseline 提供 localization 与 damage classification 两阶段流程，定位部分基于 SpaceNet building detection 思路，和 xBD/xView2 数据格式最贴近。

xview2_strong_baseline
  推荐作为损毁评估工具接入。Xview2 Strong Baseline 是面向 xBD building damage detection 的 PyTorch/PyTorch-Lightning 强基线，适合作为自动 damage mask / damage polygon 预测模块。
```

推荐工程分工：

```text
Building Localization Tool
  首选：官方 xView2 baseline localization
  当前可运行：provided_mask / gt_mask / external_mask

Damage Assessment Tool
  首选：Xview2 Strong Baseline
  备选：官方 xView2 baseline damage classification
  当前可运行：provided_mask / gt_mask / external_mask / diff_fallback

VLM Review Tool
  Qwen3.5-4B / GeoChat 负责解释、复核、问答和报告，不负责替代像素级分割模型。
```

参考仓库：

```text
官方 xView2 baseline: https://github.com/DIUx-xView/xView2_baseline
Xview2 Strong Baseline: https://github.com/PaulBorneP/Xview2_Strong_Baseline
xBD 数据集论文: https://arxiv.org/abs/1911.09296
```

### 8.2 外部工具命令接入方式

本项目已经支持 `external_command` 后端。外部工具只需要满足一个约定：运行后把预测 mask 写到 `{output_mask}`。

可用占位符：

```text
{pre_image}          灾前图像路径
{post_image}         灾后图像路径
{pre_mask}           当前已有建筑物 mask，可能为空
{post_damage_mask}   当前已有损毁 mask，可能为空
{out_dir}            外部工具输出目录
{output_mask}        外部工具必须写入的 mask 路径
```

不要在命令模板里给占位符额外加引号，程序会自动处理路径。

示例：用项目自带的 mock 外部工具验证 Agent 调用链路：

```bash
VLM_BACKEND=mock python scripts/run_analyze.py \
  --pre_image outputs/toy_sample/pre.png \
  --post_image outputs/toy_sample/post.png \
  --localization_backend external_command \
  --damage_backend external_command \
  --localization_command "python scripts/mock_external_mask_tool.py --pre {pre_image} --post {post_image} --output {output_mask} --mode localization" \
  --damage_command "python scripts/mock_external_mask_tool.py --pre {pre_image} --post {post_image} --output {output_mask} --mode damage" \
  --out_dir outputs/demo_report_external_tool
```

接入官方 xView2 baseline 时，项目已经提供封装脚本 `scripts/run_xview2_baseline.py`。官方 xView2 baseline 推理需要：

```text
1. DIUx-xView/xView2_baseline 仓库
2. localization model weights
3. classification model weights
4. 灾前图像和灾后图像
```

官方 `utils/inference.sh` 的输出本身就是本项目需要的损毁等级 PNG：

```text
0 = no building
1 = no-damage
2 = minor-damage
3 = major-damage
4 = destroyed
```

配置 `.env`：

```bash
LOCALIZATION_BACKEND=xview2_baseline
DAMAGE_BACKEND=xview2_baseline
XVIEW2_REPO=/root/autodl-tmp/xView2_baseline
XVIEW2_LOCALIZATION_WEIGHTS=/root/autodl-tmp/xView2_baseline/weights/localization.h5
XVIEW2_CLASSIFICATION_WEIGHTS=/root/autodl-tmp/xView2_baseline/weights/classification.hdf5
```

然后直接运行：

```bash
python scripts/run_analyze.py \
  --pre_image /path/to/pre_disaster.png \
  --post_image /path/to/post_disaster.png \
  --out_dir outputs/case_xview2_baseline
```

也可以不写 `.env`，直接显式调用封装脚本作为外部命令：

```bash
python scripts/run_analyze.py \
  --pre_image /path/to/pre_disaster.png \
  --post_image /path/to/post_disaster.png \
  --localization_backend external_command \
  --damage_backend external_command \
  --localization_command "python scripts/run_xview2_baseline.py --repo /root/autodl-tmp/xView2_baseline --pre {pre_image} --post {post_image} --localization-weights /path/to/localization.h5 --classification-weights /path/to/classification.hdf5 --output {output_mask} --mode localization" \
  --damage_command "python scripts/run_xview2_baseline.py --repo /root/autodl-tmp/xView2_baseline --pre {pre_image} --post {post_image} --localization-weights /path/to/localization.h5 --classification-weights /path/to/classification.hdf5 --output {output_mask} --mode damage" \
  --out_dir outputs/case_xview2_baseline
```

如果要接入 Strong Baseline，把命令替换成该仓库自己的推理封装脚本即可。例如：

```bash
python scripts/run_analyze.py \
  --pre_image /path/to/pre.png \
  --post_image /path/to/post.png \
  --damage_backend xview2_strong_baseline \
  --damage_command "python /path/to/Xview2_Strong_Baseline/infer_damage.py --pre {pre_image} --post {post_image} --pre-mask {pre_mask} --output {output_mask}" \
  --out_dir outputs/case_with_strong_baseline
```

上面的 `infer_localization.py` / `infer_damage.py` 是示意名称。实际接入时，需要根据第三方仓库真实的推理入口写一个薄封装脚本，把它们的 polygon、json 或概率图转换成本项目要求的 PNG mask：

```text
建筑物定位 mask: 0 背景，1 建筑物
损毁等级 mask:   0 背景，1 no-damage，2 minor，3 major，4 destroyed
```

---

## 9. 构造面向“损毁解释与复核”的多模态指令数据

微调目标不是让 Qwen / GeoChat 直接学习输出像素级 damage mask，而是让多模态模型学习：

```text
1. 根据灾前/灾后图像和 overlay 解释损毁结果。
2. 根据结构化统计生成报告。
3. 根据局部 crop 回答指定建筑的损毁原因。
4. 输出低置信度样本或可疑区域的人工复核建议。
```

如果只需要原始统计报告风格的 baseline 数据，可以构造项目内部 JSONL：

```bash
python scripts/build_instruction_dataset.py \
  --image_dir /path/to/xBD/images \
  --mask_index outputs/xbd_masks/index.jsonl \
  --out_file outputs/xbd_instruction.jsonl
```

导出为 LLaMA-Factory 多模态格式：

```bash
python scripts/export_llamafactory_dataset.py \
  --input outputs/xbd_instruction.jsonl \
  --output outputs/llamafactory_xbd.json
```

更推荐用于 LoRA 的是解释与复核三段式数据，它会把回答组织为“复核结论 / 主要依据 / 人工复核建议”，并加入建筑物像素、损毁等级比例、严重损毁比例和严重区域连通性等可计算依据：

```bash
python scripts/build_instruction_dataset.py \
  --image_dir /root/autodl-tmp/xBD/images \
  --mask_index outputs/xbd_masks/index.jsonl \
  --out_file outputs/xbd_instruction_review.jsonl \
  --style review

python scripts/export_llamafactory_dataset.py \
  --input outputs/xbd_instruction_review.jsonl \
  --output outputs/llamafactory_xbd_review.json
```

默认导出格式包含：

```json
{
  "messages": [
    {
      "role": "user",
      "content": "<image><image>请对这组灾前和灾后遥感图像进行建筑物损毁评估，并说明主要依据。"
    },
    {
      "role": "assistant",
      "content": "根据灾后建筑物损毁标注统计，该区域..."
    }
  ],
  "images": ["xxx_pre_disaster.png", "xxx_post_disaster.png"]
}
```

### 本次 LoRA 微调的数据性质与局限

本项目当前已完成一次 Qwen3.5-4B 的 xBD 多模态 QLoRA 实践，训练数据使用的是：

```text
文本 JSON 指令 + 灾前原始图像 + 灾后原始图像
```

也就是说，训练不是纯文本 SFT。LLaMA-Factory 会读取 `messages` 字段中的文本，同时根据 `images` 字段加载两张遥感图像，并在训练样本中展开为类似：

```text
<|vision_start|><|image_pad|>...<|vision_end|>
<|vision_start|><|image_pad|>...<|vision_end|>
```

但需要注意：虽然输入包含图像，当前 assistant 标签主要由 xBD damage mask 的统计模板生成，监督信号更强调“统计结果如何写成复核报告”，而不是人工标注的视觉证据描述。因此这版 LoRA 的实际效果是：

```text
优势：三段式报告格式更稳定，输出更贴近“复核结论 / 主要依据 / 人工复核建议”。
不足：主要依据容易变成统计字段复述，对原始灾前/灾后图像的视觉理解没有明显增强；还可能出现损毁等级术语或百分比混淆。
```

实际对比中，LoRA 版报告比 Base Qwen3.5 更稳定，但曾出现把“完全损毁 380 像素 / 9.36%”泛化为“严重损毁”的问题。这说明当前 LoRA 更适合作为 **报告格式对齐 baseline**，不能证明模型已经学会可靠的图像级损毁判读。

下一步更合理的数据增强方向是构造区域级视觉复核样本，例如加入建筑物局部 crop、灾前 crop、灾后 crop 和 damage overlay crop，并在 assistant 标签中显式写入：

```text
1. 灾前图像中建筑物轮廓、屋顶纹理和周边环境状态。
2. 灾后图像中是否出现屋顶破碎、结构缺失、边界模糊、碎片或明显纹理变化。
3. damage mask 与原始图像视觉证据是否一致。
4. 哪些区域需要人工复核，以及可能的误检/漏检原因。
```

更推荐的增强样本形式是加入 overlay 或局部 crop：

```json
{
  "messages": [
    {
      "role": "user",
      "content": "<image><image><image>请结合灾前图、灾后图和损毁等级叠加图，解释右侧建筑物为什么被判定为 no-damage，并指出是否需要人工复核。"
    },
    {
      "role": "assistant",
      "content": "该建筑在灾后图像中屋顶轮廓仍保持完整，未见明显坍塌或消失，因此判定为 no-damage；但其周边地表变化明显，建议人工复核建筑周边道路和遮挡区域。"
    }
  ],
  "images": [
    "xxx_pre_disaster.png",
    "xxx_post_disaster.png",
    "xxx_damage_overlay.png"
  ]
}
```

如果只想训练纯文本后端，可以导出文本 SFT 格式：

```bash
python scripts/export_llamafactory_dataset.py \
  --input outputs/xbd_instruction.jsonl \
  --output outputs/llamafactory_xbd_text.json \
  --text_only
```

---

## 10. 接入 ModelScope Qwen/Qwen3.5-4B

默认配置仍然是 `mock`，方便无 GPU 环境跑通。要启用 Qwen3.5-4B：

1. 安装依赖：

```bash
pip install torch transformers accelerate modelscope qwen-vl-utils -i https://mirrors.aliyun.com/pypi/simple
```

2. 下载模型权重，可选但推荐：

```bash
modelscope download \
  --model Qwen/Qwen3.5-4B \
  --local_dir /root/autodl-tmp/models/Qwen3.5-4B
```

3. 创建或修改 `.env`：

```bash
cp .env.example .env
```

使用本地权重：

```bash
VLM_BACKEND=qwen_multimodal
MLLM_MODEL_NAME=/root/autodl-tmp/models/Qwen3.5-4B
MLLM_DEVICE=auto
```

也可以直接写 ModelScope 模型名，代码会尝试通过 `modelscope.snapshot_download` 下载：

```bash
VLM_BACKEND=qwen_multimodal
MLLM_MODEL_NAME=Qwen/Qwen3.5-4B
MLLM_DEVICE=auto
```

旧变量 `MLLM_BACKEND` 仍然兼容，但推荐新配置名 `VLM_BACKEND`，因为它更准确地表达“大模型只负责解释与复核”。

4. 运行分析：

```bash
python scripts/run_analyze.py \
  --pre_image outputs/toy_sample/pre.png \
  --post_image outputs/toy_sample/post.png \
  --pre_mask outputs/toy_sample/pre_building_mask.png \
  --post_damage_mask outputs/toy_sample/post_damage_mask.png \
  --out_dir outputs/demo_report_qwen35
```

`qwen_multimodal` 后端会向模型输入：

```text
1. 灾前图像
2. 灾后图像
3. change_heatmap.png，如果已生成
4. damage_overlay.png，如果已生成
5. 结构化 summary JSON，包括变化比例、损毁等级统计、bbox 等
```

注意：Qwen3.5-4B 的作用是视觉证据解释、结果复核和报告生成，不是替代专用损毁分割模型。

---

## 11. GeoChat 对照实验与可选后端

GeoChat 是面向遥感场景的 grounded vision-language model，论文和官方仓库说明它基于 LLaVA-1.5 架构进行遥感领域微调，支持遥感图像问答、场景理解、grounding、区域级对话等能力。官方仓库：https://github.com/mbzuai-oryx/GeoChat，权重地址：https://huggingface.co/MBZUAI/geochat-7B。

当前版本优先将 GeoChat 作为独立遥感 VLM 对照模型；只有在实现 `GeoChatModel.generate()` 适配器后，才作为内置后端启用。

### 11.1 安装 GeoChat 独立环境

GeoChat 的依赖与本项目当前 `transformers` / Qwen 环境可能存在版本差异，建议单独创建环境，不要直接混装到 `xbd-mllm` 环境里。

```bash
cd /root/autodl-tmp

git clone https://github.com/mbzuai-oryx/GeoChat.git
cd GeoChat

conda create -n geochat python=3.10 -y
conda activate geochat

pip install --upgrade pip
pip install -e .
```

如果你需要训练或遇到加速算子需求，官方 README 还给出：

```bash
pip install ninja
pip install flash-attn --no-build-isolation
```

`flash-attn` 对 CUDA / PyTorch / GCC 版本比较敏感，安装失败时可以先不装，只做基础推理验证。

### 11.2 下载 GeoChat-7B 权重

推荐用 Hugging Face CLI 下载：

```bash
pip install -U huggingface_hub

huggingface-cli download MBZUAI/geochat-7B \
  --local-dir /root/autodl-tmp/models/geochat-7B
```

如果网络访问 Hugging Face 不稳定，可以手动在浏览器打开权重页下载：

```text
https://huggingface.co/MBZUAI/geochat-7B
```

### 11.3 启动 GeoChat 官方 Demo

在 GeoChat 仓库目录中运行：

```bash
cd /root/autodl-tmp/GeoChat
conda activate geochat

python geochat_demo.py \
  --model-path /root/autodl-tmp/models/geochat-7B
```

然后在 GeoChat Web UI 中上传 xBD 灾前 / 灾后图像，提问示例：

```text
Please analyze this remote sensing image and describe damaged buildings or disaster-related changes.
```

或者对灾后图像提问：

```text
Are there visibly damaged buildings in this post-disaster remote sensing image? Please explain the visual evidence.
```

### 11.4 与本项目结合的推荐方式

```text
xBD 图像
-> 本项目生成 change_heatmap / damage_overlay / report.md
-> GeoChat 官方 Demo 对灾前、灾后、叠加图进行问答
-> 对比 Qwen3.5-4B 与 GeoChat 的复核意见和遥感解释能力
```

如果后续要把 GeoChat 接成本项目内置后端，建议新增：

```text
src/rs_xbd_agent/models/geochat_model.py
```

并让它实现和 `MockVisionLanguageModel` / `QwenMultimodalModel` 一样的接口：

```python
def generate(self, prompt: str, image_paths: list[str] | None = None, context: dict | None = None) -> VLMResponse:
    ...
```

然后在 `src/rs_xbd_agent/models/factory.py` 中增加：

```python
if cfg.mllm_backend == "geochat":
    return GeoChatModel(...)
```

`.env` 可以设计成：

```bash
VLM_BACKEND=geochat
MLLM_MODEL_NAME=/root/autodl-tmp/models/geochat-7B
```

### 11.5 能力边界

GeoChat 是遥感领域 VLM，比通用 VLM 更适合遥感图像问答和 grounding。但它仍然不是专门为 xBD 像素级 damage mask 预测训练的分割模型。对于严格的 xBD 评估，仍建议：

```text
损毁分割/变化检测模型负责输出 mask
GeoChat 或 Qwen3.5-4B 负责解释图像、结合统计信息生成复核意见和报告
```

---

## 12. 启动 FastAPI 服务

```bash
uvicorn app.api:app --host 0.0.0.0 --port 8000 --reload
```

接口：

```bash
curl -X POST "http://127.0.0.1:8000/analyze" \
  -F "pre_image=@outputs/toy_sample/pre.png" \
  -F "post_image=@outputs/toy_sample/post.png" \
  -F "pre_mask=@outputs/toy_sample/pre_building_mask.png" \
  -F "post_damage_mask=@outputs/toy_sample/post_damage_mask.png"
```

API 会读取 `.env` 或 `configs/default.yaml` 中的 `VLM_BACKEND` / `MLLM_BACKEND` 设置。

---

## 13. 启动 Streamlit Demo

```bash
streamlit run app/streamlit_demo.py
```

页面支持上传：

- 灾前图像
- 灾后图像
- 灾前建筑物 mask，可选
- 灾后损毁等级 mask，可选

没有 mask 时，系统会使用图像差分生成粗略变化热力图；有 xBD mask 或外部预测 mask 时，报告会包含更可靠的损毁等级统计。

---

## 14. 评估方式

像素级指标用于评估建筑物损毁预测模型或外部 mask，不用于直接评价多模态大模型的报告质量。

预测 mask 评估：

```text
1. IoU：各损毁类别的交并比
2. F1：像素级分类 F1
3. OA：Overall Accuracy
```

多模态解释与复核模块可以单独评估：

```text
1. 等级一致性：报告中的损毁等级是否与 mask 统计一致。
2. 位置一致性：报告提到的区域是否与 bbox / overlay 对应。
3. 幻觉率：是否描述了图像中不存在的建筑或损毁。
4. 复核有效性：低置信度样本是否被正确标记为需要人工复核。
5. 人工评分：解释是否合理、清晰、可用于辅助决策。
```

---

## 15. 能力边界

当前项目中，损毁评估结果来自以下来源：

```text
有 post_damage_mask：统计 xBD 标注或外部模型预测的损毁等级 mask
有 pre_mask 但无 post_damage_mask：仅统计建筑物区域内的变化，不能判断损毁等级
无 mask：仅做灾前/灾后像素差分变化检测初筛
```

Qwen3.5-4B / GeoChat 可以参与图像理解、报告生成、视觉一致性检查和交互式问答，但它们不是专门的 xBD 像素级损毁分割网络。更可靠的工程路线是：

```text
pre/post image
-> 建筑物检测 / 损毁分割模型预测 mask
-> 后处理与损毁统计
-> 多模态大模型结合图像与结构化证据生成解释、复核建议和报告
```

---

## 16. 简历项目介绍

项目名称建议：

> 基于遥感损毁评估模型与多模态大模型的 xBD 灾害影像交互式解译系统

简历描述：

```text
基于遥感损毁评估模型与多模态大模型的 xBD 灾害影像交互式解译系统
技术栈：Python、PyTorch、Transformers、ModelScope、Qwen3.5/Qwen-VL、FastAPI、Streamlit、xBD、遥感变化检测、LoRA

- 面向 xBD 灾前/灾后遥感影像，构建建筑物损毁评估与交互式解译系统，支持建筑物 mask、损毁等级 mask、变化热力图、实例级 bbox 和结构化统计结果的统一管理。
- 设计预测模块与解释模块解耦的工程架构，支持读取 xBD 标注 mask、外部损毁评估模型预测结果或规则版 mock 输出，并基于后处理流程生成建筑物数量、损毁等级分布、面积占比和低置信度区域。
- 引入多模态大模型作为结果解释与复核模块，将灾前图、灾后图、损毁叠加图、局部建筑 crop 和结构化统计信息联合输入，实现损毁等级解释、指定区域问答、低置信度样本复核建议和 Markdown 灾情报告生成。
- 封装 FastAPI 服务与 Streamlit 可视化 Demo，支持图像上传、mask 可视化、损毁统计、区域问答和报告导出，并提供 IoU、F1、OA 等指标用于评估外部损毁预测模型。
```

---

## 17. 后续可扩展方向

- 将官方 xView2 baseline localization 封装为 `BuildingLocalizationTool`，在缺少 `pre_mask` 时自动预测建筑物区域。
- 将 Xview2 Strong Baseline 或官方 damage classifier 封装为 `DamageAssessmentTool`，在缺少 `post_damage_mask` 时自动预测损毁等级。
- 为高风险 bbox 自动裁剪灾前/灾后 crop、overlay crop 和局部统计，再交给 Qwen3.5-4B / GeoChat 做二次复核。
- 增加低置信度区域和疑似误检/漏检区域的自动抽取。
- 增加视觉 grounding，让报告引用具体损毁区域 bbox。
- 使用 Qwen3.5-4B 或 GeoChat 对 xBD 多模态“解释与复核”指令数据做 LoRA / QLoRA 微调。
- 加入 LangGraph，把定位、损毁评估、统计、复核、问答、报告生成拆成状态节点。
- 增加批处理模式，对整个 xBD 测试集自动生成评估报告和复核清单。
