# Learn AI Infra · MoE vs Dense 推理流程动画

[![Deploy to GitHub Pages](https://github.com/hawkli-1994/learn-ai-infra/actions/workflows/deploy.yml/badge.svg)](https://github.com/hawkli-1994/learn-ai-infra/actions/workflows/deploy.yml)

**在线演示**：[https://hawkli-1994.github.io/learn-ai-infra/](https://hawkli-1994.github.io/learn-ai-infra/)（`/` 作品落地页 · `/flow` 2D 流程 · `/gpu` 3D 实验室）

一个交互式动画演示：一个 prompt 从客户端通过网络进入 LLM 推理引擎，内部经历 Prefill / Decode 循环，最终变成回答返回的完整旅程。

支持 **MoE（Mixture-of-Experts，混合专家）** 与 **Dense（稠密模型）** 两种架构一键切换对比，动画逐步展示两者在 FFN 层的核心差异：

| | MoE | Dense |
|---|---|---|
| FFN 层 | N 个 Expert + Router，每 token 只激活 Top-2 | 单一 FFN，每 token 动用全部参数 |
| 动画亮点 | 门控打分柱状图、专家发光/闲置、加权求和 | 无路由直通、全量 FFN 发光 |
| 代价 | 全部专家需驻留显存 | 每步计算量 ∝ 总参数量 |

## 功能

- 🎬 15 步自动播放动画：客户端 → 网络 → API 网关 → Tokenizer → Prefill → decode 循环 ×N → EOS → Detokenizer → 响应返回
- 🧩 / 🔩 标签页切换 MoE / Dense 架构（支持 `#dense` 链接直达）
- ⏯ 播放 / 暂停 / 单步 / 重置 / 0.5–2× 变速 / 点击进度点跳转
- 🧊 **3D GPU 实验室**（`/gpu`，Three.js）：Prefill 算力高峰、Decode 带宽瓶颈、NVLink 全互联高速路与 PCIe 星型拓扑的 3D 可交互场景（拖动旋转 / 滚轮缩放）
- ✂️🏭📦 **并行策略 3D 对比**：张量并行 TP（权重切片 + all-reduce）、流水线并行 PP（微批次错位推进 + 气泡）、专家并行 EP（Router 定向 all-to-all）
- 📖 每一步的同步文字解说 + 4 张架构知识卡片

## 技术栈

React + TypeScript + Vite + Tailwind CSS，纯前端单页应用，无后端依赖。

## 本地运行

```bash
npm install
npm run dev
```

构建：

```bash
npm run build   # 输出到 dist/
```

## 灵感

源于 Together AI 论文 [Mixture-of-Agents Enhances Large Language Model Capabilities](https://arxiv.org/abs/2406.04692)（arXiv:2406.04692）的阅读笔记——MoE 是模型内部、激活级的稀疏专家组合；MoA 是模型之间、提示词级的协作，两者都源于「术业有专攻」。

## License

MIT
