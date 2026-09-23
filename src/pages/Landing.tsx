import { Link } from 'react-router'
import '../App.css'

type Entry = {
  to: string
  thumb: string
  title: string
  desc: string
  tags: string[]
}

const SEC_2D: Entry[] = [
  {
    to: '/flow',
    thumb: 'flow.jpg',
    title: '2D 推理流程动画',
    desc: '一个 prompt 的完整旅程：客户端 → 网络 → Tokenizer → Prefill → decode 循环 → 回答。MoE / Dense 双架构切换，15 步可交互播放。',
    tags: ['MoE vs Dense', '15 步动画', '可单步/变速'],
  },
]

const SEC_ENGINE: Entry[] = [
  {
    to: '/gpu#prefill', thumb: 'prefill.jpg', title: 'Prefill 的 GPU',
    desc: '算力密集：8 路 token 并行涌入、SM 阵列全闪、KV Cache 逐格写满。', tags: ['Compute-bound', 'Three.js'],
  },
  {
    to: '/gpu#decode', thumb: 'decode.jpg', title: 'Decode 的 GPU',
    desc: '带宽密集：每步仅 1 个 token，SM 大面积闲置，HBM 与 KV Cache 被打满。', tags: ['Memory-bound', 'GEMV'],
  },
  {
    to: '/gpu#nvlink', thumb: 'nvlink.jpg', title: 'NVLink 多卡互联',
    desc: '8 卡环形全互联，经 NVSwitch 900 GB/s 直达——张量并行的基石。', tags: ['900 GB/s', '全互联'],
  },
  {
    to: '/gpu#pcie', thumb: 'pcie.jpg', title: 'PCIe 多卡互联',
    desc: '星型拓扑经 PCIe Switch/CPU 中转，带宽约为 NVLink 的 1/14。', tags: ['64 GB/s', '星型拓扑'],
  },
]

const SEC_PARALLEL: Entry[] = [
  {
    to: '/gpu#tp', thumb: 'tp.jpg', title: '张量并行 TP',
    desc: '一层权重切成 4 片放到 4 张卡，token 广播 + all-reduce 合并（Megatron-LM）。', tags: ['Megatron-LM', 'all-reduce'],
  },
  {
    to: '/gpu#pp', thumb: 'pp.jpg', title: '流水线并行 PP',
    desc: '按层分段，4 色微批次错位推进——流水线气泡与 1F1B 调度（GPipe / PipeDream）。', tags: ['微批次', '气泡'],
  },
  {
    to: '/gpu#ep', thumb: 'ep.jpg', title: '专家并行 EP',
    desc: '8 个彩色专家分置 4 卡，Router 把 token 定向「快递」——all-to-all（GShard / DeepSeek）。', tags: ['all-to-all', 'DeepSeek'],
  },
]

function Card({ e }: { e: Entry }) {
  return (
    <Link to={e.to}
      className="group rounded-xl border border-slate-800 bg-[#0d1626] overflow-hidden transition-all duration-300 hover:border-cyan-500/60 hover:shadow-[0_0_28px_rgba(34,211,238,0.15)] hover:-translate-y-1">
      <div className="relative aspect-video overflow-hidden">
        <img src={`${import.meta.env.BASE_URL}thumbs/${e.thumb}`} alt={e.title}
          className="w-full h-full object-cover object-center transition-transform duration-500 group-hover:scale-105" />
        <div className="absolute inset-0 bg-gradient-to-t from-[#0d1626] via-transparent to-transparent" />
      </div>
      <div className="p-4">
        <h3 className="text-base font-bold text-white group-hover:text-cyan-300 transition-colors">{e.title}</h3>
        <p className="mt-1.5 text-[13px] text-slate-400 leading-relaxed">{e.desc}</p>
        <div className="mt-3 flex flex-wrap gap-1.5">
          {e.tags.map((t) => (
            <span key={t} className="text-[10px] px-2 py-0.5 rounded-full bg-slate-800 text-slate-400 border border-slate-700">
              {t}
            </span>
          ))}
        </div>
      </div>
    </Link>
  )
}

export default function Landing() {
  return (
    <div className="min-h-screen bg-[#070d1a] text-slate-200 page-font">
      {/* Hero */}
      <header className="max-w-[1360px] mx-auto px-5 pt-16 pb-12 text-center">
        <div className="inline-block text-[11px] px-3 py-1 rounded-full bg-cyan-500/10 text-cyan-300 border border-cyan-500/30 tracking-wide">
          LEARN AI INFRA · 交互式学习项目
        </div>
        <h1 className="mt-5 text-4xl md:text-5xl font-bold tracking-tight text-white">
          AI 基础设施，<span className="text-cyan-300">一眼看懂</span>
        </h1>
        <p className="mt-4 text-slate-400 max-w-2xl mx-auto leading-relaxed">
          用可交互的 2D 动画和 3D 场景理解大模型推理：MoE 与 Dense 的引擎内部、Prefill 与 Decode 的 GPU 状态、NVLink 与 PCIe 的互联差异、TP / PP / EP 三大并行策略。
        </p>
        <div className="mt-8 flex flex-wrap justify-center gap-3">
          <Link to="/flow" className="ctl-btn bg-cyan-500/90 hover:bg-cyan-400 text-slate-950 font-semibold !px-6 !py-2.5 !text-sm">
            ▶ 进入 2D 推理流程
          </Link>
          <Link to="/gpu" className="ctl-btn bg-slate-800 hover:bg-slate-700 border border-slate-700 !px-6 !py-2.5 !text-sm">
            🧊 打开 3D GPU 实验室
          </Link>
          <a href="https://github.com/hawkli-1994/learn-ai-infra" target="_blank" rel="noreferrer"
            className="ctl-btn bg-slate-800 hover:bg-slate-700 border border-slate-700 !px-6 !py-2.5 !text-sm">
            ⭐ GitHub
          </a>
        </div>
      </header>

      {/* 内容索引 */}
      <main className="max-w-[1360px] mx-auto px-5 pb-16">
        <section className="mt-4">
          <h2 className="text-lg font-bold text-white mb-4 flex items-center gap-2">
            <span className="text-cyan-300">01</span> 2D 推理流程
            <span className="text-xs font-normal text-slate-500">一个 prompt 从发出到回答的完整旅程</span>
          </h2>
          <div className="grid md:grid-cols-2 xl:grid-cols-3 gap-4">
            {SEC_2D.map((e) => <Card key={e.to} e={e} />)}
          </div>
        </section>

        <section className="mt-12">
          <h2 className="text-lg font-bold text-white mb-4 flex items-center gap-2">
            <span className="text-cyan-300">02</span> 3D 引擎内部
            <span className="text-xs font-normal text-slate-500">推理时 GPU 里发生了什么 · 可拖动旋转</span>
          </h2>
          <div className="grid md:grid-cols-2 xl:grid-cols-4 gap-4">
            {SEC_ENGINE.map((e) => <Card key={e.to} e={e} />)}
          </div>
        </section>

        <section className="mt-12">
          <h2 className="text-lg font-bold text-white mb-4 flex items-center gap-2">
            <span className="text-cyan-300">03</span> 3D 并行策略
            <span className="text-xs font-normal text-slate-500">大模型如何撒到多张卡上</span>
          </h2>
          <div className="grid md:grid-cols-2 xl:grid-cols-3 gap-4">
            {SEC_PARALLEL.map((e) => <Card key={e.to} e={e} />)}
          </div>
        </section>
      </main>

      <footer className="border-t border-slate-800/60 py-8 text-center text-xs text-slate-500">
        Learn AI Infra · 灵感来自 <a className="hover:text-cyan-300" href="https://arxiv.org/abs/2406.04692" target="_blank" rel="noreferrer">Mixture-of-Agents (arXiv:2406.04692)</a> · MIT License ·{' '}
        <a className="hover:text-cyan-300" href="https://github.com/hawkli-1994/learn-ai-infra" target="_blank" rel="noreferrer">GitHub</a>
      </footer>
    </div>
  )
}
