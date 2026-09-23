import { useEffect, useRef, useState } from 'react'
import { Link } from 'react-router'
import '../App.css'

/* ------------------------------------------------------------------ */
/* 画布坐标系：1300 x 800                                              */
/* ------------------------------------------------------------------ */

const W = 1300
const H = 800

type Pt = { x: number; y: number }
type Arch = 'moe' | 'dense'

/** 每个动画步骤的定义 */
type StepDef = {
  title: string
  desc: string
  dur: number // 秒
  packet: Pt | null
}

const COMMON_DESC: Record<number, string> = {
  0: '点击「播放」，看一个 prompt 如何穿过网络、进入推理引擎，再变成回答返回给你。',
  1: '客户端把你的输入和采样参数（temperature、max_tokens 等）打包成一个 ChatCompletion 请求。',
  2: 'HTTP POST /v1/chat/completions 穿过互联网到达推理服务网关。流式回答时，响应会以 SSE 逐段推回。',
  3: '网关做负载均衡、鉴权、限流，把请求放进调度队列，等待空闲的 GPU 推理实例接手。',
  4: '文本被切成 token 并映射为 id，例如 [法国][的][首都][是][哪里] → [2157, 6532, 9821, ...]。模型只认识 token，不认识文字。',
  9: '按 temperature 从分布中采样（贪心解码则直接取 argmax），得到「巴黎」，拼回上下文末尾。',
  11: '模型输出结束符 <eos>，推理引擎判定生成完毕，把 token 序列整体交给后处理。',
  12: 'token id 序列被逐条还原为自然语言字符串，并做后处理（去特殊符号、合并片段等）。',
  13: 'HTTP/SSE 把文本（流式模式下逐段）送回客户端。',
}

const MOE_STEPS: StepDef[] = [
  { title: '准备好了吗？', desc: COMMON_DESC[0], dur: 2, packet: null },
  { title: '① 构造请求', desc: COMMON_DESC[1], dur: 1.8, packet: { x: 110, y: 130 } },
  { title: '② 网络传输', desc: COMMON_DESC[2], dur: 1.9, packet: { x: 330, y: 110 } },
  { title: '③ API 服务器 / 调度', desc: COMMON_DESC[3], dur: 2.1, packet: { x: 545, y: 110 } },
  { title: '④ 分词 Tokenizer', desc: COMMON_DESC[4], dur: 2.2, packet: { x: 775, y: 110 } },
  { title: '⑤ 预填充 Prefill', desc: '所有 prompt token 一次性并行通过主干网络：每层 Attention 做全序列交互，每层的 MoE 层为每个 token 各自选专家。这一步建立起 KV Cache，之后每步不必重算历史。', dur: 2.6, packet: { x: 895, y: 245 } },
  { title: '⑥ 路由 Router 打分', desc: '当前 token 的隐状态 x 被送进小小的门控网络：G(x)=softmax(Wr·x)，为全部 8 个 Expert 打分，选出 Top-2。', dur: 2.0, packet: { x: 895, y: 245 } },
  { title: '⑦ 被选中的 Expert 开始计算', desc: '只有 2 个被选中的 Expert（各自是一个完整 FFN 子网络：up-proj → 激活函数 → down-proj）真正参与计算，其余 6 个完全闲置——这就是「稀疏激活」。', dur: 2.2, packet: { x: 895, y: 245 } },
  { title: '⑧ 加权求和 → 词表概率', desc: '两个 Expert 的输出按门控权重加权合并：y = 0.52·E₃(x) + 0.48·E₆(x)，再经 LM Head 映射为整个词表上的概率分布。', dur: 2.2, packet: { x: 895, y: 245 } },
  { title: '⑨ 采样出下一个 token', desc: COMMON_DESC[9], dur: 2.0, packet: { x: 895, y: 245 } },
  { title: '⑩ Decode 循环', desc: '新 token 成为下一步的输入，回到第⑥步再来一遍——每次只前进 1 个 token。演示跑 3 步；真实回答可能要几十到几百步。', dur: 1.8, packet: { x: 895, y: 245 } },
  { title: '⑪ 遇到 EOS，停止生成', desc: COMMON_DESC[11], dur: 1.7, packet: { x: 1080, y: 608 } },
  { title: '⑫ Detokenizer 还原文本', desc: COMMON_DESC[12], dur: 1.8, packet: { x: 860, y: 650 } },
  { title: '⑬ 响应沿网络返回', desc: COMMON_DESC[13], dur: 1.9, packet: { x: 625, y: 650 } },
  { title: '⑭ 客户端渲染回答', desc: '前端把文本显示在屏幕上——用户终于看到了回答。从你按下回车到这一刻，引擎内部已经跑了几十次「路由 → 选专家 → 加权 → 采样」的循环。', dur: 2.4, packet: { x: 320, y: 650 } },
  { title: '✅ 旅程结束', desc: '一次完整的 MoE 推理之旅：稀疏激活让每个 token 只动用 2/8 的专家算力，却拥有全部 8 个专家的知识。', dur: 2.5, packet: { x: 320, y: 650 } },
]

const DENSE_STEPS: StepDef[] = [
  { title: '准备好了吗？', desc: COMMON_DESC[0], dur: 2, packet: null },
  { title: '① 构造请求', desc: COMMON_DESC[1], dur: 1.8, packet: { x: 110, y: 130 } },
  { title: '② 网络传输', desc: COMMON_DESC[2], dur: 1.9, packet: { x: 330, y: 110 } },
  { title: '③ API 服务器 / 调度', desc: COMMON_DESC[3], dur: 2.1, packet: { x: 545, y: 110 } },
  { title: '④ 分词 Tokenizer', desc: COMMON_DESC[4], dur: 2.2, packet: { x: 775, y: 110 } },
  { title: '⑤ 预填充 Prefill', desc: '所有 prompt token 一次性并行通过主干网络：每层 Attention 做全序列交互，随后进入该层唯一的 FFN——没有任何选择，所有 token 走完全相同的路径。这一步建立起 KV Cache。', dur: 2.6, packet: { x: 895, y: 245 } },
  { title: '⑥ 隐状态直达 FFN', desc: '当前 token 的隐状态 x 直接送入这个 block 里唯一的 FFN。没有门控网络、没有 Top-K——每个 token 的计算图一模一样。', dur: 2.0, packet: { x: 895, y: 245 } },
  { title: '⑦ FFN 全量计算', desc: '整个 FFN（up-proj → 激活函数 → down-proj）的全部参数都参与本次计算。没有专家闲置，也没有选择发生——稠密模型的每一步都是「全员上班」。', dur: 2.2, packet: { x: 895, y: 245 } },
  { title: '⑧ FFN 输出 → 词表概率', desc: 'FFN 输出经残差连接与 LayerNorm 后，由 LM Head 映射为整个词表上的概率分布。相比 MoE，这里少了「多专家加权合并」这一步，结构更简单。', dur: 2.2, packet: { x: 895, y: 245 } },
  { title: '⑨ 采样出下一个 token', desc: COMMON_DESC[9], dur: 2.0, packet: { x: 895, y: 245 } },
  { title: '⑩ Decode 循环', desc: '新 token 成为下一步的输入，回到第⑥步再来一遍——每次只前进 1 个 token。演示跑 3 步；真实回答可能要几十到几百步。', dur: 1.8, packet: { x: 895, y: 245 } },
  { title: '⑪ 遇到 EOS，停止生成', desc: COMMON_DESC[11], dur: 1.7, packet: { x: 1080, y: 608 } },
  { title: '⑫ Detokenizer 还原文本', desc: COMMON_DESC[12], dur: 1.8, packet: { x: 860, y: 650 } },
  { title: '⑬ 响应沿网络返回', desc: COMMON_DESC[13], dur: 1.9, packet: { x: 625, y: 650 } },
  { title: '⑭ 客户端渲染回答', desc: '前端把文本显示在屏幕上——用户终于看到了回答。从你按下回车到这一刻，引擎内部已经跑了几十次「FFN → 采样」的循环。', dur: 2.4, packet: { x: 320, y: 650 } },
  { title: '✅ 旅程结束', desc: '一次完整的 Dense 推理之旅：每个 token 都动用了模型的全部参数——结构简单、行为稳定，但参数量越大，每一步的计算成本越高。这正是 MoE 想要攻克的痛点。', dur: 2.5, packet: { x: 320, y: 650 } },
]

const SAMPLED = ['巴黎', '是', '法国', '的', '首都', '。']
const ROUTER_SCORES = [0.06, 0.1, 0.72, 0.05, 0.09, 0.66, 0.04, 0.08]
const TOPK = [2, 5]

/* ------------------------------------------------------------------ */
/* 小组件                                                             */
/* ------------------------------------------------------------------ */

function Node({
  x, y, w, active, dim, children, className = '',
}: { x: number; y: number; w: number; active: boolean; dim?: boolean; children: React.ReactNode; className?: string }) {
  return (
    <div
      className={`moe-node ${active ? 'node-active' : ''} ${dim ? 'node-dim' : ''} ${className}`}
      style={{ left: x, top: y, width: w }}
    >
      {children}
    </div>
  )
}

/* ------------------------------------------------------------------ */
/* 主页面                                                             */
/* ------------------------------------------------------------------ */

export default function Home() {
  const [arch, setArch] = useState<Arch>(() =>
    typeof window !== 'undefined' && window.location.hash === '#dense' ? 'dense' : 'moe')
  const [step, setStep] = useState(0)
  const [playing, setPlaying] = useState(true)
  const [speed, setSpeed] = useState(1)
  const [decodeRound, setDecodeRound] = useState(0)
  const [genTokens, setGenTokens] = useState<string[]>([])
  const [scale, setScale] = useState(1)
  const wrapRef = useRef<HTMLDivElement>(null)

  const STEPS = arch === 'moe' ? MOE_STEPS : DENSE_STEPS
  const isMoe = arch === 'moe'

  // 自适应缩放画布
  useEffect(() => {
    const el = wrapRef.current
    if (!el) return
    const ro = new ResizeObserver(() => setScale(Math.min(1, el.clientWidth / W)))
    ro.observe(el)
    return () => ro.disconnect()
  }, [])

  // 主时钟：驱动步骤推进
  useEffect(() => {
    if (!playing) return
    const def = STEPS[step]
    const t = setTimeout(() => {
      if (step === 9) {
        // 离开采样步骤：记录本轮生成的 token
        setGenTokens((prev) => [...prev, SAMPLED[decodeRound]])
        if (decodeRound < 2) {
          setDecodeRound((r) => r + 1)
          setStep(6)
        } else {
          setStep(11)
        }
        return
      }
      setStep((s) => Math.min(s + 1, STEPS.length - 1))
    }, (def.dur * 1000) / speed)
    return () => clearTimeout(t)
  }, [step, playing, speed, decodeRound, STEPS])

  const switchArch = (a: Arch) => {
    if (a === arch) return
    setArch(a)
    window.location.hash = a === 'moe' ? '' : '#dense'
    setStep(0)
    setDecodeRound(0)
    setGenTokens([])
    setPlaying(true)
  }

  const reset = () => {
    setPlaying(false)
    setStep(0)
    setDecodeRound(0)
    setGenTokens([])
  }

  const jumpTo = (i: number) => {
    setPlaying(false)
    setStep(i)
    if (i < 6) { setDecodeRound(0); setGenTokens([]) }
  }

  const stepOnce = () => {
    setPlaying(false)
    if (step === 9) {
      setGenTokens((prev) => [...prev, SAMPLED[decodeRound]])
      if (decodeRound < 2) {
        setDecodeRound((r) => r + 1)
        setStep(6)
      } else setStep(11)
      return
    }
    setStep((s) => Math.min(s + 1, STEPS.length - 1))
  }

  const def = STEPS[step]
  const inDecode = step >= 6 && step <= 10
  const curToken = genTokens.length > 0 ? genTokens[genTokens.length - 1] : '哪里'
  const outToken = SAMPLED[Math.min(decodeRound, SAMPLED.length - 1)]

  return (
    <div className="min-h-screen bg-[#070d1a] text-slate-200 page-font">
      {/* 头部 */}
      <header className="max-w-[1360px] mx-auto px-5 pt-8 pb-4">
        <div className="flex flex-wrap items-center gap-4">
          <h1 className="text-2xl md:text-3xl font-bold tracking-tight text-white">
            推理引擎内部之旅
          </h1>
          {/* 架构切换标签页 */}
          <div className="flex rounded-xl border border-slate-700 bg-slate-900/70 p-1">
            <button
              onClick={() => switchArch('moe')}
              className={`arch-tab ${arch === 'moe' ? 'arch-tab-on-moe' : 'arch-tab-off'}`}>
              🧩 MoE 混合专家
            </button>
            <button
              onClick={() => switchArch('dense')}
              className={`arch-tab ${arch === 'dense' ? 'arch-tab-on-dense' : 'arch-tab-off'}`}>
              🔩 Dense 稠密模型
            </button>
          </div>
          <Link to="/gpu" className="arch-tab arch-tab-off border border-slate-700 hover:border-cyan-500/50">
            🧊 3D GPU 实验室 →
          </Link>
          <Link to="/" className="text-sm text-slate-500 hover:text-cyan-300 transition-colors">
            🏠 首页
          </Link>
        </div>
        <p className="mt-2 text-sm text-slate-400 max-w-3xl">
          {isMoe
            ? '当前：MoE（Mixture-of-Experts）。同一个 prompt 的旅程，但引擎内部是「路由 → 选 Top-2 专家 → 加权 → 采样」的稀疏激活循环。切到 Dense 标签可逐步对比差异。'
            : '当前：Dense 稠密模型（如 LLaMA、Qwen 的稠密版、GPT-3）。引擎内部没有路由器，每个 token 都经过同一个 FFN、动用全部参数。切回 MoE 标签逐步对比。'}
        </p>
      </header>

      {/* 控制条 */}
      <div className="max-w-[1360px] mx-auto px-5 flex flex-wrap items-center gap-3 pb-4">
        <button className="ctl-btn bg-cyan-500/90 hover:bg-cyan-400 text-slate-950 font-semibold" onClick={() => setPlaying((p) => !p)}>
          {playing ? '⏸ 暂停' : '▶ 播放'}
        </button>
        <button className="ctl-btn bg-slate-800 hover:bg-slate-700 border border-slate-700" onClick={stepOnce}>⏭ 单步</button>
        <button className="ctl-btn bg-slate-800 hover:bg-slate-700 border border-slate-700" onClick={reset}>⟲ 重置</button>
        <label className="flex items-center gap-2 text-xs text-slate-400 ml-2">
          速度
          <input type="range" min={0.5} max={2} step={0.25} value={speed}
            onChange={(e) => setSpeed(parseFloat(e.target.value))} className="w-28 accent-cyan-400" />
          <span className="text-cyan-300 w-9">{speed}×</span>
        </label>
        <div className="flex items-center gap-1.5 ml-auto flex-wrap">
          {STEPS.map((_, i) => (
            <button key={i} title={STEPS[i].title}
              onClick={() => jumpTo(i)}
              className={`step-dot ${i === step ? 'dot-on' : i < step ? 'dot-past' : ''}`} />
          ))}
        </div>
      </div>

      {/* 画布 */}
      <div ref={wrapRef} className="max-w-[1360px] mx-auto px-5">
        <div style={{ height: H * scale }} className="relative w-full">
          <div style={{ width: W, height: H, transform: `scale(${scale})`, transformOrigin: 'top left' }} className="relative">

            {/* ---- 连线（SVG 底层） ---- */}
            <svg width={W} height={H} className="absolute inset-0">
              <defs>
                <marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
                  <path d="M 0 0 L 10 5 L 0 10 z" fill="#3b82a6" />
                </marker>
                <marker id="arrowHot" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
                  <path d="M 0 0 L 10 5 L 0 10 z" fill="#22d3ee" />
                </marker>
              </defs>
              {/* 请求链路 */}
              <Wire d="M 210 130 L 250 110" on={step >= 1} />
              <Wire d="M 410 110 L 456 110" on={step >= 2} />
              <Wire d="M 632 110 L 676 110" on={step >= 3} />
              <Wire d="M 872 110 C 890 130 892 180 896 225" on={step >= 4} />
              {/* 返回链路 */}
              <Wire d="M 1080 612 L 1080 650 L 952 650" on={step >= 11} />
              <Wire d="M 768 650 L 708 650" on={step >= 12} />
              <Wire d="M 542 650 L 434 650" on={step >= 13} />
              {/* decode 循环箭头 */}
              <path
                d="M 900 470 C 878 470 866 458 866 436 L 866 282 C 866 258 878 246 900 246"
                fill="none" stroke={step === 10 ? '#22d3ee' : '#22354f'} strokeWidth="2.5"
                markerEnd={`url(#${step === 10 ? 'arrowHot' : 'arrow'})`}
                className={step === 10 ? 'wire-dash-hot' : ''}
              />
            </svg>

            {/* 循环标注 */}
            <div className="absolute text-[11px] leading-tight text-slate-500"
              style={{ left: 796, top: 340, width: 66, textAlign: 'center' }}>
              <span className={step === 10 ? 'text-cyan-300 font-semibold' : ''}>decode<br />循环 ×N</span>
            </div>

            {/* ---- 节点：客户端 ---- */}
            <Node x={110} y={130} w={200} active={step === 1 || step >= 14} dim={step > 1 && step < 14}>
              <div className="text-[10px] uppercase tracking-wider text-slate-500 mb-1">你的设备 · 客户端</div>
              <div className="text-[13px] bg-slate-800/80 rounded px-2 py-1.5 text-slate-200">
                法国的首都是哪里？
              </div>
              <div className={`mt-1.5 text-[11px] inline-block px-2 py-0.5 rounded-full ${step === 1 ? 'bg-cyan-500 text-slate-950 font-bold' : 'bg-slate-800 text-slate-400'}`}>
                {step >= 14 ? '✅ 已收到回答' : '⏎ 发送'}
              </div>
            </Node>

            {/* ---- 节点：网络（请求） ---- */}
            <Node x={330} y={110} w={158} active={step === 2} dim={step < 2 || step > 3}>
              <div className="text-[10px] uppercase tracking-wider text-slate-500 mb-1">互联网</div>
              <div className="text-[13px]">📡 HTTP POST</div>
              <div className="text-[11px] text-slate-400">/v1/chat/completions</div>
              {step === 2 && <div className="mt-1 text-[11px] text-cyan-300 animate-pulse">数据包传输中…</div>}
            </Node>

            {/* ---- 节点：API 服务器 ---- */}
            <Node x={545} y={110} w={174} active={step === 3} dim={step < 3 || step > 4}>
              <div className="text-[10px] uppercase tracking-wider text-slate-500 mb-1">API 服务器</div>
              <div className="text-[13px]">🚦 网关 · 队列</div>
              <div className="text-[11px] text-slate-400">鉴权 → 限流 → 调度 GPU</div>
              {step === 3 && (
                <div className="mt-1 flex gap-1">
                  {[0, 1, 2].map((i) => (
                    <span key={i} className="q-block" style={{ animationDelay: `${i * 0.25}s` }} />
                  ))}
                </div>
              )}
            </Node>

            {/* ---- 节点：Tokenizer ---- */}
            <Node x={775} y={110} w={192} active={step === 4} dim={step < 4 || step > 5}>
              <div className="text-[10px] uppercase tracking-wider text-slate-500 mb-1">Tokenizer 分词器</div>
              <div className="flex flex-wrap gap-1 justify-center mt-1">
                {['法国', '的', '首都', '是', '哪里', '?'].map((t, i) => (
                  <span key={i} className={`tok-chip ${step === 4 ? 'tok-in' : ''}`} style={{ animationDelay: `${i * 0.18}s` }}>
                    {t}
                  </span>
                ))}
              </div>
              <div className="text-[11px] text-slate-400 mt-1">文本 → token id 序列</div>
            </Node>

            {/* ================= 推理引擎核心 ================= */}
            <div className={`moe-node engine-box ${step >= 5 && step <= 11 ? 'node-active' : ''}`}
              style={{ left: 1080, top: 375, width: 372 }}>
              <div className="flex items-center justify-between mb-1">
                <div className="text-[10px] uppercase tracking-wider text-slate-500">
                  推理引擎 · {isMoe ? 'MoE Transformer' : 'Dense Transformer'}
                </div>
                <div className={`text-[10px] px-2 py-0.5 rounded-full border font-semibold ${isMoe
                  ? 'bg-amber-400/15 text-amber-300 border-amber-400/40'
                  : 'bg-sky-400/15 text-sky-300 border-sky-400/40'}`}>
                  {isMoe ? '每 token 激活 2/8 专家' : '每 token 激活 100% 参数'}
                </div>
              </div>
              {inDecode && (
                <div className="text-[10px] text-slate-500 mb-1">decode 第 {decodeRound + 1}/3 步</div>
              )}

              {/* 输入 token */}
              <div className={`engine-row ${step >= 6 ? 'row-on' : ''}`}>
                <span className="text-[11px] text-slate-400">输入 token：</span>
                <span className="tok-chip">「{curToken}」</span>
                {step >= 9 && <span className="text-[11px] text-slate-500 mx-1">→</span>}
                {step >= 9 && <span className="tok-chip tok-gold">「{outToken}」</span>}
              </div>

              {/* 注意力层 */}
              <div className={`engine-box2 ${step === 5 ? 'attn-flash' : step > 5 ? 'bg-slate-800/60' : ''}`}>
                <div className="text-[11px] font-semibold text-slate-300">🧠 共享层：Attention（Q·K·V，全序列交互）</div>
                <div className="text-[10px] text-slate-500">每 token 一个隐状态 x · 逐步建立 KV Cache</div>
              </div>

              {isMoe ? (
                <>
                  {/* Router（MoE 专属） */}
                  <div className={`engine-box2 ${step === 6 ? 'router-on' : ''}`}>
                    <div className="text-[11px] font-semibold text-slate-300 mb-1">
                      🎛️ 门控网络 Router <span className="text-slate-500 font-normal">G(x)=softmax(Wr·x) · 取 Top-2</span>
                    </div>
                    <div className="flex gap-[5px] h-14">
                      {ROUTER_SCORES.map((s, i) => (
                        <div key={i} className="flex-1 h-full flex flex-col justify-end items-center gap-0.5">
                          <div className={`w-full router-bar ${step >= 6 ? 'bar-grow' : ''} ${TOPK.includes(i) ? 'bar-hot' : 'bar-cold'}`}
                            style={{ height: `${s * 100}%`, transitionDelay: `${i * 60}ms` }} />
                          <span className={`text-[9px] ${TOPK.includes(i) && step >= 6 ? 'text-amber-300 font-bold' : 'text-slate-500'}`}>
                            E{i + 1}
                          </span>
                        </div>
                      ))}
                    </div>
                  </div>

                  {/* Experts（MoE 专属） */}
                  <div className={`engine-box2 ${step === 7 ? 'router-on' : ''}`}>
                    <div className="text-[11px] font-semibold text-slate-300 mb-1">
                      🧩 专家层 <span className="text-slate-500 font-normal">每个 Expert = 独立 FFN 子网络（↑投影→激活→↓投影）</span>
                    </div>
                    <div className="grid grid-cols-4 gap-1.5">
                      {ROUTER_SCORES.map((_, i) => (
                        <div key={i}
                          className={`expert-chip ${TOPK.includes(i) ? (step >= 7 ? 'expert-on' : 'expert-standby') : step >= 7 ? 'expert-off' : ''}`}>
                          <div className="text-[10px] font-bold">E{i + 1}</div>
                          <div className="text-[8px] opacity-70">FFN</div>
                          {TOPK.includes(i) && step >= 7 && <div className="text-[8px] text-amber-200 font-semibold">{i === 2 ? 'w=0.52' : 'w=0.48'}</div>}
                        </div>
                      ))}
                    </div>
                  </div>
                </>
              ) : (
                <>
                  {/* Dense：无路由直通 */}
                  <div className={`engine-box2 ${step === 6 ? 'dense-on' : ''}`}>
                    <div className="text-[11px] font-semibold text-slate-300">
                      🛤️ 无路由、无选择 <span className="text-slate-500 font-normal">隐状态 x 直通唯一的 FFN，每个 token 路径相同</span>
                    </div>
                  </div>

                  {/* Dense：单一 FFN 全量计算 */}
                  <div className={`engine-box2 ${step === 7 ? 'router-on' : ''}`}>
                    <div className="text-[11px] font-semibold text-slate-300 mb-1">
                      🔩 稠密 FFN <span className="text-slate-500 font-normal">up-proj → 激活 → down-proj</span>
                    </div>
                    <div className={`dense-ffn ${step >= 7 ? 'dense-ffn-on' : ''}`}>
                      <div className="text-[11px] font-bold">FFN（全量参数参与计算）</div>
                      <div className="text-[9px] opacity-80">每个 token · 每一层 · 永远全员上班</div>
                    </div>
                  </div>
                </>
              )}

              {/* 求和 / 输出 + LM Head */}
              <div className={`engine-box2 ${step === 8 ? 'router-on' : ''}`}>
                <div className="text-[11px] font-semibold text-slate-300">
                  {isMoe
                    ? (step >= 8 ? '⚖️ y = 0.52·E₃(x) + 0.48·E₆(x)' : '⚖️ Σ 按门控权重加权求和')
                    : (step >= 8 ? '🔗 残差连接 + LayerNorm → LM Head' : '🔗 FFN 输出直接送入 LM Head')}
                </div>
                <div className="mt-1">
                  <div className="text-[10px] text-slate-500 mb-0.5">LM Head → 词表概率 Top-4</div>
                  {[['巴黎', 0.62], ['是', 0.15], ['法', 0.09], ['…', 0.14]].map(([t, p], i) => (
                    <div key={i} className="flex items-center gap-1.5">
                      <span className="text-[9px] w-6 text-slate-400">{t}</span>
                      <div className="flex-1 h-2 bg-slate-800 rounded overflow-hidden">
                        <div className={`prob-bar ${step >= 8 ? 'bar-grow' : ''} ${i === 0 ? 'prob-hot' : 'prob-cold'}`}
                          style={{ width: `${(p as number) * 100}%`, transitionDelay: `${i * 90}ms` }} />
                      </div>
                      <span className="text-[9px] text-slate-500 w-7">{p}</span>
                    </div>
                  ))}
                </div>
              </div>

              {/* 采样 */}
              <div className={`engine-row justify-between ${step >= 9 ? 'row-on' : ''}`}>
                <span className="text-[11px] text-slate-400">🎲 采样（temperature / argmax）</span>
                <span className={`text-[11px] font-bold ${step >= 9 ? 'text-emerald-300' : 'text-slate-500'}`}>
                  {step >= 9 ? `→「${outToken}」 ✓ 拼回上下文` : '等待分布…'}
                </span>
              </div>

              {/* 上下文条 */}
              <div className="engine-row flex-wrap !items-start">
                <span className="text-[10px] text-slate-500 mr-1 mt-0.5">上下文:</span>
                {['法国', '的', '首都', '是', '哪里', '?'].map((t, i) => (
                  <span key={i} className="tok-chip !text-[9px]">{t}</span>
                ))}
                {genTokens.map((t, i) => (
                  <span key={`g${i}`} className="tok-chip tok-gold !text-[9px] tok-in">{t}</span>
                ))}
                {inDecode && <span className="tok-chip tok-gold !text-[9px] opacity-60">▌</span>}
              </div>
            </div>

            {/* ---- 节点：Detokenizer ---- */}
            <Node x={860} y={650} w={182} active={step === 12} dim={step < 12}>
              <div className="text-[10px] uppercase tracking-wider text-slate-500 mb-1">Detokenizer</div>
              <div className="text-[13px]">🔤 token → 文本</div>
              <div className="text-[11px] text-slate-400">还原字符串 + 后处理</div>
            </Node>

            {/* ---- 节点：网络（返回） ---- */}
            <Node x={625} y={650} w={160} active={step === 13} dim={step < 13}>
              <div className="text-[10px] uppercase tracking-wider text-slate-500 mb-1">互联网</div>
              <div className="text-[13px]">📨 响应返回</div>
              <div className="text-[11px] text-slate-400">HTTP / SSE 流式</div>
            </Node>

            {/* ---- 节点：收到回答 ---- */}
            <Node x={320} y={650} w={220} active={step >= 14} dim={step < 13}>
              <div className="text-[10px] uppercase tracking-wider text-slate-500 mb-1">客户端渲染</div>
              {step >= 14 ? (
                <div className="answer-pop text-[13px] bg-emerald-500/10 border border-emerald-400/30 rounded px-2 py-1.5 text-emerald-200">
                  法国的首都是巴黎。🇫🇷
                </div>
              ) : (
                <div className="text-[13px] text-slate-500">等待响应<span className="cursor-blink">▌</span></div>
              )}
            </Node>

            {/* ---- 数据包 ---- */}
            {def.packet && (
              <div className="packet" style={{ left: def.packet.x, top: def.packet.y }}>
                <div className="packet-core" />
                <div className="packet-ring" />
              </div>
            )}

            {/* 引擎内部高亮遮罩：预填充提示 */}
            {step === 5 && (
              <div className="absolute px-3 py-1.5 rounded-lg bg-cyan-500/10 border border-cyan-400/40 text-cyan-200 text-[11px]"
                style={{ left: 906, top: 128, width: 360 }}>
                {isMoe
                  ? '⚡ Prefill：全部 prompt token 并行通过各层，每层 MoE 各自选专家 —— 只需一次前向'
                  : '⚡ Prefill：全部 prompt token 并行通过各层，每层 FFN 全量计算 —— 只需一次前向'}
              </div>
            )}
          </div>
        </div>
      </div>

      {/* 当前步骤解说 */}
      <div className="max-w-[1360px] mx-auto px-5 mt-4">
        <div className="rounded-xl border border-slate-800 bg-slate-900/60 px-5 py-4">
          <div className="flex items-center gap-3">
            <span className="text-[11px] font-mono px-2 py-0.5 rounded bg-slate-800 text-cyan-300 border border-slate-700">
              {step}/{STEPS.length - 1}
            </span>
            <h2 className="text-base font-bold text-white">{def.title}</h2>
          </div>
          <p className="mt-2 text-sm text-slate-300 leading-relaxed">{def.desc}</p>
        </div>
      </div>

      {/* 知识卡片 */}
      <div className="max-w-[1360px] mx-auto px-5 mt-6 pb-14 grid md:grid-cols-2 xl:grid-cols-4 gap-4">
        <div className="info-card">
          <h3 className="info-title">{isMoe ? '🏗️ MoE 改了 Transformer 的哪里？' : '🏗️ Dense 的 Transformer 长什么样？'}</h3>
          <p>{isMoe
            ? <>把每个 block 里原本的 <b>一个 FFN</b> 换成 <b>N 个并行的 Expert FFN</b>，前面加一个小的门控网络（Router）。代表模型：Switch Transformer、Mixtral 8x7B / 8x22B、DeepSeek 系列。Attention 部分通常保持不变，全员参与。</>
            : <>每个 block = Attention + <b>一个 FFN</b>，逐层堆叠。每个 token 都必须经过每一层的全部参数——故名「稠密」。代表模型：LLaMA、Qwen 稠密版、GPT-3、OPT。结构最简单，行为最可预测。</>}</p>
        </div>
        <div className="info-card">
          <h3 className="info-title">{isMoe ? '🎛️ Router 如何选专家？' : '🛤️ Dense 没有路由器'}</h3>
          <p>{isMoe
            ? <>对隐状态 x 算分数 G(x)=softmax(W<sub>r</sub>·x)，取 <b>Top-K</b>（Mixtral K=2，Switch K=1）。被选专家的输出按各自门控权重加权求和：y = Σ g<sub>i</sub>(x)·E<sub>i</sub>(x)。选择是<b>逐 token</b> 进行的——同一句话里不同 token 可以走不同专家。</>
            : <>没有门控、没有 Top-K：每个 token 的隐状态直接走<b>同一条计算路径</b>。代价是没有任何「用显存换算力」的余地——模型多大，每步计算就多贵。好处是延迟稳定、易于推理优化（如量化、张量并行），也不会出现路由不均。</>}</p>
        </div>
        <div className="info-card">
          <h3 className="info-title">{isMoe ? '⚡ 为什么又强又省？' : '💰 Dense 的账怎么算？'}</h3>
          <p>{isMoe
            ? <>8 个专家每次只激活 2 个：每个 token 的计算量 ≈ dense 模型的 2/8，但模型拥有的<b>总参数量</b>是 8 个专家之和。即「用显存换算力」——全部专家都得装进显存，可配合 expert offloading / 专家并行来缓解。训练时常加辅助损失防止路由坍塌。</>
            : <>每 token 的计算量与<b>总参数量</b>成正比：想变强就堆参数，堆参数就线性变贵。70B 稠密模型每个 decode 步都要搬动全部 70B 参数对应的权重（受显存带宽限制），这正是「大模型推理贵」的根本原因，也是 MoE 出现之前 scaling 的主要瓶颈。</>}</p>
        </div>
        <div className="info-card">
          <h3 className="info-title">⚖️ 一句话对比</h3>
          <p>
            <b>同</b>：Attention、采样、KV Cache、Prefill/Decode 两阶段——旅程完全相同。<br />
            <b>异</b>：Dense 每 token 用 100% 参数、路径固定；MoE 每 token 只用 2/8 参数、路径由 Router 动态决定。<br />
            <b>果</b>：同等计算预算下 MoE 能塞进更多知识；同等显存下 Dense 更省事。
          </p>
        </div>
      </div>
    </div>
  )
}

/** 连线组件：激活时流动发光 */
function Wire({ d, on }: { d: string; on: boolean }) {
  return (
    <path d={d} fill="none" stroke={on ? '#22d3ee' : '#22354f'} strokeWidth="2.5"
      strokeLinecap="round" markerEnd={`url(#${on ? 'arrowHot' : 'arrow'})`}
      className={on ? 'wire-dash-hot' : ''} />
  )
}
