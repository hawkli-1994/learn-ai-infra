import { useEffect, useRef, useState } from 'react'
import { Link } from 'react-router'
import * as THREE from 'three'
import { OrbitControls } from 'three/addons/controls/OrbitControls.js'
import '../App.css'

/* ------------------------------------------------------------------ */
/* 模式定义                                                            */
/* ------------------------------------------------------------------ */

type Mode = 'prefill' | 'decode' | 'nvlink' | 'pcie'

const MODES: { id: Mode; label: string; icon: string }[] = [
  { id: 'prefill', label: 'Prefill 的 GPU', icon: '⚡' },
  { id: 'decode', label: 'Decode 的 GPU', icon: '🐢' },
  { id: 'nvlink', label: 'NVLink 多卡互联', icon: '🔗' },
  { id: 'pcie', label: 'PCIe 多卡互联', icon: '🚌' },
]

const HUD: Record<Mode, { title: string; chips: string[]; desc: string }> = {
  prefill: {
    title: 'Prefill 阶段：算力密集（Compute-bound）',
    chips: ['SM 利用率 ≈ 100%', '显存带宽：中等', '全部 prompt token 并行涌入', 'KV Cache：高速写入'],
    desc: '所有 prompt token 一次性并行进入，矩阵乘法 GEMM 的矩阵很大，SM 阵列全部打满。此时瓶颈是算力，GPU 像一台全力运转的工厂。',
  },
  decode: {
    title: 'Decode 阶段：带宽密集（Memory-bound）',
    chips: ['SM 利用率 < 5%', '显存带宽 ≈ 打满', '每步仅 1 个 token', 'KV Cache：每步全量读取'],
    desc: '每步只处理 1 个 token，GEMM 退化成 GEMV（向量×矩阵），算力需求骤降；但每步都要从显存搬入全部权重和 KV Cache，被显存带宽卡死。量化（权重变小）、投机采样（一步多 token）、Flash/PagedAttention（省 KV 读取）都是为这个瓶颈而生。',
  },
  nvlink: {
    title: 'NVLink 多卡互联：高速公路',
    chips: ['H100 NVLink4：900 GB/s/GPU', '拓扑：经 NVSwitch 全互联', 'GPU 直连，不经 CPU', '延迟极低，专卡专用'],
    desc: '每张卡通过 NVLink 交换机与其他所有 GPU 直接相连，像多车道高速。张量并行的激活值/梯度即时同步，all-reduce 几乎不拖慢训练——DGX/HGX 节点的基石。',
  },
  pcie: {
    title: 'PCIe 多卡互联：乡间小路',
    chips: ['PCIe 5.0 x16 ≈ 64 GB/s', '拓扑：星型，经 PCIe Switch/CPU', '与 CPU 内存/外设共享', '带宽约为 NVLink 的 1/14'],
    desc: '没有 NVLink 时，卡间通信只能走 PCIe 交换机（甚至要经 CPU 内存中转），带宽低一个数量级、延迟更高。消费级显卡组多卡训练/推理时，通信开销会显著吃掉算力红利——这也是跨节点专家并行、DeepSeek 的跨节点 DMA 等优化存在的原因。',
  },
}

/* ------------------------------------------------------------------ */
/* 场景构建工具                                                        */
/* ------------------------------------------------------------------ */

type Updater = (dt: number, t: number) => void

/** GPU 板卡模型：PCB + 计算 die + 4 个 HBM 堆叠 */
function makeBoard(opts: { accent?: number } = {}) {
  const g = new THREE.Group()
  const pcb = new THREE.Mesh(
    new THREE.BoxGeometry(6, 0.18, 4),
    new THREE.MeshStandardMaterial({ color: 0x111c2e, roughness: 0.85 })
  )
  g.add(pcb)

  const dieMat = new THREE.MeshStandardMaterial({
    color: 0x2b3a52, metalness: 0.75, roughness: 0.3,
    emissive: opts.accent ?? 0x22d3ee, emissiveIntensity: 0.06,
  })
  const die = new THREE.Mesh(new THREE.BoxGeometry(2.8, 0.26, 2.8), dieMat)
  die.position.y = 0.22
  g.add(die)

  const hbmMats: THREE.MeshStandardMaterial[] = []
  for (const sx of [-1, 1]) {
    for (let i = 0; i < 2; i++) {
      const m = new THREE.MeshStandardMaterial({
        color: 0x4a3a1a, metalness: 0.4, roughness: 0.5,
        emissive: 0xf59e0b, emissiveIntensity: 0.05,
      })
      const hbm = new THREE.Mesh(new THREE.BoxGeometry(0.85, 0.3, 1.5), m)
      hbm.position.set(sx * 2.3, 0.24, -0.85 + i * 1.7)
      g.add(hbm)
      hbmMats.push(m)
    }
  }
  return { group: g, dieMat, hbmMats }
}

/** SM 阵列：返回材质列表用于闪烁动画 */
function makeSMGrid(parent: THREE.Object3D, size = 8) {
  const mats: THREE.MeshStandardMaterial[] = []
  const cell = 2.6 / size
  const geo = new THREE.BoxGeometry(cell * 0.72, 0.1, cell * 0.72)
  for (let x = 0; x < size; x++) {
    for (let z = 0; z < size; z++) {
      const m = new THREE.MeshStandardMaterial({
        color: 0x1a2638, emissive: 0x22d3ee, emissiveIntensity: 0,
      })
      const sm = new THREE.Mesh(geo, m)
      sm.position.set(-1.3 + cell * (x + 0.5), 0.36, -1.3 + cell * (z + 0.5))
      parent.add(sm)
      mats.push(m)
    }
  }
  return mats
}

/** 沿线段流动的粒子 */
function makeFlow(
  scene: THREE.Scene, updaters: Updater[],
  points: THREE.Vector3[], color: number, speed: number, size = 0.13, tubeRadius = 0
) {
  const curve = new THREE.CatmullRomCurve3(points)
  if (tubeRadius > 0) {
    const tube = new THREE.Mesh(
      new THREE.TubeGeometry(curve, 32, tubeRadius, 8, false),
      new THREE.MeshStandardMaterial({
        color, transparent: true, opacity: 0.18, emissive: color, emissiveIntensity: 0.35,
      })
    )
    scene.add(tube)
  }
  const mat = new THREE.MeshBasicMaterial({ color })
  const p = new THREE.Mesh(new THREE.SphereGeometry(size, 10, 10), mat)
  scene.add(p)
  let t = Math.random()
  updaters.push((dt) => {
    t += dt * speed
    if (t > 1) t -= 1
    p.position.copy(curve.getPointAt(t))
  })
  return p
}

/** 拖尾粒子组（同一路径多个粒子） */
function makeFlowGroup(
  scene: THREE.Scene, updaters: Updater[],
  points: THREE.Vector3[], color: number, speed: number, count: number, size = 0.13
) {
  const curve = new THREE.CatmullRomCurve3(points)
  const mat = new THREE.MeshBasicMaterial({ color })
  const geo = new THREE.SphereGeometry(size, 10, 10)
  for (let i = 0; i < count; i++) {
    const p = new THREE.Mesh(geo, mat)
    scene.add(p)
    let t = i / count
    updaters.push((dt) => {
      t += dt * speed
      if (t > 1) t -= 1
      p.position.copy(curve.getPointAt(t))
    })
  }
}

/** 网格地面 */
function makeGrid(scene: THREE.Scene) {
  const grid = new THREE.GridHelper(60, 60, 0x1c2c47, 0x12203a)
  ;(grid.material as THREE.Material).transparent = true
  ;(grid.material as THREE.Material).opacity = 0.5
  scene.add(grid)
}

function addLights(scene: THREE.Scene) {
  scene.add(new THREE.AmbientLight(0xffffff, 0.55))
  const dir = new THREE.DirectionalLight(0xffffff, 1.2)
  dir.position.set(8, 14, 6)
  scene.add(dir)
  const rim = new THREE.DirectionalLight(0x22d3ee, 0.5)
  rim.position.set(-10, 6, -8)
  scene.add(rim)
}

/* ------------------------------------------------------------------ */
/* 各模式场景                                                          */
/* ------------------------------------------------------------------ */

function buildPrefill(scene: THREE.Scene, updaters: Updater[]) {
  makeGrid(scene)
  const board = makeBoard()
  scene.add(board.group)
  const smMats = makeSMGrid(board.group)

  // KV Cache 插槽（右侧一排盒子，循环填充）
  const kvMats: THREE.MeshStandardMaterial[] = []
  for (let i = 0; i < 14; i++) {
    const m = new THREE.MeshStandardMaterial({
      color: 0x14202f, emissive: 0x34d399, emissiveIntensity: 0,
    })
    const slot = new THREE.Mesh(new THREE.BoxGeometry(0.32, 0.5, 0.6), m)
    slot.position.set(4.6, 0.4, -3.2 + i * 0.5)
    scene.add(slot)
    kvMats.push(m)
  }

  // 并行 token 流入：8 条lane
  for (let i = 0; i < 8; i++) {
    const z = -1.75 + i * 0.5
    makeFlowGroup(scene, updaters,
      [new THREE.Vector3(-11, 1.2, z), new THREE.Vector3(-1.6, 0.6, z * 0.7)],
      0x22d3ee, 0.9, 3, 0.14)
  }
  // 写入 KV：die → 插槽
  makeFlowGroup(scene, updaters,
    [new THREE.Vector3(1.6, 0.6, 0), new THREE.Vector3(4.2, 0.6, 0)],
    0x34d399, 0.7, 4, 0.11)

  // SM 全闪（算力打满）+ KV 循环填充
  updaters.push((_dt, t) => {
    for (const m of smMats) {
      m.emissiveIntensity = 0.5 + Math.random() * 1.6
    }
    const filled = Math.floor((t % 8) / 8 * (kvMats.length + 1))
    kvMats.forEach((m, i) => {
      m.emissiveIntensity = i < filled ? 1.2 : 0.02
    })
  })
}

function buildDecode(scene: THREE.Scene, updaters: Updater[]) {
  makeGrid(scene)
  const board = makeBoard({ accent: 0xf59e0b })
  scene.add(board.group)
  const smMats = makeSMGrid(board.group)

  // KV 插槽全亮（已有历史 KV），持续被读
  const kvMats: THREE.MeshStandardMaterial[] = []
  for (let i = 0; i < 14; i++) {
    const m = new THREE.MeshStandardMaterial({
      color: 0x14202f, emissive: 0x34d399, emissiveIntensity: 0.8,
    })
    const slot = new THREE.Mesh(new THREE.BoxGeometry(0.32, 0.5, 0.6), m)
    slot.position.set(4.6, 0.4, -3.2 + i * 0.5)
    scene.add(slot)
    kvMats.push(m)
  }

  // 单个 token 慢慢流入（每步只有一个）
  makeFlow(scene, updaters,
    [new THREE.Vector3(-11, 1.2, 0), new THREE.Vector3(-1.6, 0.6, 0)],
    0x22d3ee, 0.22, 0.2)

  // KV → die 持续读取（带宽打满）：多条流
  for (let i = 0; i < 6; i++) {
    const z = -2.5 + i
    makeFlowGroup(scene, updaters,
      [new THREE.Vector3(4.2, 0.6, z * 0.8), new THREE.Vector3(1.6, 0.6, z * 0.4)],
      0xf59e0b, 0.8, 2, 0.1)
  }

  updaters.push((_dt, t) => {
    // SM 大部分闲置：随机 2-3 个微亮
    for (const m of smMats) m.emissiveIntensity = 0
    for (let k = 0; k < 3; k++) {
      smMats[Math.floor(Math.random() * smMats.length)].emissiveIntensity = 0.9
    }
    // HBM 高亮：显存带宽繁忙
    const pulse = 0.5 + Math.sin(t * 6) * 0.35
    board.hbmMats.forEach((m) => { m.emissiveIntensity = pulse })
    board.dieMat.emissiveIntensity = 0.05
    // KV 读取闪烁
    kvMats.forEach((m) => { m.emissiveIntensity = 0.5 + Math.sin(t * 8) * 0.3 })
  })
}

function buildNvlink(scene: THREE.Scene, updaters: Updater[]) {
  makeGrid(scene)
  const N = 8
  const R = 10
  const centers: THREE.Vector3[] = []
  for (let i = 0; i < N; i++) {
    const a = (i / N) * Math.PI * 2
    centers.push(new THREE.Vector3(Math.cos(a) * R, 0.6, Math.sin(a) * R))
  }
  // 8 张卡围成一圈
  centers.forEach((c, _i) => {
    const b = makeBoard({ accent: 0x22d3ee })
    b.group.position.copy(c)
    b.group.lookAt(0, 0.6, 0)
    scene.add(b.group)
    // 持续发光表示全互联
    b.dieMat.emissiveIntensity = 0.5
  })

  // NVSwitch 中心
  const sw = new THREE.Mesh(
    new THREE.BoxGeometry(2.2, 1.6, 2.2),
    new THREE.MeshStandardMaterial({
      color: 0x173042, emissive: 0x22d3ee, emissiveIntensity: 0.9,
      transparent: true, opacity: 0.92,
    })
  )
  sw.position.y = 1.4
  scene.add(sw)

  // NVLink：每卡到 NVSwitch 的粗管 + 快速粒子（全互联 = 每两两互通）
  for (let i = 0; i < N; i++) {
    const a = centers[i].clone().setY(0.5)
    const hub = new THREE.Vector3(0, 1.2, 0)
    makeFlow(scene, updaters, [a, hub], 0x22d3ee, 0.55, 0.15, 0.09)
    makeFlow(scene, updaters, [hub, a], 0x67e8f9, 0.55, 0.15, 0.09)
  }
  updaters.push((_dt, t) => {
    ;(sw.material as THREE.MeshStandardMaterial).emissiveIntensity = 0.7 + Math.sin(t * 4) * 0.3
  })
}

function buildPcie(scene: THREE.Scene, updaters: Updater[]) {
  makeGrid(scene)
  // 8 张卡：两排各 4 张
  const centers: THREE.Vector3[] = []
  for (const sx of [-1, 1]) {
    for (let i = 0; i < 4; i++) {
      centers.push(new THREE.Vector3(-7.5 + i * 5, 0.6, sx * 7))
    }
  }
  centers.forEach((c) => {
    const b = makeBoard({ accent: 0xf59e0b })
    b.group.position.copy(c)
    scene.add(b.group)
  })

  // PCIe Switch 中心（星型拓扑的枢纽）
  const sw = new THREE.Mesh(
    new THREE.BoxGeometry(2.6, 1.2, 2.6),
    new THREE.MeshStandardMaterial({
      color: 0x3a2c14, emissive: 0xf59e0b, emissiveIntensity: 0.6,
      transparent: true, opacity: 0.92,
    })
  )
  sw.position.y = 1
  scene.add(sw)

  // CPU + 主机内存
  const cpu = new THREE.Mesh(
    new THREE.BoxGeometry(2, 0.5, 2),
    new THREE.MeshStandardMaterial({ color: 0x2a3648, emissive: 0x94a3b8, emissiveIntensity: 0.2 })
  )
  cpu.position.set(0, 0.4, 0)
  scene.add(cpu)
  const mem = new THREE.Mesh(
    new THREE.BoxGeometry(4.5, 0.35, 1),
    new THREE.MeshStandardMaterial({ color: 0x1f2c22, emissive: 0x4ade80, emissiveIntensity: 0.25 })
  )
  mem.position.set(0, 0.35, 3.2)
  scene.add(mem)

  // 每卡 → Switch 的细线，慢速粒子（带宽低），部分流量还要经 CPU 内存中转
  centers.forEach((c, _i) => {
    const a = c.clone().setY(0.5)
    const hub = new THREE.Vector3(0, 0.9, 0)
    makeFlow(scene, updaters, [a, hub], 0xf59e0b, 0.1, 0.1, 0.025)
    makeFlow(scene, updaters, [hub, a], 0xfbbf24, 0.1, 0.1, 0.025)
    // 偶数卡：经主机内存中转（更慢）
    if (_i % 2 === 0) {
      makeFlow(scene, updaters, [hub, new THREE.Vector3(0, 0.5, 3.2)], 0x4ade80, 0.06, 0.09, 0.02)
    }
  })
  updaters.push((_dt, t) => {
    ;(sw.material as THREE.MeshStandardMaterial).emissiveIntensity = 0.4 + Math.sin(t * 2) * 0.25
  })
}

const BUILDERS: Record<Mode, (scene: THREE.Scene, updaters: Updater[]) => void> = {
  prefill: buildPrefill,
  decode: buildDecode,
  nvlink: buildNvlink,
  pcie: buildPcie,
}

const CAM_POS: Record<Mode, [number, number, number]> = {
  prefill: [11, 9, 13],
  decode: [11, 9, 13],
  nvlink: [0, 17, 22],
  pcie: [0, 19, 21],
}

/* ------------------------------------------------------------------ */
/* 3D 画布组件                                                         */
/* ------------------------------------------------------------------ */

function GpuScene({ mode }: { mode: Mode }) {
  const mountRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    const mount = mountRef.current
    if (!mount) return

    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true })
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))
    renderer.setSize(mount.clientWidth, mount.clientHeight)
    mount.appendChild(renderer.domElement)

    const scene = new THREE.Scene()
    scene.fog = new THREE.Fog(0x070d1a, 45, 110)
    const camera = new THREE.PerspectiveCamera(50, mount.clientWidth / mount.clientHeight, 0.1, 300)
    camera.position.set(...CAM_POS[mode])

    const controls = new OrbitControls(camera, renderer.domElement)
    controls.enableDamping = true
    controls.target.set(0, 0.5, 0)
    controls.maxPolarAngle = Math.PI * 0.49

    addLights(scene)
    const updaters: Updater[] = []
    BUILDERS[mode](scene, updaters)

    const clock = new THREE.Clock()
    let raf = 0
    const loop = () => {
      raf = requestAnimationFrame(loop)
      const dt = Math.min(clock.getDelta(), 0.05)
      const t = clock.elapsedTime
      for (const u of updaters) u(dt, t)
      controls.update()
      renderer.render(scene, camera)
    }
    loop()

    const onResize = () => {
      if (!mount) return
      camera.aspect = mount.clientWidth / mount.clientHeight
      camera.updateProjectionMatrix()
      renderer.setSize(mount.clientWidth, mount.clientHeight)
    }
    window.addEventListener('resize', onResize)

    return () => {
      cancelAnimationFrame(raf)
      window.removeEventListener('resize', onResize)
      controls.dispose()
      scene.traverse((obj) => {
        if (obj instanceof THREE.Mesh) {
          obj.geometry.dispose()
          const m = obj.material as THREE.Material | THREE.Material[]
          if (Array.isArray(m)) m.forEach((x) => x.dispose())
          else m.dispose()
        }
      })
      renderer.dispose()
      mount.removeChild(renderer.domElement)
    }
  }, [mode])

  return <div ref={mountRef} className="w-full h-full cursor-grab active:cursor-grabbing" />
}

/* ------------------------------------------------------------------ */
/* 页面                                                                */
/* ------------------------------------------------------------------ */

export default function GpuLab() {
  const [mode, setMode] = useState<Mode>(() => {
    const h = typeof window !== 'undefined' ? window.location.hash.replace('#', '') : ''
    return (MODES.some((m) => m.id === h) ? h : 'prefill') as Mode
  })
  const hud = HUD[mode]
  const selectMode = (m: Mode) => {
    setMode(m)
    window.location.hash = m
  }

  return (
    <div className="min-h-screen bg-[#070d1a] text-slate-200 page-font">
      {/* 头部 */}
      <header className="max-w-[1360px] mx-auto px-5 pt-8 pb-4">
        <div className="flex flex-wrap items-center gap-4">
          <h1 className="text-2xl md:text-3xl font-bold tracking-tight text-white">
            GPU 3D 实验室
          </h1>
          <Link to="/" className="text-sm text-slate-400 hover:text-cyan-300 transition-colors">
            ← 返回 2D 推理流程
          </Link>
        </div>
        <p className="mt-2 text-sm text-slate-400 max-w-3xl">
          用 3D 看推理时 GPU 里发生了什么：Prefill 的算力高峰、Decode 的带宽瓶颈、多卡互联的 NVLink 高速路与 PCIe 羊肠小道。拖动旋转，滚轮缩放。
        </p>
      </header>

      {/* 场景切换 */}
      <div className="max-w-[1360px] mx-auto px-5 flex flex-wrap gap-2 pb-4">
        {MODES.map((m) => (
          <button key={m.id} onClick={() => selectMode(m.id)}
            className={`arch-tab ${mode === m.id ? 'arch-tab-on-moe' : 'arch-tab-off'}`}>
            {m.icon} {m.label}
          </button>
        ))}
      </div>

      {/* 3D 画布 + HUD */}
      <div className="max-w-[1360px] mx-auto px-5">
        <div className="relative rounded-xl border border-slate-800 bg-[#0a1120] overflow-hidden" style={{ height: 560 }}>
          <GpuScene mode={mode} />
          {/* HUD 覆盖层 */}
          <div className="absolute top-4 left-4 right-4 pointer-events-none flex flex-wrap items-start gap-3">
            <div className="rounded-lg bg-slate-900/85 border border-slate-700 px-4 py-3 max-w-md backdrop-blur">
              <h2 className="text-sm font-bold text-white">{hud.title}</h2>
              <p className="mt-1.5 text-xs text-slate-300 leading-relaxed">{hud.desc}</p>
            </div>
            <div className="flex flex-wrap gap-2">
              {hud.chips.map((c, i) => (
                <span key={i} className="text-[11px] px-2.5 py-1 rounded-full bg-slate-900/85 border border-slate-700 text-cyan-200 backdrop-blur">
                  {c}
                </span>
              ))}
            </div>
          </div>
          <div className="absolute bottom-3 right-4 text-[11px] text-slate-500 pointer-events-none">
            🖱️ 拖动旋转 · 滚轮缩放
          </div>
        </div>
      </div>

      {/* 图例 / 说明 */}
      <div className="max-w-[1360px] mx-auto px-5 mt-6 pb-14 grid md:grid-cols-2 gap-4">
        <div className="info-card">
          <h3 className="info-title">🗺️ 场景图例</h3>
          <p>
            <span className="text-cyan-300">●</span> 青色粒子/灯：token 流、计算单元（SM）活跃<br />
            <span className="text-amber-400">●</span> 琥珀色：HBM 显存 / PCIe 流量（带宽密集）<br />
            <span className="text-emerald-400">●</span> 绿色：KV Cache 写入/读取、主机内存中转<br />
            黑色大板 = GPU 板卡：中间方块是计算 die，两侧 4 块是 HBM 显存堆叠。
          </p>
        </div>
        <div className="info-card">
          <h3 className="info-title">⚖️ NVLink vs PCIe 速查</h3>
          <p>
            <b>NVLink 4（H100）</b>：900 GB/s/GPU，GPU 直连全互联，走专用 NVSwitch<br />
            <b>PCIe 5.0 x16</b>：~64 GB/s，星型拓扑经 PCIe Switch，常与 CPU/外设共享<br />
            <b>差距</b>：约 <b>14×</b> 带宽 + 更低延迟。卡间通信密集的张量并行/流水并行，在 PCIe 机器上会明显变慢。
          </p>
        </div>
        <div className="info-card">
          <h3 className="info-title">⚡ 为什么 Prefill 快、Decode 慢？</h3>
          <p>
            Prefill 一次算 N 个 token 的大矩阵，SM 阵列吃满，显存带宽摊薄到每个 token 上很便宜；Decode 每步 1 个 token，矩阵退化为 GEMV，SM 大多空转，却要每步把<b>全部权重 + 历史 KV</b> 从显存搬一遍——所以 decode 的吞吐主要由「显存带宽 ÷ 模型大小」决定，而非算力。
          </p>
        </div>
        <div className="info-card">
          <h3 className="info-title">🛠️ 工程上怎么补 PCIe 的短板？</h3>
          <p>
            ① 减少跨卡通信：数据并行代替张量并行；② 通信与计算重叠（overlap）；③ DeepSeek 的跨节点 DMA / 专家并行把通信摊到训练全程；④ 消费级组集群时优先同机 NVLink 域内做张量并行，跨机走 RDMA 网络。
          </p>
        </div>
      </div>
    </div>
  )
}
