import { useEffect, useRef, useState } from 'react'
import { Link } from 'react-router'
import * as THREE from 'three'
import { OrbitControls } from 'three/addons/controls/OrbitControls.js'
import '../App.css'

/* ------------------------------------------------------------------ */
/* 模式定义                                                            */
/* ------------------------------------------------------------------ */

type Mode = 'vllm' | 'prefill' | 'decode' | 'nvlink' | 'pcie' | 'tp' | 'pp' | 'ep'

const MODES: { id: Mode; label: string; icon: string }[] = [
  { id: 'vllm', label: 'vLLM 引擎全景', icon: '🧠' },
  { id: 'prefill', label: 'Prefill 的 GPU', icon: '⚡' },
  { id: 'decode', label: 'Decode 的 GPU', icon: '🐢' },
  { id: 'nvlink', label: 'NVLink 多卡互联', icon: '🔗' },
  { id: 'pcie', label: 'PCIe 多卡互联', icon: '🚌' },
]

const PARALLEL_MODES: { id: Mode; label: string; icon: string }[] = [
  { id: 'tp', label: '张量并行 TP', icon: '✂️' },
  { id: 'pp', label: '流水线并行 PP', icon: '🏭' },
  { id: 'ep', label: '专家并行 EP', icon: '📦' },
]

const EXPERT_COLORS = [0x22d3ee, 0x34d399, 0xfbbf24, 0xf472b6, 0xa78bfa, 0xfb923c, 0x4ade80, 0x38bdf8]
const MB_COLORS = [0x22d3ee, 0x34d399, 0xfbbf24, 0xf472b6]
const STAGE_X = [-7.5, -2.5, 2.5, 7.5]

const HUD: Record<Mode, { title: string; chips: string[]; desc: string }> = {
  vllm: {
    title: 'vLLM 全景：一次请求的一生',
    chips: ['Continuous Batching：prefill / decode 混在同一个 batch', 'PagedAttention：KV 按 Block（16 token）分页，像 OS 内存', 'Chunked Prefill：长 prompt 切块，不饿死 decode', 'KV 不够时：抢占 / LRU 驱逐，请求回队列'],
    desc: '从左到右跟一遍：请求经 API Server 分词后进入等待队列；Scheduler 做连续批处理——每个调度周期把新请求插入 batch 的空槽（prefill，一次算一串 token），已在跑的序列每步只算 1 个 token（decode），两者共享同一块 GPU。GPU 每算一步都把新 KV 写进 Block 池（PagedAttention，按 16 token 一块分页分配，满了就驱逐最久没用的块）；采样器选出下一个 token 后经弧线送回输入——这就是自回归循环；遇到 EOS 的序列离开 batch、释放块、流式吐回客户端。拖动旋转，可以从上空俯瞰整条流水线。',
  },
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
  tp: {
    title: '张量并行 TP：把一层切开（Megatron-LM）',
    chips: ['切分：同一层 W 切成 4 片（切缝可见）', '同一份 token 广播给 4 卡', '各卡算出自己的 1/4 部分和', '头顶光环 = NVLink all-reduce 环'],
    desc: '上方的半透明矩阵是切分前的完整一层 W，彩色实块是被竖切走的 4 片、各放一张卡。左侧的 token 被广播到每张卡，与本地切片相乘得到 1/4 部分和；4 份部分和沿头顶光环（all-reduce 环）逐卡合并，最终环上只剩一份完整结果，以亮金色 token 输出。通信发生在每一层、每个 token 上，带宽需求极高——TP 几乎是 NVLink 存在的理由，也只活在单机 NVLink 域（通常 ≤8 卡）。',
  },
  pp: {
    title: '流水线并行 PP：把模型分层（GPipe / 1F1B）',
    chips: ['切分：按层分段（stage）', '只传相邻 stage 之间的激活', '通信量：小 · 可跨机扩展', '代价：流水线气泡（bubble）'],
    desc: 'GPU0 算前几层、GPU1 算中间几层……像流水线工厂：4 个颜色的微批次（micro-batch）错位推进，同一时刻不同 GPU 在算不同的批次。气泡 = 流水线填充/排空时的空转时间，GPipe、1F1B、PipeDream 的各种调度都是在压缩气泡。',
  },
  ep: {
    title: '专家并行 EP：把专家搬到不同卡（GShard / DeepSeek）',
    chips: ['切分：MoE 的 Expert 分散放置', 'Router 在每个 GPU 本地运行', 'dispatch：token → 目标专家所在卡', 'combine：结果 → 回 token 来源卡（all-to-all 对称）'],
    desc: '每个 GPU 只放 2 个专家；注意并没有中央 Router——每张卡用本地门控网络给自己的 token 选专家，然后 all-to-all：token 被发往目标专家所在的卡（dispatch），专家算完后结果再回到 token 的来源卡，继续走下一层（combine）。画面中每块卡既发出也接收彩色 token，流量是对称的。专家越多扩展性越好，所以 DeepSeek 用 EP 替换大部分 TP。',
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
  points: THREE.Vector3[], color: number, speed: number, count: number, size = 0.13,
  startOffset = 0
) {
  const curve = new THREE.CatmullRomCurve3(points)
  const mat = new THREE.MeshBasicMaterial({ color })
  const geo = new THREE.SphereGeometry(size, 10, 10)
  for (let i = 0; i < count; i++) {
    const p = new THREE.Mesh(geo, mat)
    scene.add(p)
    let t = (startOffset + i / count) % 1
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

/** 3D 文字标签（Canvas 贴图 Sprite） */
function makeLabel(text: string, color = '#a5f3fc', scale = 1) {
  const font = 'bold 44px "PingFang SC", "Hiragino Sans GB", "Microsoft YaHei", sans-serif'
  const c = document.createElement('canvas')
  const ctx0 = c.getContext('2d')!
  ctx0.font = font
  c.width = Math.ceil(ctx0.measureText(text).width) + 40
  c.height = 72
  const ctx = c.getContext('2d')!
  ctx.font = font
  ctx.fillStyle = color
  ctx.textAlign = 'center'
  ctx.textBaseline = 'middle'
  ctx.fillText(text, c.width / 2, 36)
  const tex = new THREE.CanvasTexture(c)
  const sp = new THREE.Sprite(new THREE.SpriteMaterial({ map: tex, transparent: true, depthWrite: false }))
  const h = 1.0 * scale
  sp.scale.set((c.width / 72) * h, h, 1)
  return sp
}

/** 引擎组件节点盒：发光底座 + 文字标签，返回中心坐标 */
function makeNode(
  scene: THREE.Scene, x: number, z: number, w: number, d: number,
  label: string, accent: number, opts: { labelColor?: string; h?: number; y?: number } = {}
) {
  const y = opts.y ?? 0
  const h = opts.h ?? 1.2
  const box = new THREE.Mesh(
    new THREE.BoxGeometry(w, h, d),
    new THREE.MeshStandardMaterial({ color: 0x101c30, emissive: accent, emissiveIntensity: 0.14 })
  )
  box.position.set(x, y + h / 2, z)
  scene.add(box)
  const edges = new THREE.LineSegments(
    new THREE.EdgesGeometry(box.geometry),
    new THREE.LineBasicMaterial({ color: accent, transparent: true, opacity: 0.65 })
  )
  edges.position.copy(box.position)
  scene.add(edges)
  const sp = makeLabel(label, opts.labelColor ?? '#a5f3fc', 0.85)
  sp.position.set(x, y + h + 0.8, z)
  scene.add(sp)
  return new THREE.Vector3(x, y + h, z)
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

/* ---------- 并行策略场景：TP / PP / EP ---------- */

/** 通用：一排 4 张 GPU，返回板卡对象 */
function makeStageRow(scene: THREE.Scene, accent = 0x22d3ee) {
  const boards = STAGE_X.map((x) => {
    const b = makeBoard({ accent })
    b.group.position.set(x, 0.6, 0)
    scene.add(b.group)
    return b
  })
  return boards
}

function buildTP(scene: THREE.Scene, updaters: Updater[]) {
  makeGrid(scene)
  const boards = makeStageRow(scene)
  const RING_Y = 6.4
  const RING_RX = 11.2
  const RING_RY = 1.7

  // 半透明「完整一层 W」ghost 矩阵（线框轮廓），表示切分前的全貌
  const ghostGeo = new THREE.BoxGeometry(21.5, 2.6, 1.6)
  const ghost = new THREE.Mesh(ghostGeo, new THREE.MeshStandardMaterial({
    color: 0x22d3ee, transparent: true, opacity: 0.09,
    emissive: 0x22d3ee, emissiveIntensity: 0.15, depthWrite: false,
  }))
  ghost.position.set(0, 3.5, 0)
  scene.add(ghost)
  const ghostEdges = new THREE.LineSegments(
    new THREE.EdgesGeometry(ghostGeo),
    new THREE.LineBasicMaterial({ color: 0x22d3ee, transparent: true, opacity: 0.4 })
  )
  ghostEdges.position.copy(ghost.position)
  scene.add(ghostEdges)

  // 4 片实体切片（切缝清晰可见），4 色各归一张卡
  const slabTops: THREE.Vector3[] = []
  STAGE_X.forEach((x, i) => {
    const color = MB_COLORS[i]
    const slab = new THREE.Mesh(
      new THREE.BoxGeometry(4.0, 2.6, 1.6),
      new THREE.MeshStandardMaterial({
        color: 0x131f33, emissive: color, emissiveIntensity: 0.45,
        transparent: true, opacity: 0.92,
      })
    )
    slab.position.set(x, 3.5, 0)
    scene.add(slab)
    slabTops.push(new THREE.Vector3(x, 4.85, 0))
  })

  // 输入 token：左侧一个发光白方块，广播复制给 4 张卡
  const token = new THREE.Mesh(
    new THREE.BoxGeometry(0.6, 0.6, 0.6),
    new THREE.MeshStandardMaterial({ color: 0xffffff, emissive: 0xffffff, emissiveIntensity: 1.1 })
  )
  token.position.set(-12.5, 1.0, 0)
  scene.add(token)
  updaters.push((_dt, t) => {
    token.rotation.y = t * 1.5
    token.position.y = 1.0 + Math.sin(t * 2) * 0.12
  })
  for (const x of STAGE_X) {
    makeFlowGroup(scene, updaters,
      [new THREE.Vector3(-12.1, 1.0, 0), new THREE.Vector3(x - 1.6, 0.7, 0)],
      0xffffff, 0.5, 2, 0.12)
  }

  // 部分积：各卡与本地切片相乘，切片颜色的部分和从 die 升进切片
  STAGE_X.forEach((x, i) => {
    makeFlowGroup(scene, updaters,
      [new THREE.Vector3(x, 0.9, 0), new THREE.Vector3(x, 2.1, 0)],
      MB_COLORS[i], 0.6, 2, 0.11)
  })

  // all-reduce 环：悬在切片上方的光环（NVLink），双向高速粒子对向环流
  const ringEllipse = new THREE.EllipseCurve(0, 0, RING_RX, RING_RY, 0, Math.PI * 2)
  const ringPts = ringEllipse.getPoints(80).map(p => new THREE.Vector3(p.x, RING_Y + p.y, 0))
  const ringCurve = new THREE.CatmullRomCurve3(ringPts, true)
  scene.add(new THREE.Mesh(
    new THREE.TubeGeometry(ringCurve, 96, 0.05, 8, true),
    new THREE.MeshStandardMaterial({ color: 0x67e8f9, transparent: true, opacity: 0.25, emissive: 0x67e8f9, emissiveIntensity: 0.5 })
  ))
  const ringMat = new THREE.MeshBasicMaterial({ color: 0xa5f3fc })
  const ringGeo = new THREE.SphereGeometry(0.12, 10, 10)
  for (let k = 0; k < 8; k++) {
    const p = new THREE.Mesh(ringGeo, ringMat)
    scene.add(p)
    let t = k / 8
    const dir = k % 2 === 0 ? 1 : -1
    updaters.push((dt) => {
      t = (t + dt * 0.35 * dir + 1) % 1
      p.position.copy(ringCurve.getPointAt(t))
    })
  }

  // 4 条支线：各切片的部分和注入环
  slabTops.forEach((top, i) => {
    const s = Math.sqrt(Math.max(0, 1 - (STAGE_X[i] / RING_RX) ** 2))
    const onRing = new THREE.Vector3(STAGE_X[i], RING_Y + RING_RY * s, 0)
    makeFlowGroup(scene, updaters, [top, onRing], MB_COLORS[i], 0.5, 2, 0.1)
  })

  // 合并输出：环上 all-reduce 完成的完整结果 = 单一亮金色 token 向右离开
  makeFlowGroup(scene, updaters,
    [new THREE.Vector3(RING_RX, RING_Y, 0), new THREE.Vector3(13.4, 3.4, 0), new THREE.Vector3(13.4, 1.2, 0)],
    0xffe9a8, 0.4, 2, 0.18)

  // 各卡 die 按切片颜色脉动（计算活跃）
  updaters.push((_dt, t) => {
    boards.forEach((b, i) => {
      b.dieMat.emissive = new THREE.Color(MB_COLORS[i])
      b.dieMat.emissiveIntensity = 0.08 + Math.max(0, Math.sin(t * 3 + i * 1.2)) * 0.25
    })
  })
}

function buildPP(scene: THREE.Scene, updaters: Updater[]) {
  makeGrid(scene)
  makeStageRow(scene, 0x34d399)

  // 每卡上方一叠「层」slab，表示各 stage 负责不同的层
  STAGE_X.forEach((x) => {
    for (let i = 0; i < 4; i++) {
      const slab = new THREE.Mesh(
        new THREE.BoxGeometry(3.2, 0.16, 2.4),
        new THREE.MeshStandardMaterial({
          color: 0x16283a, emissive: 0x34d399, emissiveIntensity: 0.12 + i * 0.08,
        })
      )
      slab.position.set(x, 0.85 + i * 0.26, 0)
      scene.add(slab)
    }
  })

  // 4 个微批次错位推进（相位错开 = 流水线）
  for (let mb = 0; mb < 4; mb++) {
    const z = -1.5 + mb
    const pts = [
      new THREE.Vector3(-13, 1.4, z),
      ...STAGE_X.map((x) => new THREE.Vector3(x, 1.4, z)),
      new THREE.Vector3(13, 1.4, z),
    ]
    makeFlowGroup(scene, updaters, pts, MB_COLORS[mb], 0.14, 2, 0.15, mb * 0.25)
  }

  // stage 间激活传递的连线
  for (let i = 0; i < 3; i++) {
    const a = new THREE.Vector3(STAGE_X[i] + 3, 1.4, 0)
    const b = new THREE.Vector3(STAGE_X[i + 1] - 3, 1.4, 0)
    makeFlow(scene, updaters, [a, b], 0x34d399, 0.14, 0.13, 0.035)
  }
}

function buildEP(scene: THREE.Scene, updaters: Updater[]) {
  makeGrid(scene)
  const boards = makeStageRow(scene, 0xf59e0b)

  // 每卡：后排 2 个本地专家（颜色各异）+ 前排「本地 token 队列」发光板
  const expertPos: THREE.Vector3[][] = []
  const homePos: THREE.Vector3[] = []
  STAGE_X.forEach((x, gi) => {
    const experts: THREE.Vector3[] = []
    for (let e = 0; e < 2; e++) {
      const idx = gi * 2 + e
      const color = EXPERT_COLORS[idx]
      const chip = new THREE.Mesh(
        new THREE.BoxGeometry(1.4, 0.6, 1.2),
        new THREE.MeshStandardMaterial({ color: 0x1c2436, emissive: color, emissiveIntensity: 0.55 })
      )
      const pos = new THREE.Vector3(x - 0.9 + e * 1.8, 1.3, -1.2)
      chip.position.copy(pos)
      scene.add(chip)
      experts.push(pos.clone().setY(1.4))
    }
    expertPos.push(experts)

    const home = new THREE.Vector3(x, 0.7, 2.6)
    const plate = new THREE.Mesh(
      new THREE.BoxGeometry(2.4, 0.16, 1.3),
      new THREE.MeshStandardMaterial({ color: 0x241d10, emissive: 0xfbbf24, emissiveIntensity: 0.35 })
    )
    plate.position.copy(home)
    scene.add(plate)
    homePos.push(home.clone().setY(0.95))
    boards[gi].dieMat.emissiveIntensity = 0.1
  })

  // all-to-all fabric：所有卡两两之间的淡线（对称点对点）
  for (let i = 0; i < 4; i++) {
    for (let j = i + 1; j < 4; j++) {
      const line = new THREE.Mesh(
        new THREE.TubeGeometry(
          new THREE.CatmullRomCurve3([homePos[i], homePos[j]]), 8, 0.02, 6, false
        ),
        new THREE.MeshBasicMaterial({ color: 0x3a4a63, transparent: true, opacity: 0.5 })
      )
      scene.add(line)
    }
  }

  // dispatch + combine：每卡 token 经本地 Router 决策，沿弧线发往目标专家，再返回来源卡
  // 每卡画 3 条出向流（自卡专家 + 左右两邻卡专家），往返共用弧线、相位错开
  for (let i = 0; i < 4; i++) {
    const targets = [i, (i + 1) % 4, (i + 3) % 4]
    targets.forEach((j, k) => {
      const e = k % 2
      const color = EXPERT_COLORS[j * 2 + e]
      const from = homePos[i]
      const to = expertPos[j][e].clone().setY(1.5)
      const mid = from.clone().lerp(to, 0.5).setY(4.4 + Math.abs(i - j) * 0.7)
      makeFlowGroup(scene, updaters, [from, mid, to], color, 0.3, 2, 0.11)
      // combine：结果沿同弧线回到 token 来源卡（错半相，形成往返）
      makeFlowGroup(scene, updaters, [to, mid, from], color, 0.3, 2, 0.11, 0.5)
    })
  }

  // 本地 Router 脉冲（每卡 die 代表本地门控决策）
  updaters.push((_dt, t) => {
    boards.forEach((b, i) => {
      b.dieMat.emissive = new THREE.Color(0xfbbf24)
      b.dieMat.emissiveIntensity = 0.08 + Math.max(0, Math.sin(t * 2.5 + i)) * 0.2
    })
  })
}

/* ---------- vLLM 引擎全景 ---------- */

function buildVllm(scene: THREE.Scene, updaters: Updater[]) {
  makeGrid(scene)
  const CYAN = 0x22d3ee, GREEN = 0x34d399, PINK = 0xf472b6, GOLD = 0xffe9a8

  // ① 客户端 → ② API Server / Tokenizer → ③ 等待队列 → ④ Scheduler
  makeNode(scene, -19, 0, 2, 1.6, 'Client', 0x94a3b8)
  makeNode(scene, -15.5, 0, 3, 2, 'API Server · Tokenizer', CYAN)
  makeNode(scene, -8.5, 0, 3, 2, 'Scheduler · 连续批处理', CYAN)

  const qLabel = makeLabel('Waiting Queue', '#94a3b8', 0.7)
  qLabel.position.set(-12, 2.6, 0)
  scene.add(qLabel)

  // 请求流：Client → API Server（3 个颜色 = 3 个在途请求）
  MB_COLORS.forEach((c, i) => {
    makeFlowGroup(scene, updaters,
      [new THREE.Vector3(-17.9, 1.1, -0.5 + i * 0.5), new THREE.Vector3(-16, 1.1, -0.3 + i * 0.3)],
      c, 0.35 + i * 0.04, 1, 0.14)
    // 分词后的 token ids → 等待队列
    makeFlowGroup(scene, updaters,
      [new THREE.Vector3(-13.9, 1.0, -0.3 + i * 0.3), new THREE.Vector3(-12.6, 1.0, -0.4 + i * 0.4)],
      0xffffff, 0.3 + i * 0.03, 2, 0.09)
    // 队列 → Scheduler
    makeFlowGroup(scene, updaters,
      [new THREE.Vector3(-11.4, 1.0, -0.4 + i * 0.4), new THREE.Vector3(-10.1, 1.0, 0)],
      c, 0.16, 1, 0.13)
  })

  // ④ Scheduler 头顶的 batch 槽位架：6 槽循环演示 空→prefill(闪)→decode(稳)→完成
  const slotMats: THREE.MeshStandardMaterial[] = []
  for (let i = 0; i < 6; i++) {
    const x = -9.9 + i * 0.56
    const frame = new THREE.LineSegments(
      new THREE.EdgesGeometry(new THREE.BoxGeometry(0.48, 0.6, 0.6)),
      new THREE.LineBasicMaterial({ color: 0x3a4a63 })
    )
    frame.position.set(x, 2.5, 0)
    scene.add(frame)
    const m = new THREE.MeshStandardMaterial({ color: 0x131f33, emissive: CYAN, emissiveIntensity: 0.03 })
    const cell = new THREE.Mesh(new THREE.BoxGeometry(0.4, 0.5, 0.5), m)
    cell.position.set(x, 2.5, 0)
    scene.add(cell)
    slotMats.push(m)
  }
  const slotLabel = makeLabel('batch 槽位：闪 = 新请求 prefill 插入', '#67e8f9', 0.55)
  slotLabel.position.set(-8.5, 3.6, 0)
  scene.add(slotLabel)

  // ⑤ GPU 大板 + SM 阵列
  const gpu = makeBoard()
  gpu.group.position.set(0, 0.25, 0)
  gpu.group.scale.setScalar(1.4)
  scene.add(gpu.group)
  const smMats = makeSMGrid(gpu.group, 10)
  const gpuLabel = makeLabel('GPU：Prefill / Decode 混跑', '#a5f3fc', 0.9)
  gpuLabel.position.set(0, 3.6, 0)
  scene.add(gpuLabel)

  // Prefill 车道（上，z=-3）：一次一串 token；Decode 车道（下，z=+3）：每步 1 token
  makeFlowGroup(scene, updaters,
    [new THREE.Vector3(-7, 1.0, -3), new THREE.Vector3(-3.4, 0.9, -1.8)],
    CYAN, 0.85, 4, 0.13)
  MB_COLORS.forEach((c, i) => {
    makeFlowGroup(scene, updaters,
      [new THREE.Vector3(-7, 1.0, 3), new THREE.Vector3(-3.4, 0.9, 1.8)],
      c, 0.3, 1, 0.12, i * 0.33)
  })
  const pfLabel = makeLabel('Prefill：一次一串 token', '#67e8f9', 0.55)
  pfLabel.position.set(-5.2, 2.3, -3)
  scene.add(pfLabel)
  const dcLabel = makeLabel('Decode：每步 1 token', '#fcd34d', 0.55)
  dcLabel.position.set(-5.2, 2.3, 3)
  scene.add(dcLabel)

  // ⑥ KV Block 池（PagedAttention：分页分配 / LRU 驱逐）
  const kvLabel = makeLabel('KV Cache · Block 池（16 token/块）', '#6ee7b7', 0.75)
  kvLabel.position.set(7, 2.9, -3)
  scene.add(kvLabel)
  const kvMats: THREE.MeshStandardMaterial[] = []
  for (let r = 0; r < 3; r++) {
    for (let c = 0; c < 4; c++) {
      const m = new THREE.MeshStandardMaterial({ color: 0x101c2a, emissive: GREEN, emissiveIntensity: 0.03 })
      const blk = new THREE.Mesh(new THREE.BoxGeometry(0.55, 0.55, 0.55), m)
      blk.position.set(6.1 + c * 0.72, 0.6 + r * 0.72, -3)
      scene.add(blk)
      kvMats.push(m)
    }
  }
  // KV 写入（GPU → 池）与读取（池 → GPU，decode 每步全量读）
  makeFlowGroup(scene, updaters,
    [new THREE.Vector3(3.4, 0.9, -1.6), new THREE.Vector3(5.6, 1.0, -2.6)],
    GREEN, 0.6, 3, 0.1)
  makeFlowGroup(scene, updaters,
    [new THREE.Vector3(5.6, 0.7, -3.3), new THREE.Vector3(3.4, 0.7, -2.0)],
    0x6ee7b7, 0.4, 2, 0.09)

  // ⑦ 采样器
  makeNode(scene, 7, 3, 2.2, 1.8, '采样 Sampling', PINK, { labelColor: '#f9a8d4' })
  makeFlowGroup(scene, updaters,
    [new THREE.Vector3(3.4, 0.9, 1.8), new THREE.Vector3(5.7, 1.1, 2.9)],
    PINK, 0.4, 3, 0.1)
  const pick = new THREE.Mesh(
    new THREE.SphereGeometry(0.24, 12, 12),
    new THREE.MeshStandardMaterial({ color: 0xffffff, emissive: PINK, emissiveIntensity: 1.4 })
  )
  pick.position.set(7, 1.65, 3)
  scene.add(pick)
  updaters.push((_dt, t) => {
    pick.material.emissiveIntensity = 0.8 + Math.max(0, Math.sin(t * 4)) * 1.2
  })

  // ⑧ 自回归回路：选中的 token 弧线送回 GPU 输入
  makeFlowGroup(scene, updaters,
    [new THREE.Vector3(7, 1.7, 3), new THREE.Vector3(3, 6.8, 3.6), new THREE.Vector3(0, 1.3, 0.9)],
    0xf9a8d4, 0.3, 2, 0.13)
  const loopLabel = makeLabel('自回归：新 token 拼回输入', '#f9a8d4', 0.6)
  loopLabel.position.set(3.4, 7.6, 3.6)
  scene.add(loopLabel)

  // ⑨ EOS：序列离开 batch → Detokenizer → 流式返回客户端
  makeNode(scene, 12, 0, 2.8, 1.8, 'Detokenizer · 流式输出', GOLD, { labelColor: '#fde68a' })
  makeFlowGroup(scene, updaters,
    [new THREE.Vector3(8.2, 1.0, 2.8), new THREE.Vector3(10.4, 1.0, 0.6)],
    GOLD, 0.4, 2, 0.12)
  makeFlowGroup(scene, updaters,
    [new THREE.Vector3(13.5, 1.0, 0), new THREE.Vector3(17, 2.6, -2.2), new THREE.Vector3(19.5, 2.8, -3.4)],
    0xfde68a, 0.45, 3, 0.12)
  // 序列结束 → 释放 KV 块（灰脉冲回池）
  makeFlowGroup(scene, updaters,
    [new THREE.Vector3(12, 0.8, -0.9), new THREE.Vector3(8.8, 0.8, -2.4)],
    0x94a3b8, 0.25, 1, 0.1)

  // ---- 动画驱动 ----
  // batch 槽位：空(暗) → prefill(爆闪) → decode(稳亮) → 完成(熄灭)
  updaters.push((_dt, t) => {
    slotMats.forEach((m, i) => {
      const s = (t + i * 1.7) % 11
      m.emissive.setHex(MB_COLORS[i % 4])
      if (s < 0.5) m.emissiveIntensity = 0.03
      else if (s < 2.2) m.emissiveIntensity = 0.7 + Math.random() * 0.8
      else if (s < 9) m.emissiveIntensity = 0.32
      else if (s < 10) m.emissiveIntensity = 0.06
      else m.emissiveIntensity = 0.03
    })
  })
  // KV 块：绿色渐进填充 → 红闪（LRU 驱逐）→ 释放变暗
  updaters.push((_dt, t) => {
    kvMats.forEach((m, i) => {
      const b = (t * 0.7 + i * 0.9) % 9
      if (b < 6.5) {
        m.emissive.setHex(GREEN)
        m.emissiveIntensity = 0.06 + b * 0.11
      } else if (b < 7.5) {
        m.emissive.setHex(0xef4444)
        m.emissiveIntensity = 0.4 + Math.random() * 0.8
      } else {
        m.emissive.setHex(GREEN)
        m.emissiveIntensity = 0.03
      }
    })
  })
  // SM 阵列：周期性 prefill 算力爆发 + decode 步进式稀疏点亮
  updaters.push((_dt, t) => {
    const burst = t % 6 < 0.7
    const row = Math.floor(t * 3) % 10
    smMats.forEach((m, i) => {
      if (burst) m.emissiveIntensity = 0.9 + Math.random() * 0.7
      else m.emissiveIntensity = Math.floor(i / 10) === row ? 1.3 : 0.08
    })
  })
}

const BUILDERS: Record<Mode, (scene: THREE.Scene, updaters: Updater[]) => void> = {
  vllm: buildVllm,
  prefill: buildPrefill,
  decode: buildDecode,
  nvlink: buildNvlink,
  pcie: buildPcie,
  tp: buildTP,
  pp: buildPP,
  ep: buildEP,
}

const CAM_POS: Record<Mode, [number, number, number]> = {
  vllm: [13, 12, 24],
  prefill: [11, 9, 13],
  decode: [11, 9, 13],
  nvlink: [0, 17, 22],
  pcie: [0, 19, 21],
  tp: [0, 8, 21],
  pp: [0, 9, 23],
  ep: [0, 9, 22],
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
    const target: [number, number, number] = mode === 'vllm' ? [0, 1.5, 0] : [0, 0.5, 0]
    controls.target.set(...target)
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
        } else if (obj instanceof THREE.Sprite) {
          const m = obj.material as THREE.SpriteMaterial
          m.map?.dispose()
          m.dispose()
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
    return ([...MODES, ...PARALLEL_MODES].some((m) => m.id === h) ? h : 'vllm') as Mode
  })
  const hud = HUD[mode]
  const isParallel = (PARALLEL_MODES as { id: Mode }[]).some((m) => m.id === mode)
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
          <Link to="/flow" className="text-sm text-slate-400 hover:text-cyan-300 transition-colors">
            ← 返回 2D 推理流程
          </Link>
          <Link to="/" className="text-sm text-slate-500 hover:text-cyan-300 transition-colors">
            🏠 首页
          </Link>
        </div>
        <p className="mt-2 text-sm text-slate-400 max-w-3xl">
          用 3D 看推理时 GPU 里发生了什么：Prefill 的算力高峰、Decode 的带宽瓶颈、多卡互联的 NVLink 高速路与 PCIe 羊肠小道。拖动旋转，滚轮缩放。
        </p>
      </header>

      {/* 场景切换 */}
      <div className="max-w-[1360px] mx-auto px-5 pb-4">
        <div className="flex flex-wrap items-center gap-x-1 gap-y-2">
          <span className="text-[11px] text-slate-500 mr-1">引擎内部</span>
          {MODES.map((m) => (
            <button key={m.id} onClick={() => selectMode(m.id)}
              className={`arch-tab ${mode === m.id ? 'arch-tab-on-moe' : 'arch-tab-off'}`}>
              {m.icon} {m.label}
            </button>
          ))}
        </div>
        <div className="flex flex-wrap items-center gap-x-1 gap-y-2 mt-2">
          <span className="text-[11px] text-slate-500 mr-1">并行策略</span>
          {PARALLEL_MODES.map((m) => (
            <button key={m.id} onClick={() => selectMode(m.id)}
              className={`arch-tab ${mode === m.id ? 'arch-tab-on-moe' : 'arch-tab-off'}`}>
              {m.icon} {m.label}
            </button>
          ))}
        </div>
      </div>

      {/* 3D 画布 + HUD */}
      <div className="max-w-[1360px] mx-auto px-5">
        <div className="relative rounded-xl border border-slate-800 bg-[#0a1120] overflow-hidden" style={{ height: 560 }}>
          <GpuScene mode={mode} />
          {/* HUD 覆盖层 */}
          <div className={`pointer-events-none flex gap-3 ${mode === 'vllm'
            ? 'absolute bottom-8 left-4 items-end'
            : 'absolute top-4 left-4 right-4 items-start flex-wrap'}`}>
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
        {!isParallel && mode !== 'vllm' && (<>
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
        </>)}
        {mode === 'vllm' && (<>
        <div className="info-card">
          <h3 className="info-title">🗺️ vLLM 全景图例</h3>
          <p>
            从左到右：<b>Client → API Server/Tokenizer → 等待队列 → Scheduler</b>（头顶 6 个 batch 槽位：爆闪 = 新请求 prefill 插入，稳亮 = decode 常驻）→ <b>GPU</b>（SM 阵列周期性全亮 = prefill 算力爆发，逐行点亮 = decode 步进）→ <b>KV Block 池</b>（绿 = 占用渐增，红闪 = LRU 驱逐，暗 = 已释放）→ <b>采样器</b>（粉）→ 亮金 token 经 <b>Detokenizer</b> 流式返回；粉色弧线是<b>自回归回路</b>，灰脉冲是 EOS 后释放 KV 块。
          </p>
        </div>
        <div className="info-card">
          <h3 className="info-title">📖 一次请求的一生（vLLM 调度循环）</h3>
          <p>
            ① 请求分词进队列；② 每个调度周期，Scheduler 给新请求分配物理块、塞进 batch 空槽跑 <b>prefill</b>（长 prompt 会被 chunked 切块）；③ 已在跑的序列每步 <b>decode 1 个 token</b>，新 KV 追加进 Block 池；④ 采样出的 token 拼回输入，直到 EOS；⑤ 序列出 batch、释放块、流式吐回。KV 不够时抢占（preempt）最老的序列回队列——画面里就是槽位熄灭、灰脉冲回池。
          </p>
        </div>
        </>)}
        {isParallel && (<>
        <div className="info-card">
          <h3 className="info-title">🗺️ 并行场景图例</h3>
          <p>
            一排 4 块 <b>GPU 板卡</b>（stage）；✂️ TP：上方半透明矩阵是<b>切分前的完整一层 W</b>，4 色实块是被竖切的 <b>1/4 切片</b>——白 token 广播到 4 卡，切片色部分和升入切片、注入头顶的 <b>all-reduce 光环</b>，合并后亮金色 token 输出；🏭 PP：卡上方是<b>不同的层 slab</b>，4 色微批次错位推进；📦 EP：每卡后排 2 个<b>彩色专家</b>、前排琥珀板是<b>本地 token 队列</b>；彩色 token 沿弧线发往目标专家再返回来源卡——无中央 Router，all-to-all 对称往返。
          </p>
        </div>
        <div className="info-card">
          <h3 className="info-title">⚖️ 三种并行速查</h3>
          <p>
            <b>TP 张量并行</b>：切层内权重 · 通信大、延迟敏感 · 只活在 NVLink 域（≤8 卡）<br />
            <b>PP 流水线并行</b>：切层 · 通信小、可跨机上百卡 · 有气泡<br />
            <b>EP 专家并行</b>：切 MoE 专家 · all-to-all、随专家数扩展 · 可跨节点
          </p>
        </div>
        <div className="info-card">
          <h3 className="info-title">🫧 流水线气泡是怎么回事？</h3>
          <p>
            朴素 GPipe 中，第一个微批次要等它流过全部 stage，后面的才能开始——填充与排空阶段的空转就是气泡，约占 <b>(P-1)/M</b>（P=stage 数，M=微批数）。<b>1F1B</b>（一次前向一次反向）让稳态时各 stage 错开算微批，大幅减少显存与气泡；交错式（interleaved）1F1B 进一步压缩。
          </p>
        </div>
        <div className="info-card">
          <h3 className="info-title">🚀 实战组合：DeepSeek-V3 与 3D 并行</h3>
          <p>
            现代大模型训练用 <b>DP × TP × PP × EP</b> 混合：数据并行切 batch，TP 在单机 NVLink 域内切少量敏感层，PP 跨机切层，EP 把 MoE 专家撒到大量卡上。DeepSeek-V3 的关键取舍就是<b>用 EP 替代大部分 TP</b>：all-to-all 对带宽要求低于 all-reduce，跨节点也能跑，再把通信和计算重叠——384 张卡一个 EP 组，把专家并行的扩展性榨干。
          </p>
        </div>
        </>)}
      </div>

      {/* 署名 */}
      <footer className="max-w-[1360px] mx-auto px-5 pb-8 text-center text-xs text-slate-500">
        <a className="hover:text-cyan-300" href="https://www.krli.org" target="_blank" rel="noreferrer">📝 博客 krli.org</a>
        {' · '}
        <a className="hover:text-cyan-300" href="https://github.com/hawkli-1994" target="_blank" rel="noreferrer">🐙 GitHub @hawkli-1994</a>
        {' · '}
        <a className="hover:text-cyan-300" href="https://github.com/hawkli-1994/learn-ai-infra" target="_blank" rel="noreferrer">项目仓库</a>
      </footer>
    </div>
  )
}
