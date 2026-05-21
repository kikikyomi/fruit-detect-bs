<template>
  <div class="page-grid">
    <div class="left-col">
      <div class="card-panel action-panel">
        <div class="section-title">摄像头控制</div>
        <ModelPicker v-model="modelKey" />
        <div class="param-row">
          <span>Confidence: {{ conf.toFixed(2) }}</span>
          <el-slider v-model="conf" :min="0" :max="1" :step="0.01" />
        </div>
        <div class="param-row">
          <span>IoU: {{ iou.toFixed(2) }}</span>
          <el-slider v-model="iou" :min="0" :max="1" :step="0.01" />
        </div>
        <div class="param-row">
          <span>最小采样间隔: {{ captureIntervalMs }} ms</span>
          <el-slider v-model="captureIntervalMs" :min="40" :max="500" :step="10" />
        </div>
        <div class="param-row">
          <span>跟踪模式</span>
          <el-select v-model="trackerBackend" class="full-width">
            <el-option label="自动" value="auto" />
            <el-option label="DeepSORT（GPU 优先）" value="deepsort" />
            <el-option label="轻量 IoU" value="naive" />
          </el-select>
        </div>
        <div class="param-row">
          <span>断轨容忍帧数: {{ trackerMaxTimeSinceUpdate }}</span>
          <el-slider v-model="trackerMaxTimeSinceUpdate" :min="3" :max="120" :step="1" />
        </div>
        <div class="hint">
          GPU 环境建议先试“DeepSORT + 60ms + 容忍帧数 80”。误检较多时可把 Confidence 调到 0.20。
        </div>
        <div class="preset-row">
          <span class="preset-label">预设方案</span>
          <el-button size="small" @click="applyCameraPreset('stable')">稳定优先</el-button>
          <el-button size="small" @click="applyCameraPreset('balanced')">平衡</el-button>
          <el-button size="small" @click="applyCameraPreset('speed')">速度优先</el-button>
        </div>
        <div class="btn-row">
          <el-button type="primary" :disabled="running" @click="startCamera">开始识别</el-button>
          <el-button type="warning" :disabled="!running" @click="stopCamera">停止识别</el-button>
          <el-button @click="clearBoxes">清空</el-button>
        </div>
        <div class="status">
          状态：{{ running ? `运行中（自适应采样，最小间隔 ${captureIntervalMs} ms）` : '已停止' }}
        </div>
        <div class="status">实时处理 FPS：{{ realtimeFps.toFixed(2) }}</div>
        <div class="status">跟踪器：{{ trackerName }} / {{ deepsortEnabled ? 'DeepSORT' : 'Fallback-IoU' }}</div>
        <div class="status">活动目标 ID：{{ trackSummaries.length }}</div>
        <div class="status mono">session_id: {{ sessionId || '-' }}</div>
      </div>

      <ResultTable :boxes="boxes" />
    </div>

    <div class="center-col">
      <div class="camera-panel card-panel">
        <div class="section-title">实时画面（平滑叠框）</div>
        <div class="camera-stage">
          <video ref="videoRef" class="camera-video" autoplay playsinline muted></video>
          <canvas ref="overlayRef" class="overlay-canvas"></canvas>
          <div v-if="!running" class="camera-empty">点击“开始识别”启动摄像头</div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { onBeforeUnmount, onMounted, ref, watch } from 'vue'
import { ElMessage } from 'element-plus'

import ModelPicker from '../components/ModelPicker.vue'
import ResultTable from '../components/ResultTable.vue'
import { getApiErrorMessage } from '../api/client'
import { cameraFrameDetect, createCameraSession, resetCameraSession, type DetectBox, type ModelKey } from '../api/detect'

interface TrackPoint {
  x?: number
  y?: number
  break?: boolean
}

interface TrackSummary {
  track_id: number
  cls_name: string
  cls_id: number
  first_frame: number
  last_frame: number
  frames_seen: number
  path_length: number
}

const videoRef = ref<HTMLVideoElement | null>(null)
const overlayRef = ref<HTMLCanvasElement | null>(null)

const running = ref(false)
const conf = ref(0.15)
const iou = ref(0.45)
const imgsz = ref(480)
const captureIntervalMs = ref(60)
const trackerBackend = ref<'auto' | 'deepsort' | 'naive'>('deepsort')
const trackerMaxTimeSinceUpdate = ref(80)
const modelKey = ref<ModelKey>('fruit')
const realtimeFps = ref(0)
const boxes = ref<DetectBox[]>([])
const imageSize = ref({ w: 0, h: 0 })
const trajectories = ref<Record<string, TrackPoint[]>>({})
const trackSummaries = ref<TrackSummary[]>([])
const trackerName = ref('-')
const deepsortEnabled = ref(false)
const sessionId = ref('')

let stream: MediaStream | null = null
let detectTimerId: number | null = null
let renderFrameId: number | null = null
let inFlight = false
let captureCanvas: HTMLCanvasElement | null = null
let captureCtx: CanvasRenderingContext2D | null = null
let lastDetectAt = 0

let displayBoxes: DetectBox[] = []
let transitionFrom = new Map<string, DetectBox>()
let transitionTo = new Map<string, DetectBox>()
let transitionStartedAt = 0
let transitionDurationMs = 160

const CAPTURE_MAX_WIDTH = 640
const CAPTURE_JPEG_QUALITY = 0.75

const applyCameraPreset = (preset: 'stable' | 'balanced' | 'speed') => {
  if (preset === 'stable') {
    trackerBackend.value = 'deepsort'
    captureIntervalMs.value = 60
    trackerMaxTimeSinceUpdate.value = 80
    return
  }

  if (preset === 'speed') {
    trackerBackend.value = 'deepsort'
    captureIntervalMs.value = 90
    trackerMaxTimeSinceUpdate.value = 60
    return
  }

  trackerBackend.value = 'deepsort'
  captureIntervalMs.value = 75
  trackerMaxTimeSinceUpdate.value = 80
}

const cloneBox = (box: DetectBox): DetectBox => ({ ...box })

const boxKey = (box: DetectBox, index: number): string =>
  box.track_id !== undefined ? `track:${box.track_id}` : `box:${index}:${box.cls_name}`

const boxesToMap = (items: DetectBox[]): Map<string, DetectBox> => {
  const result = new Map<string, DetectBox>()
  items.forEach((box, index) => {
    result.set(boxKey(box, index), cloneBox(box))
  })
  return result
}

const interpolateNumber = (start: number, end: number, progress: number): number =>
  start + (end - start) * progress

const interpolateBox = (from: DetectBox, to: DetectBox, progress: number): DetectBox => ({
  x1: interpolateNumber(from.x1, to.x1, progress),
  y1: interpolateNumber(from.y1, to.y1, progress),
  x2: interpolateNumber(from.x2, to.x2, progress),
  y2: interpolateNumber(from.y2, to.y2, progress),
  conf: interpolateNumber(from.conf, to.conf, progress),
  cls_id: to.cls_id,
  cls_name: to.cls_name,
  track_id: to.track_id ?? from.track_id,
})

const colorForTrack = (trackId?: number): string => {
  if (trackId === undefined || trackId === null) {
    return '#1d70ff'
  }
  const hue = (trackId * 47) % 360
  return `hsl(${hue} 90% 52%)`
}

const startBoxTransition = (nextBoxes: DetectBox[]) => {
  transitionFrom = boxesToMap(displayBoxes)
  transitionTo = boxesToMap(nextBoxes)
  transitionStartedAt = performance.now()
  transitionDurationMs = Math.max(70, Math.min(220, captureIntervalMs.value * 1.25))

  if (!transitionFrom.size) {
    displayBoxes = nextBoxes.map(cloneBox)
  }
}

const getInterpolatedBoxes = (now: number): DetectBox[] => {
  if (!transitionFrom.size && !transitionTo.size) {
    return displayBoxes.map(cloneBox)
  }

  const progress =
    transitionDurationMs <= 0 ? 1 : Math.min(1, Math.max(0, (now - transitionStartedAt) / transitionDurationMs))

  const mergedKeys = new Set<string>([...transitionFrom.keys(), ...transitionTo.keys()])
  const nextDisplayBoxes: DetectBox[] = []

  mergedKeys.forEach((key) => {
    const fromBox = transitionFrom.get(key)
    const toBox = transitionTo.get(key)

    if (fromBox && toBox) {
      nextDisplayBoxes.push(interpolateBox(fromBox, toBox, progress))
      return
    }

    if (toBox) {
      nextDisplayBoxes.push(cloneBox(toBox))
      return
    }

    if (fromBox && progress < 1) {
      nextDisplayBoxes.push(cloneBox(fromBox))
    }
  })

  if (progress >= 1) {
    transitionFrom = new Map()
    transitionTo = boxesToMap(nextDisplayBoxes)
  }

  return nextDisplayBoxes
}

const resizeOverlay = () => {
  const video = videoRef.value
  const canvas = overlayRef.value
  if (!video || !canvas) return

  canvas.width = Math.max(1, Math.floor(video.clientWidth))
  canvas.height = Math.max(1, Math.floor(video.clientHeight))
}

const buildTrajectoryPoints = (boxesToDraw: DetectBox[]) => {
  const boxCentersByTrack = new Map<number, TrackPoint>()
  boxesToDraw.forEach((box) => {
    if (box.track_id === undefined) return
    boxCentersByTrack.set(box.track_id, {
      x: Math.round((box.x1 + box.x2) / 2),
      y: Math.round((box.y1 + box.y2) / 2),
    })
  })

  const result = new Map<number, TrackPoint[]>()
  Object.entries(trajectories.value).forEach(([trackIdRaw, points]) => {
    const trackId = Number(trackIdRaw)
    const merged: TrackPoint[] = points.map((point) => {
      if (point.break || point.x === undefined || point.y === undefined) {
        return { break: true }
      }
      return { x: point.x, y: point.y }
    })
    const currentCenter = boxCentersByTrack.get(trackId)
    if (currentCenter) {
      let lastPoint: TrackPoint | undefined
      for (let index = merged.length - 1; index >= 0; index -= 1) {
        if (!merged[index].break) {
          lastPoint = merged[index]
          break
        }
      }
      if (!lastPoint || lastPoint.x !== currentCenter.x || lastPoint.y !== currentCenter.y) {
        merged.push(currentCenter)
      }
    }
    result.set(trackId, merged)
  })

  return result
}

const drawOverlay = (boxesToDraw: DetectBox[]) => {
  const canvas = overlayRef.value
  const ctx = canvas?.getContext('2d')
  if (!canvas || !ctx) return

  ctx.clearRect(0, 0, canvas.width, canvas.height)
  if (!imageSize.value.w || !imageSize.value.h) return

  const scaleX = canvas.width / imageSize.value.w
  const scaleY = canvas.height / imageSize.value.h
  const trajectoriesToDraw = buildTrajectoryPoints(boxesToDraw)

  ctx.lineWidth = 2
  ctx.font = '15px Barlow'
  ctx.textBaseline = 'top'

  trajectoriesToDraw.forEach((points, trackId) => {
    if (points.length < 2) return
    ctx.strokeStyle = colorForTrack(trackId)
    ctx.beginPath()
    let hasSegment = false
    points.forEach((point, index) => {
      if (point.break || point.x === undefined || point.y === undefined) {
        hasSegment = false
        return
      }
      const sx = point.x * scaleX
      const sy = point.y * scaleY
      if (index === 0 || !hasSegment) {
        ctx.moveTo(sx, sy)
        hasSegment = true
      } else {
        ctx.lineTo(sx, sy)
      }
    })
    ctx.stroke()
  })

  boxesToDraw.forEach((box) => {
    const x1 = box.x1 * scaleX
    const y1 = box.y1 * scaleY
    const x2 = box.x2 * scaleX
    const y2 = box.y2 * scaleY

    const color = colorForTrack(box.track_id)
    ctx.strokeStyle = color
    ctx.fillStyle = color
    ctx.strokeRect(x1, y1, x2 - x1, y2 - y1)

    const trackPart = box.track_id !== undefined ? `ID ${box.track_id} ` : ''
    const label = `${trackPart}${box.cls_name} ${box.conf.toFixed(2)}`
    const labelWidth = ctx.measureText(label).width + 8
    const labelY = Math.max(0, y1 - 22)
    ctx.fillRect(x1, labelY, labelWidth, 20)
    ctx.fillStyle = '#fff'
    ctx.fillText(label, x1 + 4, labelY + 3)
  })
}

const renderLoop = (now: number) => {
  displayBoxes = getInterpolatedBoxes(now)
  drawOverlay(displayBoxes)
  renderFrameId = window.requestAnimationFrame(renderLoop)
}

const startRenderLoop = () => {
  if (renderFrameId !== null) return
  renderFrameId = window.requestAnimationFrame(renderLoop)
}

const stopRenderLoop = () => {
  if (renderFrameId !== null) {
    window.cancelAnimationFrame(renderFrameId)
    renderFrameId = null
  }
}

const captureFrame = async (): Promise<Blob | null> => {
  const video = videoRef.value
  if (!video || video.videoWidth === 0 || video.videoHeight === 0) return null

  if (!captureCanvas) {
    captureCanvas = document.createElement('canvas')
    captureCtx = captureCanvas.getContext('2d')
  }
  if (!captureCanvas || !captureCtx) return null

  const targetWidth = Math.max(1, Math.min(video.videoWidth, CAPTURE_MAX_WIDTH))
  const targetHeight = Math.max(1, Math.round((video.videoHeight * targetWidth) / video.videoWidth))

  if (captureCanvas.width !== targetWidth || captureCanvas.height !== targetHeight) {
    captureCanvas.width = targetWidth
    captureCanvas.height = targetHeight
  }

  captureCtx.clearRect(0, 0, captureCanvas.width, captureCanvas.height)
  captureCtx.drawImage(video, 0, 0, captureCanvas.width, captureCanvas.height)
  return new Promise((resolve) => {
    captureCanvas?.toBlob((blob) => resolve(blob), 'image/jpeg', CAPTURE_JPEG_QUALITY)
  })
}

const scheduleNextDetect = (delayMs = 0) => {
  if (!running.value) return
  if (detectTimerId !== null) {
    window.clearTimeout(detectTimerId)
  }
  detectTimerId = window.setTimeout(() => {
    void tickDetect()
  }, Math.max(0, Math.round(delayMs)))
}

const tickDetect = async () => {
  if (!running.value || inFlight) return

  const startedAt = performance.now()
  inFlight = true
  try {
    const blob = await captureFrame()
    if (!blob) return

    const file = new File([blob], `camera_${Date.now()}.jpg`, { type: 'image/jpeg' })
    const data = await cameraFrameDetect(
      file,
      conf.value,
      iou.value,
      sessionId.value || undefined,
      trackerBackend.value,
      trackerMaxTimeSinceUpdate.value,
      modelKey.value,
      imgsz.value,
    )
    sessionId.value = data.session_id
    boxes.value = (data.result.boxes || []).map(cloneBox)
    imageSize.value = data.result.image_size
    trajectories.value = data.tracking.trajectories || {}
    trackSummaries.value = data.tracking.track_summaries || []
    trackerName.value = data.tracking.tracker || '-'
    deepsortEnabled.value = Boolean(data.tracking.deepsort_enabled)
    startBoxTransition(boxes.value)

    const now = performance.now()
    if (lastDetectAt > 0) {
      const delta = now - lastDetectAt
      if (delta > 0) {
        realtimeFps.value = 1000 / delta
      }
    }
    lastDetectAt = now
  } catch (error) {
    console.error(error)
  } finally {
    inFlight = false
    if (running.value) {
      const elapsed = performance.now() - startedAt
      scheduleNextDetect(Math.max(0, captureIntervalMs.value - elapsed))
    }
  }
}

const startCamera = async () => {
  if (running.value) return

  const hostname = window.location.hostname
  const isLocalhost = hostname === 'localhost' || hostname === '127.0.0.1'
  if (!window.isSecureContext && !isLocalhost) {
    ElMessage.error('摄像头识别需要通过 localhost 或 HTTPS 访问，当前地址会被浏览器拦截')
    return
  }

  if (!navigator.mediaDevices?.getUserMedia) {
    ElMessage.error('当前浏览器不支持摄像头访问')
    return
  }

  try {
    const session = await createCameraSession()
    sessionId.value = session.session_id
  } catch (error) {
    console.error(error)
    ElMessage.warning('创建跟踪会话失败，将使用后端自动会话')
    sessionId.value = ''
  }

  try {
    stream = await navigator.mediaDevices.getUserMedia({
      video: {
        width: { ideal: 640 },
        height: { ideal: 480 },
        frameRate: { ideal: 20, max: 24 },
        facingMode: 'environment',
      },
      audio: false,
    })
    const video = videoRef.value
    if (!video) {
      stream.getTracks().forEach((track) => track.stop())
      stream = null
      return
    }

    video.srcObject = stream
    await video.play()
    resizeOverlay()

    displayBoxes = []
    transitionFrom = new Map()
    transitionTo = new Map()
    lastDetectAt = 0
    realtimeFps.value = 0

    running.value = true
    startRenderLoop()
    scheduleNextDetect(0)
  } catch (error) {
    console.error(error)
    ElMessage.error(getApiErrorMessage(error, '摄像头启动失败，请检查权限或后端地址'))
  }
}

const stopCamera = () => {
  running.value = false

  if (detectTimerId !== null) {
    window.clearTimeout(detectTimerId)
    detectTimerId = null
  }
  stopRenderLoop()

  if (stream) {
    stream.getTracks().forEach((track) => track.stop())
    stream = null
  }

  const video = videoRef.value
  if (video) {
    video.srcObject = null
  }

  if (sessionId.value) {
    void resetCameraSession(sessionId.value)
  }

  sessionId.value = ''
  boxes.value = []
  imageSize.value = { w: 0, h: 0 }
  trajectories.value = {}
  trackSummaries.value = []
  trackerName.value = '-'
  deepsortEnabled.value = false
  captureCanvas = null
  captureCtx = null
  displayBoxes = []
  transitionFrom = new Map()
  transitionTo = new Map()
  lastDetectAt = 0
  realtimeFps.value = 0
  drawOverlay([])
}

const clearBoxes = () => {
  boxes.value = []
  trajectories.value = {}
  trackSummaries.value = []
  displayBoxes = []
  transitionFrom = new Map()
  transitionTo = new Map()
  drawOverlay([])
}

watch(modelKey, () => {
  clearBoxes()
  if (sessionId.value) {
    void resetCameraSession(sessionId.value)
  }
})

onMounted(() => {
  window.addEventListener('resize', resizeOverlay)
})

onBeforeUnmount(() => {
  stopCamera()
  window.removeEventListener('resize', resizeOverlay)
})
</script>

<style scoped>
.page-grid {
  display: grid;
  grid-template-columns: 420px 1fr;
  gap: 14px;
}

.left-col,
.center-col {
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.action-panel {
  padding: 14px;
}

.param-row {
  margin-bottom: 14px;
  color: #35506e;
  font-size: 13px;
}

.full-width {
  width: 100%;
  margin-top: 8px;
}

.hint {
  margin-bottom: 12px;
  color: #5d728f;
  font-size: 12px;
  line-height: 1.6;
}

.preset-row {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 8px;
  margin-bottom: 12px;
}

.preset-label {
  color: #5d728f;
  font-size: 12px;
}

.btn-row {
  display: flex;
  gap: 10px;
}

.status {
  margin-top: 8px;
  color: #3d5678;
  font-size: 12px;
}

.camera-panel {
  padding: 14px;
  min-height: 600px;
}

.camera-stage {
  position: relative;
  border: 1px solid #d7e4f3;
  border-radius: 12px;
  overflow: hidden;
  background: #d8e2ee;
}

.camera-video {
  width: 100%;
  display: block;
}

.overlay-canvas {
  position: absolute;
  inset: 0;
  width: 100%;
  height: 100%;
  pointer-events: none;
}

.camera-empty {
  position: absolute;
  inset: 0;
  display: grid;
  place-items: center;
  color: #3d4f69;
  font-size: 16px;
  background: rgba(240, 246, 255, 0.64);
}

@media (max-width: 1400px) {
  .page-grid {
    grid-template-columns: 1fr;
  }
}
</style>
