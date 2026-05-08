<template>
  <div class="detect-canvas card-panel">
    <div class="canvas-head">
      <div class="section-title">{{ title }}</div>
    </div>
    <div class="canvas-body">
      <canvas ref="canvasRef" class="canvas-el"></canvas>
      <div v-if="!imageUrl" class="empty">暂无画面</div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { onBeforeUnmount, ref, watch } from 'vue'

interface DetectBox {
  x1: number
  y1: number
  x2: number
  y2: number
  conf: number
  cls_name: string
}

const props = withDefaults(
  defineProps<{
    imageUrl?: string
    boxes?: DetectBox[]
    title?: string
  }>(),
  {
    imageUrl: '',
    boxes: () => [],
    title: '识别画面',
  },
)

const canvasRef = ref<HTMLCanvasElement | null>(null)
let imageObj: HTMLImageElement | null = null
let imageVersion = 0

const draw = () => {
  const canvas = canvasRef.value
  if (!canvas) {
    return
  }

  const ctx = canvas.getContext('2d')
  if (!ctx) {
    return
  }

  if (!imageObj || !props.imageUrl) {
    canvas.width = 900
    canvas.height = 520
    ctx.clearRect(0, 0, canvas.width, canvas.height)
    return
  }

  const width = imageObj.naturalWidth || imageObj.width
  const height = imageObj.naturalHeight || imageObj.height
  canvas.width = width
  canvas.height = height

  ctx.clearRect(0, 0, width, height)
  ctx.drawImage(imageObj, 0, 0, width, height)

  const lineWidth = Math.max(2, Math.round(Math.min(width, height) / 320))
  const fontSize = Math.max(16, Math.round(Math.min(width, height) / 52))
  const labelHeight = Math.max(22, fontSize + 8)

  ctx.lineWidth = lineWidth
  ctx.font = `${fontSize}px Barlow`
  ctx.textBaseline = 'top'

  for (const box of props.boxes) {
    const x = box.x1
    const y = box.y1
    const w = box.x2 - box.x1
    const h = box.y2 - box.y1

    ctx.strokeStyle = '#1d70ff'
    ctx.fillStyle = '#1d70ff'
    ctx.strokeRect(x, y, w, h)

    const label = `${box.cls_name} ${box.conf.toFixed(2)}`
    const labelWidth = ctx.measureText(label).width + 10
    const labelX = Math.max(0, x)
    const labelY = Math.max(0, y - labelHeight - 2)

    ctx.fillRect(labelX, labelY, labelWidth, labelHeight)
    ctx.fillStyle = '#ffffff'
    ctx.fillText(label, labelX + 5, labelY + Math.max(3, Math.round((labelHeight - fontSize) / 2)))
  }
}

const loadImage = (url?: string) => {
  if (!url) {
    imageObj = null
    draw()
    return
  }

  const version = ++imageVersion
  const img = new Image()
  img.crossOrigin = 'anonymous'
  img.onload = () => {
    if (version !== imageVersion) {
      return
    }
    imageObj = img
    draw()
  }
  img.onerror = () => {
    if (version !== imageVersion) {
      return
    }
    imageObj = null
    draw()
  }
  img.src = url
}

watch(
  () => props.imageUrl,
  (val) => loadImage(val),
  { immediate: true },
)

watch(
  () => props.boxes,
  () => draw(),
  { deep: true },
)

onBeforeUnmount(() => {
  imageVersion += 1
})
</script>

<style scoped>
.detect-canvas {
  padding: 14px;
  min-height: 560px;
}

.canvas-head {
  margin-bottom: 10px;
}

.canvas-body {
  position: relative;
  min-height: 500px;
  border: 1px solid #d7e4f3;
  border-radius: 12px;
  background: radial-gradient(circle at 20% 20%, #f8fcff 0%, #edf4ff 60%, #e6effb 100%);
  display: grid;
  place-items: center;
  overflow: hidden;
}

.canvas-el {
  width: 100%;
  max-height: 70vh;
  object-fit: contain;
}

.empty {
  position: absolute;
  color: #72839e;
  font-size: 16px;
}
</style>
