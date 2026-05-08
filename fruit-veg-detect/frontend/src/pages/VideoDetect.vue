<template>
  <div class="page-grid">
    <div class="left-col">
      <UploadPanel title="上传视频" tip="支持 mp4 / avi / mov" accept="video/*" @file-change="onFileChange" />

      <div class="card-panel action-panel">
        <div class="section-title">参数设置</div>
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
          <span>抽帧间隔: {{ frameInterval }}</span>
          <el-slider v-model="frameInterval" :min="1" :max="12" :step="1" />
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
          <el-slider v-model="trackerMaxTimeSinceUpdate" :min="3" :max="20" :step="1" />
        </div>
        <div class="hint">
          GPU 环境建议先试“DeepSORT + 抽帧 1 或 2 + 容忍帧数 10”。如果速度不够，再改成“自动”或“轻量 IoU”。
        </div>
        <div class="preset-row">
          <span class="preset-label">预设方案</span>
          <el-button size="small" @click="applyVideoPreset('stable')">稳定优先</el-button>
          <el-button size="small" @click="applyVideoPreset('balanced')">平衡</el-button>
          <el-button size="small" @click="applyVideoPreset('speed')">速度优先</el-button>
        </div>
        <div class="btn-row">
          <el-button type="primary" :loading="loading" @click="startDetect">开始识别</el-button>
          <el-button @click="clearAll">清空</el-button>
        </div>
        <div class="hint">record_id: {{ recordId || '-' }}</div>
      </div>

      <div class="card-panel summary-panel">
        <div class="section-title">处理统计</div>
        <div class="summary-line">总帧数：{{ summary.total_frames }}</div>
        <div class="summary-line">采样帧数：{{ summary.sampled_frames }}</div>
        <div class="summary-line">检测总数：{{ summary.detections }}</div>
        <div class="summary-line">源视频 FPS：{{ summary.input_fps.toFixed(2) }}</div>
        <div class="summary-line">总耗时：{{ summary.processing_seconds.toFixed(2) }}s</div>
        <div class="summary-line">检测阶段：{{ summary.analysis_seconds?.toFixed(2) ?? '0.00' }}s</div>
        <div class="summary-line">渲染阶段：{{ summary.render_seconds?.toFixed(2) ?? '0.00' }}s</div>
        <div class="summary-line">综合处理 FPS：{{ summary.processing_fps.toFixed(2) }}</div>
        <div class="summary-line">采样检测 FPS：{{ summary.detection_fps.toFixed(2) }}</div>
        <div class="summary-line">15 FPS 目标：{{ summary.target_fps_met ? '已达到' : '未达到' }}</div>
        <div class="summary-line">抽帧间隔：{{ summary.frame_interval }}</div>
        <div class="summary-line">轨迹数量：{{ summary.unique_tracks }}</div>
        <div class="summary-line">跟踪器：{{ summary.tracker }}</div>
        <div class="summary-line">DeepSORT n_init：{{ summary.tracker_n_init }}</div>
        <div class="summary-line">max_time_since_update：{{ summary.tracker_max_time_since_update }}</div>
        <div class="summary-line">
          DeepSORT：{{ summary.deepsort_enabled ? '已启用' : '未启用（当前走轻量 IoU 跟踪）' }}
        </div>
        <div class="class-tags">
          <el-tag v-for="(count, name) in summary.class_stats" :key="`det-${name}`" type="primary" effect="plain">
            检测 {{ name }}: {{ count }}
          </el-tag>
        </div>
        <div class="class-tags">
          <el-tag
            v-for="(count, name) in summary.track_class_stats"
            :key="`track-${name}`"
            type="success"
            effect="plain"
          >
            轨迹 {{ name }}: {{ count }}
          </el-tag>
        </div>
      </div>
    </div>

    <div class="center-col">
      <div class="card-panel media-panel">
        <div class="section-title">原视频</div>
        <video v-if="sourceVideoUrl" :key="sourceVideoUrl" :src="sourceVideoUrl" controls class="video-el"></video>
        <div v-else class="empty">请先上传视频</div>
      </div>

      <div class="card-panel media-panel">
        <div class="section-title">轨迹回放流</div>
        <img v-if="resultStreamUrl" :src="resultStreamUrl" class="video-el stream-el" alt="trajectory stream" />
        <div v-else class="empty">暂无轨迹回放流</div>
      </div>

      <div class="card-panel media-panel">
        <div class="section-title">输出视频文件</div>
        <video v-if="resultVideoUrl" :key="resultVideoUrl" :src="resultVideoUrl" controls class="video-el"></video>
        <div v-else class="empty">暂无输出视频文件</div>
      </div>
    </div>

    <div class="right-col">
      <DetectCanvas title="关键帧预览" :image-url="selectedKeyframe" :boxes="[]" />

      <div class="card-panel frame-list-panel">
        <div class="section-title">关键帧列表</div>
        <div class="frames-wrap">
          <el-image
            v-for="url in keyframes"
            :key="url"
            :src="url"
            fit="cover"
            class="frame-thumb"
            @click="selectedKeyframe = url"
          />
        </div>
      </div>

      <div class="card-panel track-table-panel">
        <div class="section-title">轨迹列表（按 ID）</div>
        <el-table :data="trackSummaries" size="small" border height="260">
          <el-table-column prop="track_id" label="ID" width="70" />
          <el-table-column prop="cls_name" label="类别" min-width="110" />
          <el-table-column prop="frames_seen" label="出现帧数" width="96" />
          <el-table-column prop="path_length" label="轨迹长度(px)" width="120" />
        </el-table>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { onBeforeUnmount, ref, watch } from 'vue'
import { ElMessage } from 'element-plus'

import DetectCanvas from '../components/DetectCanvas.vue'
import ModelPicker from '../components/ModelPicker.vue'
import UploadPanel from '../components/UploadPanel.vue'
import { getApiErrorMessage, toFileUrl } from '../api/client'
import { videoDetect, type ModelKey } from '../api/detect'

interface TrackSummary {
  track_id: number
  cls_name: string
  cls_id: number
  first_frame: number
  last_frame: number
  frames_seen: number
  path_length: number
}

interface Summary {
  total_frames: number
  sampled_frames: number
  detections: number
  input_fps: number
  processing_seconds: number
  processing_fps: number
  detection_fps: number
  target_fps_met: boolean
  analysis_seconds?: number
  render_seconds?: number
  class_stats: Record<string, number>
  frame_interval: number
  unique_tracks: number
  track_class_stats: Record<string, number>
  tracker: string
  deepsort_enabled: boolean
  tracker_n_init: number
  tracker_max_time_since_update: number
  model_key?: ModelKey
  saved?: boolean
}

const fileRef = ref<File | null>(null)
const sourceVideoUrl = ref('')
const resultVideoUrl = ref('')
const resultStreamUrl = ref('')
const keyframes = ref<string[]>([])
const selectedKeyframe = ref('')
const loading = ref(false)
const recordId = ref<number | null>(null)
const trackSummaries = ref<TrackSummary[]>([])

const conf = ref(0.25)
const iou = ref(0.45)
const frameInterval = ref(2)
const trackerBackend = ref<'auto' | 'deepsort' | 'naive'>('auto')
const trackerMaxTimeSinceUpdate = ref(10)
const modelKey = ref<ModelKey>('fruit')

const applyVideoPreset = (preset: 'stable' | 'balanced' | 'speed') => {
  if (preset === 'stable') {
    trackerBackend.value = 'deepsort'
    frameInterval.value = 1
    trackerMaxTimeSinceUpdate.value = 12
    return
  }

  if (preset === 'speed') {
    trackerBackend.value = 'naive'
    frameInterval.value = 3
    trackerMaxTimeSinceUpdate.value = 6
    return
  }

  trackerBackend.value = 'auto'
  frameInterval.value = 2
  trackerMaxTimeSinceUpdate.value = 10
}

const summary = ref<Summary>({
  total_frames: 0,
  sampled_frames: 0,
  detections: 0,
  input_fps: 0,
  processing_seconds: 0,
  processing_fps: 0,
  detection_fps: 0,
  target_fps_met: false,
  analysis_seconds: 0,
  render_seconds: 0,
  class_stats: {},
  frame_interval: frameInterval.value,
  unique_tracks: 0,
  track_class_stats: {},
  tracker: '-',
  deepsort_enabled: false,
  tracker_n_init: 1,
  tracker_max_time_since_update: 0,
})

const resetSummary = () => {
  summary.value = {
    total_frames: 0,
    sampled_frames: 0,
    detections: 0,
    input_fps: 0,
    processing_seconds: 0,
    processing_fps: 0,
    detection_fps: 0,
    target_fps_met: false,
    analysis_seconds: 0,
    render_seconds: 0,
    class_stats: {},
    frame_interval: frameInterval.value,
    unique_tracks: 0,
    track_class_stats: {},
    tracker: '-',
    deepsort_enabled: false,
    tracker_n_init: 1,
    tracker_max_time_since_update: 0,
  }
}

const revokeSourceUrl = () => {
  if (sourceVideoUrl.value.startsWith('blob:')) {
    URL.revokeObjectURL(sourceVideoUrl.value)
  }
}

const onFileChange = (file: File | null) => {
  revokeSourceUrl()
  fileRef.value = file
  sourceVideoUrl.value = file ? URL.createObjectURL(file) : ''
  resultVideoUrl.value = ''
  resultStreamUrl.value = ''
  keyframes.value = []
  selectedKeyframe.value = ''
  recordId.value = null
  trackSummaries.value = []
  resetSummary()
}

watch(modelKey, () => {
  resultVideoUrl.value = ''
  resultStreamUrl.value = ''
  keyframes.value = []
  selectedKeyframe.value = ''
  recordId.value = null
  trackSummaries.value = []
  resetSummary()
})

const startDetect = async () => {
  if (!fileRef.value) {
    ElMessage.warning('请先上传视频')
    return
  }

  loading.value = true
  try {
    const data = await videoDetect(
      fileRef.value,
      conf.value,
      iou.value,
      frameInterval.value,
      trackerBackend.value,
      trackerMaxTimeSinceUpdate.value,
      modelKey.value,
    )
    summary.value = data.summary
    resultVideoUrl.value = toFileUrl(data.outputs.video_url || '')
    resultStreamUrl.value = toFileUrl(data.outputs.stream_url || '')
    keyframes.value = (data.outputs.keyframes || []).map((url) => toFileUrl(url))
    selectedKeyframe.value = keyframes.value[0] || ''
    recordId.value = data.record_id
    trackSummaries.value = data.tracking.track_summaries || []

    if (data.saved) {
      ElMessage.success(
        `视频处理完成，检测 ${summary.value.detections} 次，轨迹 ${summary.value.unique_tracks} 条`,
      )
    } else {
      ElMessage.info('视频处理完成，但没有检测到目标，临时文件已自动清理')
    }
  } catch (error) {
    console.error(error)
    ElMessage.error(getApiErrorMessage(error, '视频处理失败，请检查后端服务'))
  } finally {
    loading.value = false
  }
}

const clearAll = () => {
  revokeSourceUrl()
  fileRef.value = null
  sourceVideoUrl.value = ''
  resultVideoUrl.value = ''
  resultStreamUrl.value = ''
  keyframes.value = []
  selectedKeyframe.value = ''
  recordId.value = null
  trackSummaries.value = []
  resetSummary()
}

onBeforeUnmount(() => {
  revokeSourceUrl()
})
</script>

<style scoped>
.page-grid {
  display: grid;
  grid-template-columns: 320px 1fr 460px;
  gap: 14px;
}

.left-col,
.right-col,
.center-col {
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.action-panel,
.summary-panel,
.frame-list-panel,
.media-panel,
.track-table-panel {
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

.btn-row {
  display: flex;
  gap: 10px;
}

.hint {
  margin-top: 10px;
  color: #5d728f;
  font-size: 12px;
  line-height: 1.6;
}

.preset-row {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 8px;
  margin-top: 10px;
}

.preset-label {
  color: #5d728f;
  font-size: 12px;
}

.summary-line {
  color: #3e5474;
  margin-bottom: 6px;
  font-size: 13px;
}

.class-tags {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  margin-top: 8px;
}

.video-el {
  width: 100%;
  border-radius: 12px;
  border: 1px solid #d9e4f2;
}

.stream-el {
  display: block;
  background: #091421;
  object-fit: contain;
}

.frames-wrap {
  display: flex;
  gap: 8px;
  flex-wrap: wrap;
  max-height: 300px;
  overflow-y: auto;
}

.frame-thumb {
  width: 120px;
  height: 84px;
  border-radius: 8px;
  border: 1px solid #d6e0ee;
  cursor: pointer;
}

.empty {
  min-height: 120px;
  display: grid;
  place-items: center;
  color: #72839e;
}

@media (max-width: 1400px) {
  .page-grid {
    grid-template-columns: 1fr;
  }
}
</style>
