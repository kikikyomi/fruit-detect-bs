<template>
  <div class="page-grid">
    <div class="left-col">
      <UploadPanel title="上传视频" tip="支持 mp4 / avi / mov / mkv，默认 720p 输出" accept="video/*" @file-change="onFileChange" />

      <div class="card-panel action-panel">
        <div class="section-title">视频参数</div>

        <div class="param-row">
          <span>识别类型</span>
          <ModelPicker v-model="modelKey" />
        </div>

        <div class="preset-row">
          <span class="preset-label">预设方案</span>
          <el-button size="small" type="primary" plain @click="applyPreset('stable')">ID稳定优先</el-button>
          <el-button size="small" @click="applyPreset('balanced')">平衡</el-button>
          <el-button size="small" @click="applyPreset('speed')">速度优先</el-button>
        </div>

        <div class="param-row">
          <span>模式</span>
          <el-select v-model="mode" class="full-width">
            <el-option label="YOLO + DeepSORT 跟踪" value="deepsort" />
            <el-option label="YOLO 检测" value="yolo" />
          </el-select>
        </div>

        <div class="param-row">
          <span>Confidence: {{ conf.toFixed(2) }}</span>
          <el-slider v-model="conf" :min="0" :max="1" :step="0.01" />
        </div>
        <div class="param-row">
          <span>IoU: {{ iou.toFixed(2) }}</span>
          <el-slider v-model="iou" :min="0" :max="1" :step="0.01" />
        </div>
        <div class="param-row">
          <span>frame_skip: {{ frameSkip }}</span>
          <el-slider v-model="frameSkip" :min="1" :max="6" :step="1" />
        </div>
        <div v-if="mode === 'deepsort'" class="param-row">
          <span>断帧容忍度: {{ trackerMaxTimeSinceUpdate }} 帧</span>
          <el-slider v-model="trackerMaxTimeSinceUpdate" :min="0" :max="120" :step="1" />
        </div>

        <div v-if="mode === 'deepsort'" class="param-grid">
          <label>max_age <el-input-number v-model="maxAge" :min="1" :max="160" /></label>
          <label>n_init <el-input-number v-model="nInit" :min="1" :max="10" /></label>
          <label>trail_length <el-input-number v-model="trailLength" :min="1" :max="100" /></label>
          <label>smooth_alpha <el-input-number v-model="smoothAlpha" :min="0" :max="1" :step="0.05" /></label>
        </div>

        <div class="param-grid">
          <label>开始秒 <el-input-number v-model="startTime" :min="0" :step="1" /></label>
          <label>结束秒 <el-input-number v-model="endTime" :min="0" :step="1" /></label>
        </div>

        <div class="switch-list">
          <el-checkbox v-model="saveCsv">保存 CSV</el-checkbox>
          <el-checkbox v-model="saveJson">保存 JSON</el-checkbox>
          <el-checkbox v-model="showStats">画面显示 FPS/帧数</el-checkbox>
        </div>

        <el-collapse v-model="advancedPanels" class="advanced-collapse">
          <el-collapse-item title="高级设置" name="advanced">
            <div class="param-grid">
              <label>imgsz <el-input-number v-model="imgsz" :min="320" :max="1280" :step="32" /></label>
              <label>输出宽度 <el-input-number v-model="outputWidth" :min="320" :max="3840" :step="160" /></label>
              <label>输出高度 <el-input-number v-model="outputHeight" :min="240" :max="2160" :step="90" /></label>
              <label>iou_distance <el-input-number v-model="maxIouDistance" :min="0.1" :max="1" :step="0.05" /></label>
              <label>cosine <el-input-number v-model="maxCosineDistance" :min="0.05" :max="1" :step="0.01" /></label>
              <label>nn_budget <el-input-number v-model="nnBudget" :min="0" :max="300" :step="10" /></label>
              <label>min_box_area <el-input-number v-model="minBoxArea" :min="0" :max="20000" :step="50" /></label>
              <label>max_center_jump <el-input-number v-model="maxCenterJump" :min="0" :max="1000" :step="10" /></label>
            </div>
            <el-checkbox v-model="keepOriginalResolution">保持原始分辨率</el-checkbox>
            <el-checkbox v-model="smoothingEnabled">启用 bbox/轨迹平滑</el-checkbox>
          </el-collapse-item>
        </el-collapse>

        <div class="hint">
          断帧容忍度为 0 时只绘制当前帧已匹配到 YOLO 检测的跟踪框，适合避免转向后预测框继续直线漂移。
        </div>

        <div class="btn-row">
          <el-button type="primary" :loading="loading" :disabled="!fileRef" @click="startDetect">开始处理</el-button>
          <el-button :disabled="loading" @click="clearAll">清空</el-button>
        </div>
      </div>

      <div class="card-panel progress-panel">
        <div class="section-title">处理进度</div>
        <el-progress :percentage="Math.round(task.progress || 0)" />
        <div class="summary-line">状态：{{ task.status || '-' }}</div>
        <div class="summary-line">帧：{{ task.current_frame || 0 }} / {{ task.total_frames || 0 }}</div>
        <div class="summary-line">处理 FPS：{{ (task.fps || 0).toFixed(2) }}</div>
        <div class="summary-line">消息：{{ task.message || '-' }}</div>
      </div>
    </div>

    <div class="center-col">
      <div class="card-panel media-panel">
        <div class="section-title">原视频预览</div>
        <video v-if="sourceVideoUrl" :key="sourceVideoUrl" :src="sourceVideoUrl" controls class="video-el"></video>
        <div v-else class="empty">请先上传视频</div>
      </div>

      <div class="card-panel media-panel">
        <div class="section-title">处理后视频预览</div>
        <video v-if="resultVideoUrl" :key="resultVideoUrl" :src="resultVideoUrl" controls class="video-el"></video>
        <div v-else class="empty">处理完成后显示 output.mp4</div>
        <div class="link-row">
          <a v-if="csvUrl" :href="csvUrl" target="_blank">results.csv</a>
          <a v-if="jsonUrl" :href="jsonUrl" target="_blank">results.json</a>
          <a v-if="summaryUrl" :href="summaryUrl" target="_blank">summary.txt</a>
        </div>
      </div>
    </div>

    <div class="right-col">
      <div class="card-panel summary-panel">
        <div class="section-title">结果统计</div>
        <div class="summary-line">原始总帧数：{{ summary.original_total_frames || 0 }}</div>
        <div class="summary-line">处理帧数：{{ summary.processed_frames || 0 }}</div>
        <div class="summary-line">原始 FPS：{{ summary.original_fps || 0 }}</div>
        <div class="summary-line">原始分辨率：{{ summary.original_resolution || '-' }}</div>
        <div class="summary-line">输出分辨率：{{ summary.output_resolution || '-' }}</div>
        <div class="summary-line">imgsz：{{ summary.imgsz || imgsz }}</div>
        <div class="summary-line">frame_skip：{{ summary.frame_skip || frameSkip }}</div>
        <div class="summary-line">
          断帧容忍度：{{ summary.tracker_max_time_since_update ?? trackerMaxTimeSinceUpdate }} 帧
        </div>
        <div class="summary-line">检测框总数：{{ summary.total_detections || 0 }}</div>
        <div class="summary-line">TrackID 数量：{{ summary.unique_tracks || 0 }}</div>
        <div class="summary-line">平均处理 FPS：{{ summary.average_processing_fps || 0 }}</div>
        <div class="summary-line">YOLO 平均耗时：{{ summary.average_yolo_ms || 0 }} ms</div>
        <div class="summary-line">DeepSORT 平均耗时：{{ summary.average_deepsort_ms || 0 }} ms</div>
        <div class="summary-line">绘制平均耗时：{{ summary.average_draw_ms || 0 }} ms</div>
        <div class="class-tags">
          <el-tag v-for="(count, name) in summary.class_counts" :key="`det-${name}`" effect="plain">
            {{ name }}: {{ count }}
          </el-tag>
        </div>
        <div class="class-tags">
          <el-tag v-for="(count, name) in summary.track_class_counts" :key="`trk-${name}`" type="success" effect="plain">
            Track {{ name }}: {{ count }}
          </el-tag>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { onBeforeUnmount, ref } from 'vue'
import { ElMessage } from 'element-plus'

import ModelPicker from '../components/ModelPicker.vue'
import UploadPanel from '../components/UploadPanel.vue'
import { getApiErrorMessage, toFileUrl } from '../api/client'
import {
  getVideoTask,
  startVideoDetectTask,
  type ModelKey,
  type VideoSummary,
  type VideoTaskResponse,
} from '../api/detect'

type Preset = 'stable' | 'balanced' | 'speed'
type Mode = 'deepsort' | 'yolo'

const fileRef = ref<File | null>(null)
const sourceVideoUrl = ref('')
const resultVideoUrl = ref('')
const csvUrl = ref('')
const jsonUrl = ref('')
const summaryUrl = ref('')
const loading = ref(false)
const modelKey = ref<ModelKey>('fruit')
const mode = ref<Mode>('deepsort')
const advancedPanels = ref<string[]>([])

const imgsz = ref(512)
const conf = ref(0.3)
const iou = ref(0.45)
const frameSkip = ref(1)
const outputWidth = ref(1280)
const outputHeight = ref(720)
const keepOriginalResolution = ref(false)
const maxAge = ref(80)
const trackerMaxTimeSinceUpdate = ref(0)
const nInit = ref(2)
const maxIouDistance = ref(0.8)
const maxCosineDistance = ref(0.5)
const nnBudget = ref(150)
const trailLength = ref(40)
const smoothWindow = ref(1)
const smoothingEnabled = ref(true)
const smoothAlpha = ref(0.65)
const minBoxArea = ref(300)
const maxCenterJump = ref(220)
const saveCsv = ref(true)
const saveJson = ref(true)
const showStats = ref(false)
const startTime = ref<number | undefined>()
const endTime = ref<number | undefined>()
const task = ref<Partial<VideoTaskResponse>>({})
const summary = ref<Partial<VideoSummary>>({})

let pollTimer: number | null = null

const stopPolling = () => {
  if (pollTimer !== null) {
    window.clearInterval(pollTimer)
    pollTimer = null
  }
}

const applyPreset = (preset: Preset) => {
  mode.value = 'deepsort'
  outputWidth.value = 1280
  outputHeight.value = 720
  keepOriginalResolution.value = false
  minBoxArea.value = 300
  showStats.value = false
  smoothingEnabled.value = true
  smoothAlpha.value = 0.65
  trackerMaxTimeSinceUpdate.value = 0

  if (preset === 'speed') {
    imgsz.value = 480
    conf.value = 0.2
    iou.value = 0.45
    frameSkip.value = 1
    maxAge.value = 60
    nInit.value = 2
    maxIouDistance.value = 0.8
    maxCosineDistance.value = 0.5
    nnBudget.value = 100
    trailLength.value = 30
    smoothWindow.value = 1
    maxCenterJump.value = 220
    return
  }

  if (preset === 'balanced') {
    imgsz.value = 512
    conf.value = 0.2
    iou.value = 0.45
    frameSkip.value = 1
    maxAge.value = 80
    nInit.value = 2
    maxIouDistance.value = 0.8
    maxCosineDistance.value = 0.5
    nnBudget.value = 150
    trailLength.value = 40
    smoothWindow.value = 1
    maxCenterJump.value = 220
    return
  }

  imgsz.value = 512
  conf.value = 0.3
  iou.value = 0.45
  frameSkip.value = 1
  maxAge.value = 80
  nInit.value = 2
  maxIouDistance.value = 0.8
  maxCosineDistance.value = 0.5
  nnBudget.value = 150
  trailLength.value = 40
  smoothWindow.value = 1
  maxCenterJump.value = 220
}

const revokeSourceUrl = () => {
  if (sourceVideoUrl.value.startsWith('blob:')) {
    URL.revokeObjectURL(sourceVideoUrl.value)
  }
}

const resetOutputs = () => {
  resultVideoUrl.value = ''
  csvUrl.value = ''
  jsonUrl.value = ''
  summaryUrl.value = ''
  task.value = {}
  summary.value = {}
}

const onFileChange = (file: File | null) => {
  stopPolling()
  revokeSourceUrl()
  fileRef.value = file
  sourceVideoUrl.value = file ? URL.createObjectURL(file) : ''
  resetOutputs()
}

const updateFromTask = (data: VideoTaskResponse) => {
  task.value = data
  if (data.summary) {
    summary.value = data.summary
  }
  if (data.output_video_url) {
    resultVideoUrl.value = toFileUrl(data.output_video_url)
  }
  if (data.result_csv_url) {
    csvUrl.value = toFileUrl(data.result_csv_url)
  }
  if (data.result_json_url) {
    jsonUrl.value = toFileUrl(data.result_json_url)
  }
  if (data.summary_url) {
    summaryUrl.value = toFileUrl(data.summary_url)
  }
}

const pollTask = (taskId: string) => {
  stopPolling()
  pollTimer = window.setInterval(async () => {
    try {
      const data = await getVideoTask(taskId)
      updateFromTask(data)
      if (data.status === 'finished') {
        stopPolling()
        loading.value = false
        ElMessage.success('视频处理完成')
      }
      if (data.status === 'failed') {
        stopPolling()
        loading.value = false
        ElMessage.error(data.error || data.message || '视频处理失败')
      }
    } catch (error) {
      console.error(error)
      stopPolling()
      loading.value = false
      ElMessage.error(getApiErrorMessage(error, '查询视频处理进度失败'))
    }
  }, 1000)
}

const startDetect = async () => {
  if (!fileRef.value) {
    ElMessage.warning('请先上传视频')
    return
  }
  stopPolling()
  resetOutputs()
  loading.value = true
  try {
    const data = await startVideoDetectTask({
      file: fileRef.value,
      conf: conf.value,
      iou: iou.value,
      imgsz: imgsz.value,
      frameSkip: frameSkip.value,
      enableDeepSort: mode.value === 'deepsort',
      trackerBackend: mode.value === 'deepsort' ? 'deepsort' : 'naive',
      modelKey: modelKey.value,
      outputWidth: outputWidth.value,
      outputHeight: outputHeight.value,
      keepOriginalResolution: keepOriginalResolution.value,
      resizeOutput: true,
      maxAge: maxAge.value,
      trackerMaxTimeSinceUpdate: trackerMaxTimeSinceUpdate.value,
      nInit: nInit.value,
      maxIouDistance: maxIouDistance.value,
      maxCosineDistance: maxCosineDistance.value,
      nnBudget: nnBudget.value,
      trailLength: trailLength.value,
      smoothWindow: smoothWindow.value,
      smoothingEnabled: smoothingEnabled.value,
      smoothAlpha: smoothAlpha.value,
      minBoxArea: minBoxArea.value,
      maxCenterJump: maxCenterJump.value,
      saveCsv: saveCsv.value,
      saveJson: saveJson.value,
      saveVideo: true,
      showStats: showStats.value,
      startTime: startTime.value,
      endTime: endTime.value,
    })
    updateFromTask(data)
    pollTask(data.task_id)
  } catch (error) {
    console.error(error)
    loading.value = false
    ElMessage.error(getApiErrorMessage(error, '视频任务创建失败'))
  }
}

const clearAll = () => {
  stopPolling()
  revokeSourceUrl()
  fileRef.value = null
  sourceVideoUrl.value = ''
  resetOutputs()
  loading.value = false
}

applyPreset('stable')

onBeforeUnmount(() => {
  stopPolling()
  revokeSourceUrl()
})
</script>

<style scoped>
.page-grid {
  display: grid;
  grid-template-columns: 360px 1fr 360px;
  gap: 14px;
}

.left-col,
.center-col,
.right-col {
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.action-panel,
.progress-panel,
.media-panel,
.summary-panel {
  padding: 14px;
}

.preset-row,
.btn-row,
.switch-list,
.link-row {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 8px;
}

.preset-row {
  margin: 10px 0 14px;
}

.preset-label,
.hint,
.summary-line,
.param-row,
.param-grid label {
  color: #3e5474;
  font-size: 13px;
}

.param-row {
  margin-bottom: 14px;
}

.param-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 10px;
  margin-bottom: 12px;
}

.param-grid label {
  display: flex;
  flex-direction: column;
  gap: 6px;
}

.full-width {
  width: 100%;
  margin-top: 8px;
}

.advanced-collapse {
  margin-top: 8px;
  --el-collapse-header-height: 36px;
}

.hint {
  line-height: 1.6;
  margin: 10px 0;
}

.summary-line {
  margin-bottom: 7px;
}

.class-tags {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  margin-top: 8px;
}

.video-el {
  width: 100%;
  border-radius: 8px;
  border: 1px solid #d9e4f2;
  background: #091421;
}

.empty {
  min-height: 180px;
  display: grid;
  place-items: center;
  color: #72839e;
  border: 1px dashed #d2deec;
  border-radius: 8px;
}

.link-row {
  margin-top: 10px;
}

.link-row a {
  color: #1d70ff;
  font-size: 13px;
}

@media (max-width: 1400px) {
  .page-grid {
    grid-template-columns: 1fr;
  }
}
</style>
