<template>
  <div class="page-grid">
    <div class="left-col">
      <UploadPanel title="上传图片" tip="支持 jpg/png/jpeg" accept="image/*" @file-change="onFileChange" />

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
        <div class="btn-row">
          <el-button type="primary" :loading="loading" @click="startDetect">开始识别</el-button>
          <el-button @click="clearAll">清空</el-button>
        </div>
        <div class="hint">record_id: {{ recordId || '-' }}</div>
      </div>
    </div>

    <div class="center-col">
      <div class="card-panel switch-panel">
        <el-radio-group v-model="viewMode" size="small">
          <el-radio-button label="original">原图+叠框</el-radio-button>
          <el-radio-button label="annotated">后端标注图</el-radio-button>
        </el-radio-group>
      </div>
      <DetectCanvas :title="canvasTitle" :image-url="displayImageUrl" :boxes="displayBoxes" />
    </div>

    <div class="right-col">
      <ResultTable :boxes="boxes" />
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed, onBeforeUnmount, ref, watch } from 'vue'
import { ElMessage } from 'element-plus'

import DetectCanvas from '../components/DetectCanvas.vue'
import ModelPicker from '../components/ModelPicker.vue'
import ResultTable from '../components/ResultTable.vue'
import UploadPanel from '../components/UploadPanel.vue'
import { imageDetect, type DetectBox, type ModelKey } from '../api/detect'
import { getApiErrorMessage, toFileUrl } from '../api/client'

const fileRef = ref<File | null>(null)
const localImageUrl = ref('')
const annotatedImageUrl = ref('')
const boxes = ref<DetectBox[]>([])
const loading = ref(false)
const recordId = ref<number | null>(null)

const conf = ref(0.25)
const iou = ref(0.45)
const viewMode = ref<'original' | 'annotated'>('original')
const modelKey = ref<ModelKey>('fruit')

const displayImageUrl = computed(() => {
  if (viewMode.value === 'annotated') {
    return annotatedImageUrl.value || localImageUrl.value
  }
  return localImageUrl.value
})

const displayBoxes = computed(() => (viewMode.value === 'annotated' ? [] : boxes.value))
const canvasTitle = computed(() => (viewMode.value === 'annotated' ? '标注图展示' : '原图叠框展示'))

watch(modelKey, () => {
  annotatedImageUrl.value = ''
  boxes.value = []
  recordId.value = null
  viewMode.value = 'original'
})

const revokeLocalUrl = () => {
  if (localImageUrl.value.startsWith('blob:')) {
    URL.revokeObjectURL(localImageUrl.value)
  }
}

const onFileChange = (file: File | null) => {
  revokeLocalUrl()
  fileRef.value = file
  localImageUrl.value = file ? URL.createObjectURL(file) : ''
  annotatedImageUrl.value = ''
  boxes.value = []
  recordId.value = null
}

const startDetect = async () => {
  if (!fileRef.value) {
    ElMessage.warning('请先上传图片')
    return
  }

  loading.value = true
  try {
    const data = await imageDetect(fileRef.value, conf.value, iou.value, modelKey.value)
    boxes.value = data.result.boxes || []
    annotatedImageUrl.value = toFileUrl(data.annotated_image_url)
    recordId.value = data.record_id
    if (data.saved) {
      ElMessage.success(`识别完成，目标数: ${boxes.value.length}`)
    } else {
      ElMessage.info('识别完成，但未检测到目标，本次结果未保存到 uploads/outputs')
    }
  } catch (error) {
    console.error(error)
    ElMessage.error(getApiErrorMessage(error, '识别失败，请检查后端服务'))
  } finally {
    loading.value = false
  }
}

const clearAll = () => {
  revokeLocalUrl()
  fileRef.value = null
  localImageUrl.value = ''
  annotatedImageUrl.value = ''
  boxes.value = []
  recordId.value = null
}

onBeforeUnmount(() => {
  revokeLocalUrl()
})
</script>

<style scoped>
.page-grid {
  display: grid;
  grid-template-columns: 320px 1fr 420px;
  gap: 14px;
}

.left-col,
.right-col,
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

.btn-row {
  display: flex;
  gap: 10px;
}

.hint {
  margin-top: 10px;
  color: #5d728f;
  font-size: 12px;
}

.switch-panel {
  padding: 10px 14px;
}

@media (max-width: 1400px) {
  .page-grid {
    grid-template-columns: 1fr;
  }
}
</style>
