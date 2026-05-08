<template>
  <div class="upload-panel card-panel">
    <div class="section-title">{{ title }}</div>
    <el-upload
      drag
      :auto-upload="false"
      :limit="1"
      :accept="accept"
      :on-change="handleChange"
      :on-remove="handleRemove"
      :before-upload="beforeUpload"
    >
      <div class="upload-inner">
        <div class="upload-text">点击或拖拽上传</div>
        <div class="upload-tip">{{ tip }}</div>
      </div>
    </el-upload>
    <div v-if="fileName" class="file-name">当前文件：{{ fileName }}</div>
  </div>
</template>

<script setup lang="ts">
import { ref } from 'vue'
import type { UploadProps } from 'element-plus'

const props = withDefaults(
  defineProps<{
    title?: string
    tip?: string
    accept?: string
  }>(),
  {
    title: '上传文件',
    tip: '支持常见图片/视频格式',
    accept: '*/*',
  },
)

const emit = defineEmits<{
  (e: 'file-change', file: File | null): void
}>()

const fileName = ref('')

const handleChange: UploadProps['onChange'] = (uploadFile) => {
  const raw = uploadFile.raw ?? null
  fileName.value = raw?.name ?? ''
  emit('file-change', raw)
}

const handleRemove: UploadProps['onRemove'] = () => {
  fileName.value = ''
  emit('file-change', null)
}

const beforeUpload: UploadProps['beforeUpload'] = () => false
</script>

<style scoped>
.upload-panel {
  padding: 14px;
}

.upload-inner {
  padding: 12px 8px;
}

.upload-text {
  font-size: 15px;
  font-weight: 600;
  color: #173b66;
}

.upload-tip {
  font-size: 12px;
  color: #65758f;
  margin-top: 5px;
}

.file-name {
  margin-top: 8px;
  color: #334e71;
  font-size: 13px;
}

:deep(.el-upload-dragger) {
  border-radius: 12px;
  border: 1px dashed #7da6df;
  background: linear-gradient(180deg, #f9fcff 0%, #f2f8ff 100%);
}
</style>
