<template>
  <div class="card-panel page-panel">
    <div class="header-row">
      <div class="section-title">视频识别记录</div>
      <el-button @click="loadData" :loading="loading">刷新</el-button>
    </div>

    <el-table :data="records" border v-loading="loading">
      <el-table-column prop="created_at" label="时间" min-width="170" />
      <el-table-column prop="file_name" label="文件名" min-width="220" />
      <el-table-column prop="detection_count" label="检测数量" width="100" />
      <el-table-column label="轨迹数量" width="100">
        <template #default="{ row }">{{ row.summary?.unique_tracks ?? 0 }}</template>
      </el-table-column>
      <el-table-column label="预览" width="120">
        <template #default="{ row }">
          <el-button type="primary" link @click="previewRecord(row)">查看</el-button>
        </template>
      </el-table-column>
      <el-table-column label="操作" width="120">
        <template #default="{ row }">
          <el-popconfirm title="确认删除该记录？" @confirm="remove(row.id)">
            <template #reference>
              <el-button type="danger" link>删除</el-button>
            </template>
          </el-popconfirm>
        </template>
      </el-table-column>
    </el-table>

    <div class="pager">
      <el-pagination
        background
        layout="prev, pager, next, total"
        :total="total"
        :page-size="pageSize"
        :current-page="page"
        @current-change="onPageChange"
      />
    </div>

    <el-dialog v-model="previewVisible" title="视频记录预览" width="980px" destroy-on-close @closed="resetPreview">
      <div class="preview-section">
        <div class="preview-title">轨迹回放流（推荐查看）</div>
        <img v-if="previewStreamUrl" :src="previewStreamUrl" class="preview-video stream-preview" alt="trajectory stream" />
        <div v-else class="empty">当前记录无轨迹回放流</div>
      </div>

      <div class="preview-section">
        <div class="preview-title">输出视频文件</div>
        <video v-if="previewVideoUrl" :key="previewVideoUrl" :src="previewVideoUrl" controls class="preview-video"></video>
        <div v-else class="empty">当前记录无输出视频文件</div>
      </div>

      <div class="preview-section">
        <div class="preview-title">关键帧列表</div>
        <div class="frame-grid">
          <div v-for="(url, index) in previewFrames" :key="url" class="frame-item">
            <el-image :src="url" fit="cover" class="frame-thumb" @click="openFramePreview(index)" />
            <el-button type="primary" link @click="openFramePreview(index)">查看大图</el-button>
          </div>
        </div>
      </div>
    </el-dialog>

    <el-dialog v-model="framePreviewVisible" title="关键帧大图" width="1100px" destroy-on-close>
      <div v-if="currentFrameUrl" class="frame-preview-wrap">
        <img :src="currentFrameUrl" class="frame-preview-image" alt="frame preview" />
      </div>
      <div v-else class="empty">暂无关键帧</div>

      <template #footer>
        <div class="frame-footer">
          <div class="frame-index" v-if="previewFrames.length">
            {{ framePreviewIndex + 1 }} / {{ previewFrames.length }}
          </div>
          <div class="frame-actions">
            <el-button :disabled="framePreviewIndex <= 0" @click="moveFrame(-1)">上一张</el-button>
            <el-button :disabled="framePreviewIndex >= previewFrames.length - 1" @click="moveFrame(1)">下一张</el-button>
          </div>
        </div>
      </template>
    </el-dialog>
  </div>
</template>

<script setup lang="ts">
import { computed, onMounted, ref } from 'vue'
import { ElMessage } from 'element-plus'

import { toFileUrl } from '../api/client'
import { deleteRecord, getVideoRecords, type RecordItem } from '../api/records'

const records = ref<RecordItem[]>([])
const loading = ref(false)
const page = ref(1)
const pageSize = ref(10)
const total = ref(0)

const previewVisible = ref(false)
const previewVideoUrl = ref('')
const previewStreamUrl = ref('')
const previewFrames = ref<string[]>([])
const framePreviewVisible = ref(false)
const framePreviewIndex = ref(0)

const currentFrameUrl = computed(() => previewFrames.value[framePreviewIndex.value] || '')

const loadData = async () => {
  loading.value = true
  try {
    const data = await getVideoRecords(page.value, pageSize.value)
    records.value = data.items
    total.value = data.total
  } catch (error) {
    console.error(error)
    ElMessage.error('加载记录失败')
  } finally {
    loading.value = false
  }
}

const onPageChange = (p: number) => {
  page.value = p
  void loadData()
}

const previewRecord = (row: RecordItem) => {
  previewStreamUrl.value = toFileUrl(row.stream_url || '')
  previewVideoUrl.value = toFileUrl(row.output_url || '')
  previewFrames.value = (row.summary?.keyframes || []).map((url) => toFileUrl(url))
  framePreviewIndex.value = 0
  previewVisible.value = true
}

const resetPreview = () => {
  previewVideoUrl.value = ''
  previewStreamUrl.value = ''
  previewFrames.value = []
  framePreviewVisible.value = false
  framePreviewIndex.value = 0
}

const openFramePreview = (index: number) => {
  framePreviewIndex.value = index
  framePreviewVisible.value = true
}

const moveFrame = (step: number) => {
  const nextIndex = framePreviewIndex.value + step
  if (nextIndex < 0 || nextIndex >= previewFrames.value.length) {
    return
  }
  framePreviewIndex.value = nextIndex
}

const remove = async (id: number) => {
  try {
    await deleteRecord(id)
    ElMessage.success('删除成功')
    await loadData()
  } catch (error) {
    console.error(error)
    ElMessage.error('删除失败')
  }
}

onMounted(() => {
  void loadData()
})
</script>

<style scoped>
.page-panel {
  padding: 14px;
}

.header-row {
  display: flex;
  align-items: center;
  justify-content: space-between;
}

.pager {
  margin-top: 14px;
  display: flex;
  justify-content: flex-end;
}

.preview-section + .preview-section {
  margin-top: 18px;
}

.preview-title {
  margin-bottom: 10px;
  color: #36506f;
  font-size: 14px;
  font-weight: 600;
}

.preview-video {
  width: 100%;
  border-radius: 10px;
  border: 1px solid #dae3f0;
}

.stream-preview {
  display: block;
  background: #091421;
}

.frame-grid {
  margin-top: 6px;
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
}

.frame-item {
  width: 132px;
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 4px;
}

.frame-thumb {
  width: 132px;
  height: 92px;
  border-radius: 8px;
  border: 1px solid #dae3f0;
  cursor: pointer;
}

.frame-preview-wrap {
  display: flex;
  justify-content: center;
}

.frame-preview-image {
  max-width: 100%;
  max-height: 72vh;
  border-radius: 10px;
  border: 1px solid #dae3f0;
}

.frame-footer {
  width: 100%;
  display: flex;
  align-items: center;
  justify-content: space-between;
}

.frame-index {
  color: #70809a;
  font-size: 13px;
}

.frame-actions {
  display: flex;
  gap: 8px;
}

.empty {
  min-height: 90px;
  display: grid;
  place-items: center;
  color: #70809a;
}
</style>
