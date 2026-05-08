<template>
  <div class="card-panel page-panel">
    <div class="header-row">
      <div class="section-title">图片识别记录</div>
      <el-button @click="loadData" :loading="loading">刷新</el-button>
    </div>

    <el-table :data="records" border v-loading="loading">
      <el-table-column prop="created_at" label="时间" min-width="170" />
      <el-table-column prop="file_name" label="文件名" min-width="220" />
      <el-table-column prop="detection_count" label="检测数量" width="100" />
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

    <el-dialog v-model="previewVisible" title="图片预览" width="780px">
      <img v-if="previewUrl" :src="previewUrl" class="preview-image" alt="preview" />
      <div v-else class="empty">暂无可预览内容</div>
    </el-dialog>
  </div>
</template>

<script setup lang="ts">
import { onMounted, ref } from 'vue'
import { ElMessage } from 'element-plus'

import { toFileUrl } from '../api/client'
import { deleteRecord, getImageRecords, type RecordItem } from '../api/records'

const records = ref<RecordItem[]>([])
const loading = ref(false)
const page = ref(1)
const pageSize = ref(10)
const total = ref(0)

const previewVisible = ref(false)
const previewUrl = ref('')

const loadData = async () => {
  loading.value = true
  try {
    const data = await getImageRecords(page.value, pageSize.value)
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
  previewUrl.value = toFileUrl(row.output_url || row.input_url || '')
  previewVisible.value = true
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

.preview-image {
  width: 100%;
  border-radius: 10px;
  border: 1px solid #dae3f0;
}

.empty {
  min-height: 120px;
  display: grid;
  place-items: center;
  color: #70809a;
}
</style>
