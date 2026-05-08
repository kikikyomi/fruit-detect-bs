<template>
  <div class="result-table card-panel">
    <div class="section-title">检测目标列表</div>
    <el-table :data="boxes" height="420" size="small" border>
      <el-table-column label="ID" min-width="70">
        <template #default="{ row }">{{ row.track_id ?? '-' }}</template>
      </el-table-column>
      <el-table-column prop="cls_name" label="类别" min-width="90" />
      <el-table-column label="置信度" min-width="90">
        <template #default="{ row }">{{ formatConf(row.conf) }}</template>
      </el-table-column>
      <el-table-column label="x1" min-width="70">
        <template #default="{ row }">{{ formatCoord(row.x1) }}</template>
      </el-table-column>
      <el-table-column label="y1" min-width="70">
        <template #default="{ row }">{{ formatCoord(row.y1) }}</template>
      </el-table-column>
      <el-table-column label="x2" min-width="70">
        <template #default="{ row }">{{ formatCoord(row.x2) }}</template>
      </el-table-column>
      <el-table-column label="y2" min-width="70">
        <template #default="{ row }">{{ formatCoord(row.y2) }}</template>
      </el-table-column>
    </el-table>
  </div>
</template>

<script setup lang="ts">
interface DetectBox {
  x1: number
  y1: number
  x2: number
  y2: number
  conf: number
  cls_name: string
  track_id?: number
}

defineProps<{
  boxes: DetectBox[]
}>()

const formatConf = (value: number) => Number(value || 0).toFixed(3)
const formatCoord = (value: number) => Math.round(Number(value || 0))
</script>

<style scoped>
.result-table {
  padding: 14px;
}
</style>
