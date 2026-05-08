<template>
  <div class="card-panel profile-panel" v-loading="loading">
    <div class="section-title">个人中心</div>
    <el-descriptions border :column="1" class="profile-desc">
      <el-descriptions-item label="用户名">{{ profile.username }}</el-descriptions-item>
      <el-descriptions-item label="角色">{{ profile.role }}</el-descriptions-item>
      <el-descriptions-item label="邮箱">{{ profile.email }}</el-descriptions-item>
    </el-descriptions>
  </div>
</template>

<script setup lang="ts">
import { onMounted, reactive, ref } from 'vue'
import { ElMessage } from 'element-plus'

import { getProfile } from '../api/users'

const loading = ref(false)
const profile = reactive({
  username: '-',
  role: '-',
  email: '-',
})

const load = async () => {
  loading.value = true
  try {
    const data = await getProfile()
    profile.username = data.username
    profile.role = data.role
    profile.email = data.email
  } catch (error) {
    console.error(error)
    ElMessage.error('加载个人信息失败')
  } finally {
    loading.value = false
  }
}

onMounted(() => {
  void load()
})
</script>

<style scoped>
.profile-panel {
  padding: 14px;
}

.profile-desc {
  max-width: 680px;
}
</style>
