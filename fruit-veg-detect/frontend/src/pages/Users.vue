<template>
  <div class="card-panel page-panel">
    <div class="header-row">
      <div class="section-title">用户管理</div>
      <el-button type="primary" @click="dialogVisible = true">新增用户</el-button>
    </div>

    <el-table :data="users" border v-loading="loading">
      <el-table-column prop="id" label="ID" width="80" />
      <el-table-column prop="username" label="用户名" min-width="180" />
      <el-table-column prop="role" label="角色" min-width="120" />
    </el-table>

    <el-dialog v-model="dialogVisible" title="新增用户" width="420px">
      <el-form :model="form" label-width="80px">
        <el-form-item label="用户名">
          <el-input v-model="form.username" maxlength="32" />
        </el-form-item>
        <el-form-item label="角色">
          <el-select v-model="form.role" style="width: 100%">
            <el-option label="admin" value="admin" />
            <el-option label="operator" value="operator" />
            <el-option label="viewer" value="viewer" />
          </el-select>
        </el-form-item>
      </el-form>
      <template #footer>
        <el-button @click="dialogVisible = false">取消</el-button>
        <el-button type="primary" :loading="submitting" @click="submit">创建</el-button>
      </template>
    </el-dialog>
  </div>
</template>

<script setup lang="ts">
import { onMounted, reactive, ref } from 'vue'
import { ElMessage } from 'element-plus'

import { createUser, getUsers, type UserItem } from '../api/users'

const users = ref<UserItem[]>([])
const loading = ref(false)
const submitting = ref(false)
const dialogVisible = ref(false)

const form = reactive<{
  username: string
  role: 'admin' | 'operator' | 'viewer'
}>({
  username: '',
  role: 'operator',
})

const loadData = async () => {
  loading.value = true
  try {
    const data = await getUsers()
    users.value = data.items
  } catch (error) {
    console.error(error)
    ElMessage.error('加载用户失败')
  } finally {
    loading.value = false
  }
}

const submit = async () => {
  if (!form.username.trim()) {
    ElMessage.warning('请输入用户名')
    return
  }

  submitting.value = true
  try {
    await createUser({
      username: form.username.trim(),
      role: form.role,
    })
    ElMessage.success('创建成功')
    dialogVisible.value = false
    form.username = ''
    form.role = 'operator'
    await loadData()
  } catch (error) {
    console.error(error)
    ElMessage.error('创建失败')
  } finally {
    submitting.value = false
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
  margin-bottom: 12px;
}
</style>
