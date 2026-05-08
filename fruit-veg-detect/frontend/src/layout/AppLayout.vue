<template>
  <el-container class="app-shell">
    <el-aside width="220px" class="sidebar">
      <div class="logo-wrap">
        <div class="logo-mark"></div>
        <div class="logo-title">果蔬识别系统</div>
      </div>
      <el-menu
        :default-active="activePath"
        class="menu"
        router
        background-color="transparent"
        text-color="#9fb2d9"
        active-text-color="#dcecff"
      >
        <el-menu-item v-for="item in menuItems" :key="item.path" :index="item.path">
          <span>{{ item.label }}</span>
        </el-menu-item>
      </el-menu>
    </el-aside>

    <el-container>
      <el-header class="topbar">
        <div class="topbar-title">{{ pageTitle }}</div>
        <div class="topbar-user">当前用户：admin</div>
      </el-header>
      <el-main class="content-area">
        <router-view />
      </el-main>
    </el-container>
  </el-container>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import { useRoute } from 'vue-router'

const route = useRoute()

const menuItems = [
  { path: '/image-detect', label: '图片检测' },
  { path: '/video-detect', label: '视频检测' },
  { path: '/camera-detect', label: '摄像头检测' },
  { path: '/image-records', label: '图片识别记录' },
  { path: '/video-records', label: '视频识别记录' },
  { path: '/users', label: '用户管理' },
  { path: '/profile', label: '个人中心' },
]

const activePath = computed(() => route.path)
const pageTitle = computed(() => (route.meta.title as string) || '果蔬识别系统')
</script>

<style scoped>
.app-shell {
  min-height: 100vh;
}

.sidebar {
  background: linear-gradient(180deg, #0f172a 0%, #111827 44%, #1e293b 100%);
  color: #e5efff;
  border-right: 1px solid rgba(148, 163, 184, 0.2);
}

.logo-wrap {
  height: 68px;
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 0 18px;
  border-bottom: 1px solid rgba(148, 163, 184, 0.16);
}

.logo-mark {
  width: 26px;
  height: 26px;
  border-radius: 8px;
  background: linear-gradient(135deg, #2dd4bf, #38bdf8);
  box-shadow: 0 8px 24px rgba(56, 189, 248, 0.4);
}

.logo-title {
  font-size: 16px;
  font-weight: 700;
  letter-spacing: 0.3px;
}

.menu {
  border-right: none;
  padding-top: 12px;
}

.menu :deep(.el-menu-item) {
  margin: 4px 10px;
  border-radius: 10px;
  height: 44px;
}

.menu :deep(.el-menu-item.is-active) {
  background: linear-gradient(90deg, #2563eb, #3b82f6);
  color: #eff6ff;
}

.topbar {
  height: 62px;
  background: #f8fbff;
  border-bottom: 1px solid #dfe7f3;
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0 22px;
}

.topbar-title {
  font-size: 19px;
  font-weight: 700;
  color: #102a43;
}

.topbar-user {
  color: #4b5d79;
  font-size: 14px;
}

.content-area {
  background: linear-gradient(180deg, #f4f7fb 0%, #eef3f8 100%);
  padding: 18px;
}
</style>
