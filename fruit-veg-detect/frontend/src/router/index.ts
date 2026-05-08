import { createRouter, createWebHistory, type RouteRecordRaw } from 'vue-router'

import AppLayout from '../layout/AppLayout.vue'
import CameraDetect from '../pages/CameraDetect.vue'
import ImageDetect from '../pages/ImageDetect.vue'
import ImageRecords from '../pages/ImageRecords.vue'
import Profile from '../pages/Profile.vue'
import Users from '../pages/Users.vue'
import VideoDetect from '../pages/VideoDetect.vue'
import VideoRecords from '../pages/VideoRecords.vue'

const routes: RouteRecordRaw[] = [
  {
    path: '/',
    component: AppLayout,
    redirect: '/image-detect',
    children: [
      { path: 'image-detect', component: ImageDetect, meta: { title: '图片检测' } },
      { path: 'video-detect', component: VideoDetect, meta: { title: '视频检测' } },
      { path: 'camera-detect', component: CameraDetect, meta: { title: '摄像头检测' } },
      { path: 'image-records', component: ImageRecords, meta: { title: '图片识别记录' } },
      { path: 'video-records', component: VideoRecords, meta: { title: '视频识别记录' } },
      { path: 'users', component: Users, meta: { title: '用户管理' } },
      { path: 'profile', component: Profile, meta: { title: '个人中心' } },
    ],
  },
]

const router = createRouter({
  history: createWebHistory(),
  routes,
})

export default router
