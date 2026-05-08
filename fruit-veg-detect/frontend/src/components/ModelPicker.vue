<template>
  <div class="model-picker">
    <el-button class="model-picker__toggle" plain @click="expanded = !expanded">
      识别类型：{{ currentLabel }}（{{ expanded ? '收起' : '展开' }}）
    </el-button>

    <div v-if="expanded" class="model-picker__options">
      <el-button
        v-for="option in options"
        :key="option.value"
        :type="modelValue === option.value ? 'primary' : 'default'"
        plain
        @click="selectModel(option.value)"
      >
        {{ option.label }}
      </el-button>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed, ref } from 'vue'

import type { ModelKey } from '../api/detect'

const props = defineProps<{
  modelValue: ModelKey
}>()

const emit = defineEmits<{
  (event: 'update:modelValue', value: ModelKey): void
}>()

const expanded = ref(false)

const options: Array<{ label: string; value: ModelKey }> = [
  { label: '水果', value: 'fruit' },
  { label: '蔬菜', value: 'vegetable' },
]

const currentLabel = computed(() => {
  return options.find((option) => option.value === props.modelValue)?.label || '水果'
})

const selectModel = (value: ModelKey) => {
  emit('update:modelValue', value)
  expanded.value = false
}
</script>

<style scoped>
.model-picker {
  margin-bottom: 14px;
}

.model-picker__toggle {
  width: 100%;
  justify-content: flex-start;
}

.model-picker__options {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 8px;
  margin-top: 8px;
}

.model-picker__options :deep(.el-button) {
  margin-left: 0;
}
</style>
