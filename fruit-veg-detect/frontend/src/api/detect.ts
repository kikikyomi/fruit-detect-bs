import client from './client'

export type ModelKey = 'fruit' | 'vegetable'

export interface DetectBox {
  x1: number
  y1: number
  x2: number
  y2: number
  conf: number
  cls_id: number
  cls_name: string
  track_id?: number
  time_since_update?: number
  raw_bbox?: number[]
  smoothed_bbox?: number[]
}

export interface ImageDetectResult {
  boxes: DetectBox[]
  image_size: {
    w: number
    h: number
  }
  model_key?: ModelKey
  imgsz?: number
  device?: string
  half?: boolean
}

export interface ImageDetectResponse {
  record_id: number | null
  result: ImageDetectResult
  annotated_image_url: string
  input_image_url: string
  saved: boolean
}

export interface VideoDetectResponse {
  task_id?: string
  status?: string
  progress?: number
  current_frame?: number
  total_frames?: number
  fps?: number
  message?: string
  record_id: number | null
  saved: boolean
  summary: VideoSummary
  outputs: {
    video_url?: string | null
    stream_url?: string | null
    keyframes: string[]
    keyframe_details?: Array<{
      frame_index: number
      image_url: string
      tracks: DetectBox[]
    }>
  }
  tracking: {
    track_summaries: Array<{
      track_id: number
      cls_name: string
      cls_id: number
      first_frame: number
      last_frame: number
      frames_seen: number
      path_length: number
    }>
    trajectories: Record<string, Array<{ x?: number; y?: number; break?: boolean }>>
  }
}

export interface VideoSummary {
    total_frames: number
    sampled_frames: number
    detections: number
    input_fps: number
    processing_seconds: number
    processing_fps: number
    detection_fps: number
    target_fps_met: boolean
    analysis_seconds?: number
    render_seconds?: number
    class_stats: Record<string, number>
    frame_interval: number
    unique_tracks: number
    track_class_stats: Record<string, number>
    tracker: string
    deepsort_enabled: boolean
    tracker_n_init: number
    tracker_max_time_since_update: number
    model_key?: ModelKey
    saved: boolean
    original_total_frames?: number
    processed_frames?: number
    original_fps?: number
    output_fps?: number
    original_resolution?: string
    output_resolution?: string
    frame_skip?: number
    total_detections?: number
    average_processing_fps?: number
    average_yolo_ms?: number
    average_deepsort_ms?: number
    average_draw_ms?: number
    average_read_ms?: number
    average_write_ms?: number
    total_seconds?: number
    class_counts?: Record<string, number>
    track_class_counts?: Record<string, number>
    show_stats?: boolean
    imgsz?: number
    smoothing_enabled?: boolean
    smoothing_alpha?: number
    max_center_jump?: number
}

export interface VideoTaskResponse {
  task_id: string
  status: 'waiting' | 'running' | 'finished' | 'failed'
  progress: number
  current_frame: number
  total_frames: number
  fps: number
  message: string
  record_id?: number | null
  output_video_url?: string | null
  result_csv_url?: string | null
  result_json_url?: string | null
  summary_url?: string | null
  summary?: Partial<VideoSummary> | null
  saved?: boolean
  error?: string
}

export interface VideoDetectOptions {
  file: File
  conf: number
  iou: number
  imgsz: number
  frameSkip: number
  enableDeepSort: boolean
  trackerBackend: string
  modelKey?: ModelKey
  outputWidth: number
  outputHeight: number
  keepOriginalResolution: boolean
  resizeOutput: boolean
  maxAge: number
  trackerMaxTimeSinceUpdate?: number | string | null
  nInit: number
  maxIouDistance: number
  maxCosineDistance: number
  nnBudget: number
  trailLength: number
  smoothWindow: number
  smoothingEnabled: boolean
  smoothAlpha: number
  minBoxArea: number
  maxCenterJump: number
  saveCsv: boolean
  saveJson: boolean
  saveVideo: boolean
  showStats: boolean
  startTime?: number | null
  endTime?: number | null
}

export interface CameraFrameResponse {
  session_id: string
  frame_index?: number
  result: ImageDetectResult
  tracking: {
    trajectories: Record<string, Array<{ x?: number; y?: number; break?: boolean }>>
    track_summaries: Array<{
      track_id: number
      cls_name: string
      cls_id: number
      first_frame: number
      last_frame: number
      frames_seen: number
      path_length: number
    }>
    tracker: string
    deepsort_enabled: boolean
    tracker_config?: Record<string, number>
  }
  performance?: {
    total_ms: number
    yolo_ms: number
    deepsort_ms: number
  }
  saved?: boolean
}

export const imageDetect = async (
  file: File,
  conf: number,
  iou: number,
  modelKey: ModelKey = 'fruit',
): Promise<ImageDetectResponse> => {
  const form = new FormData()
  form.append('file', file)
  form.append('conf', String(conf))
  form.append('iou', String(iou))
  form.append('model_key', modelKey)

  const { data } = await client.post<ImageDetectResponse>('/api/detect/image', form)
  return data
}

export const videoDetect = async (
  file: File,
  conf: number,
  iou: number,
  frameInterval?: number,
  trackerBackend?: string,
  trackerMaxTimeSinceUpdate?: number,
  modelKey: ModelKey = 'fruit',
): Promise<VideoDetectResponse> => {
  const form = new FormData()
  form.append('file', file)
  form.append('conf', String(conf))
  form.append('iou', String(iou))
  form.append('model_key', modelKey)
  if (typeof frameInterval === 'number') {
    form.append('frame_interval', String(frameInterval))
  }
  if (trackerBackend) {
    form.append('tracker_backend', trackerBackend)
  }
  if (typeof trackerMaxTimeSinceUpdate === 'number') {
    form.append('tracker_max_time_since_update', String(trackerMaxTimeSinceUpdate))
  }

  const { data } = await client.post<VideoDetectResponse>('/api/detect/video', form, {
    timeout: 0,
  })
  return data
}

export const createCameraSession = async (): Promise<{ session_id: string }> => {
  const { data } = await client.post<{ session_id: string }>('/api/detect/camera/session')
  return data
}

export const resetCameraSession = async (sessionId: string): Promise<{ ok: boolean }> => {
  const { data } = await client.post<{ ok: boolean }>('/api/detect/camera/reset', {
    session_id: sessionId,
  })
  return data
}

export const cameraFrameDetect = async (
  file: File,
  conf: number,
  iou: number,
  sessionId?: string,
  trackerBackend?: string,
  trackerMaxTimeSinceUpdate?: number,
  modelKey: ModelKey = 'fruit',
  imgsz?: number,
): Promise<CameraFrameResponse> => {
  const form = new FormData()
  form.append('file', file)
  form.append('conf', String(conf))
  form.append('iou', String(iou))
  form.append('model_key', modelKey)
  if (sessionId) {
    form.append('session_id', sessionId)
  }
  if (trackerBackend) {
    form.append('tracker_backend', trackerBackend)
  }
  if (typeof trackerMaxTimeSinceUpdate === 'number') {
    form.append('tracker_max_time_since_update', String(trackerMaxTimeSinceUpdate))
  }
  if (typeof imgsz === 'number') {
    form.append('imgsz', String(imgsz))
  }

  const { data } = await client.post<CameraFrameResponse>('/api/detect/camera/frame', form)
  return data
}

const appendFiniteNumber = (
  form: FormData,
  key: string,
  value: number | string | null | undefined,
) => {
  if (value === undefined || value === null) {
    return
  }
  const normalizedValue = typeof value === 'string' ? value.trim() : value
  if (
    normalizedValue === ''
    || String(normalizedValue).toLowerCase() === 'none'
    || String(normalizedValue).toLowerCase() === 'null'
    || String(normalizedValue).toLowerCase() === 'undefined'
  ) {
    return
  }
  const numericValue = Number(normalizedValue)
  if (Number.isFinite(numericValue)) {
    form.append(key, String(numericValue))
  }
}

export const startVideoDetectTask = async (options: VideoDetectOptions): Promise<VideoTaskResponse> => {
  const form = new FormData()
  form.append('file', options.file)
  form.append('conf', String(options.conf))
  form.append('iou', String(options.iou))
  form.append('imgsz', String(options.imgsz))
  form.append('frame_skip', String(options.frameSkip))
  form.append('enable_deepsort', String(options.enableDeepSort))
  form.append('tracker_backend', options.trackerBackend)
  form.append('model_key', options.modelKey || 'fruit')
  form.append('output_width', String(options.outputWidth))
  form.append('output_height', String(options.outputHeight))
  form.append('keep_original_resolution', String(options.keepOriginalResolution))
  form.append('resize_output', String(options.resizeOutput))
  form.append('max_age', String(options.maxAge))
  appendFiniteNumber(form, 'tracker_max_time_since_update', options.trackerMaxTimeSinceUpdate)
  form.append('n_init', String(options.nInit))
  form.append('max_iou_distance', String(options.maxIouDistance))
  form.append('max_cosine_distance', String(options.maxCosineDistance))
  form.append('nn_budget', String(options.nnBudget))
  form.append('trail_length', String(options.trailLength))
  form.append('smooth_window', String(options.smoothWindow))
  form.append('smoothing_enabled', String(options.smoothingEnabled))
  form.append('smooth_alpha', String(options.smoothAlpha))
  form.append('min_box_area', String(options.minBoxArea))
  form.append('max_center_jump', String(options.maxCenterJump))
  form.append('save_csv', String(options.saveCsv))
  form.append('save_json', String(options.saveJson))
  form.append('save_video', String(options.saveVideo))
  form.append('show_stats', String(options.showStats))
  if (typeof options.startTime === 'number') {
    form.append('start_time', String(options.startTime))
  }
  if (typeof options.endTime === 'number') {
    form.append('end_time', String(options.endTime))
  }

  const { data } = await client.post<VideoTaskResponse>('/api/detect/video', form, {
    timeout: 0,
  })
  return data
}

export const getVideoTask = async (taskId: string): Promise<VideoTaskResponse> => {
  const { data } = await client.get<VideoTaskResponse>(`/api/detect/video/tasks/${taskId}`)
  return data
}
