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
}

export interface ImageDetectResult {
  boxes: DetectBox[]
  image_size: {
    w: number
    h: number
  }
  model_key?: ModelKey
}

export interface ImageDetectResponse {
  record_id: number | null
  result: ImageDetectResult
  annotated_image_url: string
  input_image_url: string
  saved: boolean
}

export interface VideoDetectResponse {
  record_id: number | null
  saved: boolean
  summary: {
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
  }
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
    trajectories: Record<string, Array<{ x: number; y: number }>>
  }
}

export interface CameraFrameResponse {
  session_id: string
  result: ImageDetectResult
  tracking: {
    trajectories: Record<string, Array<{ x: number; y: number }>>
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

  const { data } = await client.post<CameraFrameResponse>('/api/detect/camera/frame', form)
  return data
}
