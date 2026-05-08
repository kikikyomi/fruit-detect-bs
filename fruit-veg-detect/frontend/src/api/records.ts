import client from './client'

export interface RecordItem {
  id: number
  created_at: string
  file_name: string
  detection_count: number
  input_url?: string
  output_url?: string
  stream_url?: string | null
  summary?: {
    keyframes?: string[]
    keyframe_details?: Array<{
      frame_index: number
      image_url: string
      tracks: Array<{
        track_id?: number
        cls_name: string
        conf: number
        x1: number
        y1: number
        x2: number
        y2: number
      }>
    }>
    detections?: number
    class_stats?: Record<string, number>
    frame_interval?: number
    sampled_frames?: number
    total_frames?: number
    input_fps?: number
    processing_seconds?: number
    processing_fps?: number
    detection_fps?: number
    target_fps_met?: boolean
    unique_tracks?: number
    track_class_stats?: Record<string, number>
    tracker?: string
    deepsort_enabled?: boolean
    tracker_n_init?: number
    tracker_max_time_since_update?: number
    track_summaries?: Array<{
      track_id: number
      cls_name: string
      cls_id: number
      first_frame: number
      last_frame: number
      frames_seen: number
      path_length: number
    }>
  }
}

export interface RecordListResponse {
  page: number
  page_size: number
  total: number
  items: RecordItem[]
}

export const getImageRecords = async (page = 1, pageSize = 10): Promise<RecordListResponse> => {
  const { data } = await client.get<RecordListResponse>('/api/records/images', {
    params: { page, page_size: pageSize },
  })
  return data
}

export const getVideoRecords = async (page = 1, pageSize = 10): Promise<RecordListResponse> => {
  const { data } = await client.get<RecordListResponse>('/api/records/videos', {
    params: { page, page_size: pageSize },
  })
  return data
}

export const deleteRecord = async (recordId: number): Promise<{ ok: boolean }> => {
  const { data } = await client.delete<{ ok: boolean }>(`/api/records/${recordId}`)
  return data
}
