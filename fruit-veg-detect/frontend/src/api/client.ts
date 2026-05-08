import axios from 'axios'

const defaultApiBaseUrl = (() => {
  if (typeof window === 'undefined') return 'http://localhost:8000'
  if (window.location.protocol === 'http:' || window.location.protocol === 'https:') {
    return `${window.location.protocol}//${window.location.hostname}:8000`
  }
  return 'http://localhost:8000'
})()

export const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL || defaultApiBaseUrl).replace(/\/$/, '')

const client = axios.create({
  baseURL: API_BASE_URL,
  timeout: 120000,
})

export const toFileUrl = (path?: string | null): string => {
  if (!path) return ''
  if (path.startsWith('http://') || path.startsWith('https://')) return path
  return `${API_BASE_URL}${path}`
}

const readDetail = (detail: unknown): string => {
  if (typeof detail === 'string') return detail
  if (Array.isArray(detail)) {
    return detail
      .map((item) => {
        if (typeof item === 'string') return item
        if (item && typeof item === 'object' && 'msg' in item && typeof item.msg === 'string') {
          return item.msg
        }
        try {
          return JSON.stringify(item)
        } catch {
          return ''
        }
      })
      .filter(Boolean)
      .join('; ')
  }
  if (detail && typeof detail === 'object') {
    try {
      return JSON.stringify(detail)
    } catch {
      return ''
    }
  }
  return ''
}

export const getApiErrorMessage = (error: unknown, fallback: string): string => {
  if (axios.isAxiosError(error)) {
    const detail = readDetail(error.response?.data?.detail)
    if (detail) return detail
    if (error.message) return error.message
  }
  if (error instanceof Error && error.message) return error.message
  return fallback
}

export default client
