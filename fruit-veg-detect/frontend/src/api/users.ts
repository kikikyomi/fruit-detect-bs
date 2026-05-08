import client from './client'

export interface UserItem {
  id: number
  username: string
  role: 'admin' | 'operator' | 'viewer'
}

export interface Profile {
  username: string
  role: string
  email: string
}

export const getUsers = async (): Promise<{ items: UserItem[] }> => {
  const { data } = await client.get<{ items: UserItem[] }>('/api/users')
  return data
}

export const createUser = async (payload: { username: string; role: 'admin' | 'operator' | 'viewer' }): Promise<{ item: UserItem }> => {
  const { data } = await client.post<{ item: UserItem }>('/api/users', payload)
  return data
}

export const getProfile = async (): Promise<Profile> => {
  const { data } = await client.get<Profile>('/api/profile')
  return data
}
