/**
 * API service for user notifications.
 */
import { apiV1 } from './api-v1';
import type { Notification } from '../types/notification';

export const notificationApi = {
  /** List notifications, most recent first, unread prioritized */
  async list(params?: {
    limit?: number;
    offset?: number;
    unread_only?: boolean;
  }): Promise<Notification[]> {
    const { data } = await apiV1.get<Notification[]>('/notifications/', { params });
    return data;
  },

  /** Get the count of unread notifications */
  async getUnreadCount(): Promise<number> {
    const { data } = await apiV1.get<{ count: number }>('/notifications/unread-count');
    return data.count;
  },

  /** Get a specific notification */
  async get(notificationId: string): Promise<Notification> {
    const { data } = await apiV1.get<Notification>(`/notifications/${notificationId}`);
    return data;
  },

  /** Mark a notification as read */
  async markRead(notificationId: string): Promise<Notification> {
    const { data } = await apiV1.post<Notification>(`/notifications/${notificationId}/read`);
    return data;
  },

  /** Mark all notifications as read */
  async markAllRead(): Promise<number> {
    const { data } = await apiV1.post<{ updated: number }>('/notifications/read-all');
    return data.updated;
  },

  /** Delete a notification */
  async delete(notificationId: string): Promise<void> {
    await apiV1.delete(`/notifications/${notificationId}`);
  },
};
