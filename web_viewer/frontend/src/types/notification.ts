/**
 * Types for user notifications.
 */

export interface Notification {
  id: string;
  type: string;
  title: string;
  message: string;
  entity_type?: string;
  entity_id?: string;
  read: boolean;
  created_at: string;
  created_by: string;
}
