/**
 * NotificationBell component - displays a bell icon with unread count badge
 * and a dropdown showing recent notifications.
 */
import { useState, useRef, useEffect, type CSSProperties } from 'react';
import { useNavigate } from 'react-router-dom';
import { useNotifications } from '../../context/NotificationContext';
import type { Notification } from '../../types/notification';

export function NotificationBell() {
  const { notifications, unreadCount, markRead, markAllRead } = useNotifications();
  const [isOpen, setIsOpen] = useState(false);
  const dropdownRef = useRef<HTMLDivElement>(null);
  const navigate = useNavigate();

  // Close dropdown when clicking outside
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target as Node)) {
        setIsOpen(false);
      }
    };
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  const handleNotificationClick = (notification: Notification) => {
    markRead(notification.id);
    if (notification.entity_type === 'generation_job' && notification.entity_id) {
      navigate(`/generator?job=${notification.entity_id}`);
      setIsOpen(false);
    }
  };

  const formatTime = (dateStr: string) => {
    const date = new Date(dateStr);
    const now = new Date();
    const diff = now.getTime() - date.getTime();
    const minutes = Math.floor(diff / 60000);
    const hours = Math.floor(minutes / 60);
    const days = Math.floor(hours / 24);

    if (minutes < 1) return 'just now';
    if (minutes < 60) return `${minutes}m ago`;
    if (hours < 24) return `${hours}h ago`;
    return `${days}d ago`;
  };

  const getNotificationIcon = (type: string) => {
    switch (type) {
      case 'generation_queued':
        return '⏳';
      case 'generation_started':
        return '🔄';
      case 'generation_completed':
        return '✅';
      case 'generation_failed':
        return '❌';
      default:
        return '📢';
    }
  };

  return (
    <div ref={dropdownRef} style={styles.container}>
      <button
        onClick={() => setIsOpen(!isOpen)}
        style={styles.bellButton}
        aria-label={`Notifications (${unreadCount} unread)`}
      >
        <svg
          width="20"
          height="20"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          strokeWidth="2"
          strokeLinecap="round"
          strokeLinejoin="round"
        >
          <path d="M18 8A6 6 0 0 0 6 8c0 7-3 9-3 9h18s-3-2-3-9" />
          <path d="M13.73 21a2 2 0 0 1-3.46 0" />
        </svg>
        {unreadCount > 0 && (
          <span style={styles.badge}>
            {unreadCount > 99 ? '99+' : unreadCount}
          </span>
        )}
      </button>

      {isOpen && (
        <div style={styles.dropdown}>
          <div style={styles.dropdownHeader}>
            <span style={styles.dropdownTitle}>Notifications</span>
            {unreadCount > 0 && (
              <button onClick={markAllRead} style={styles.markAllButton}>
                Mark all read
              </button>
            )}
          </div>

          <div style={styles.notificationList}>
            {notifications.length === 0 ? (
              <div style={styles.emptyState}>No notifications</div>
            ) : (
              notifications.slice(0, 10).map((notification) => (
                <div
                  key={notification.id}
                  onClick={() => handleNotificationClick(notification)}
                  style={{
                    ...styles.notificationItem,
                    backgroundColor: notification.read ? 'transparent' : 'rgba(37, 99, 235, 0.1)',
                  }}
                >
                  <span style={styles.notificationIcon}>
                    {getNotificationIcon(notification.type)}
                  </span>
                  <div style={styles.notificationContent}>
                    <div style={styles.notificationTitle}>{notification.title}</div>
                    <div style={styles.notificationMessage}>{notification.message}</div>
                    <div style={styles.notificationTime}>
                      {formatTime(notification.created_at)}
                    </div>
                  </div>
                </div>
              ))
            )}
          </div>
        </div>
      )}
    </div>
  );
}

const styles: Record<string, CSSProperties> = {
  container: {
    position: 'relative',
    display: 'inline-flex',
    alignItems: 'center',
  },
  bellButton: {
    position: 'relative',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    width: '36px',
    height: '36px',
    background: 'transparent',
    border: 'none',
    borderRadius: '8px',
    color: '#aaa',
    cursor: 'pointer',
    transition: 'background 0.2s, color 0.2s',
  },
  badge: {
    position: 'absolute',
    top: '2px',
    right: '2px',
    minWidth: '16px',
    height: '16px',
    padding: '0 4px',
    background: '#ef4444',
    borderRadius: '8px',
    fontSize: '10px',
    fontWeight: 600,
    color: '#fff',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
  },
  dropdown: {
    position: 'absolute',
    top: '100%',
    right: 0,
    width: '320px',
    marginTop: '8px',
    background: '#1a1a2e',
    borderRadius: '8px',
    border: '1px solid #333',
    boxShadow: '0 10px 40px rgba(0, 0, 0, 0.5)',
    zIndex: 1000,
    overflow: 'hidden',
  },
  dropdownHeader: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    padding: '12px 16px',
    borderBottom: '1px solid #333',
  },
  dropdownTitle: {
    fontSize: '13px',
    fontWeight: 600,
    color: '#fff',
  },
  markAllButton: {
    background: 'none',
    border: 'none',
    color: '#60a5fa',
    fontSize: '11px',
    cursor: 'pointer',
    padding: '4px 8px',
    borderRadius: '4px',
  },
  notificationList: {
    maxHeight: '400px',
    overflowY: 'auto',
  },
  notificationItem: {
    display: 'flex',
    gap: '12px',
    padding: '12px 16px',
    cursor: 'pointer',
    borderBottom: '1px solid #222',
    transition: 'background 0.2s',
  },
  notificationIcon: {
    fontSize: '18px',
    flexShrink: 0,
  },
  notificationContent: {
    flex: 1,
    minWidth: 0,
  },
  notificationTitle: {
    fontSize: '12px',
    fontWeight: 600,
    color: '#fff',
    marginBottom: '2px',
  },
  notificationMessage: {
    fontSize: '11px',
    color: '#aaa',
    overflow: 'hidden',
    textOverflow: 'ellipsis',
    whiteSpace: 'nowrap',
  },
  notificationTime: {
    fontSize: '10px',
    color: '#666',
    marginTop: '4px',
  },
  emptyState: {
    padding: '32px 16px',
    textAlign: 'center',
    color: '#666',
    fontSize: '12px',
  },
};
