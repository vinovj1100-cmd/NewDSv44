"""Sync module for Neural Fulfillment Platform v3.0
Handles offline queue synchronization and online status."""

from db import connect, get_memory
from datetime import datetime


def enqueue_action(action_type, payload):
    """Add an action to the sync queue for later processing."""
    from db import enqueue_action as db_enqueue
    db_enqueue(action_type, payload)


def process_queue():
    """Process all pending sync queue items."""
    from db import process_queue as db_process
    return db_process()


def queue_status():
    """Get current queue status summary."""
    from db import queue_status as db_status
    return db_status()


def can_sync_now():
    """Check if online sync is enabled."""
    online = get_memory("online_access")
    return online.lower() == "true" if online else True
