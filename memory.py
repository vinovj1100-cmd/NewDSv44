"""Memory module for Neural Fulfillment Platform v3.0
Handles settings, aliases, and template suggestions."""

from db import connect, save_memory, get_memory


def get_setting(key, default=""):
    """Retrieve a setting value with default fallback."""
    val = get_memory(key)
    return val if val is not None else default


def set_setting(key, value):
    """Save a setting value."""
    save_memory(key, value)


def suggest_alias(text):
    """Suggest an alias for a given text string."""
    if not text:
        return None
    with connect() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT value FROM memory WHERE key LIKE ? LIMIT 1", (f"alias:%{text.lower().strip()}%",))
        row = cursor.fetchone()
        return row["value"] if row else None


def suggest_template(text):
    """Suggest a template match for a given text string."""
    if not text:
        return None
    with connect() as conn:
        cursor = conn.cursor()
        # Fuzzy match on raw_title
        cursor.execute("""
            SELECT standard_title FROM templates 
            WHERE raw_title LIKE ? OR ? LIKE '%' || raw_title || '%'
            ORDER BY created_at DESC LIMIT 1
        """, (f"%{text}%", text))
        row = cursor.fetchone()
        return row["standard_title"] if row else None


def upsert_alias(source, target):
    """Save or update an alias mapping."""
    save_memory(f"alias:{source.lower().strip()}", target)
