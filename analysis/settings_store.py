"""Save the app's settings in browser cookies so they survive an app restart.

The deployed app's disk is wiped when it restarts, so the settings live in the
user's browser instead: the session-state values are packed into compressed
JSON and split over a few cookies (each cookie holds about 4 KB).
"""

import base64
import json
import zlib
from datetime import date, datetime

COOKIE_PREFIX = "bt_settings_"
CHUNK_SIZE = 3000
MAX_CHUNKS = 20


def _encode(value):
    if isinstance(value, bool) or value is None or isinstance(value, (int, float, str)):
        return value
    if isinstance(value, datetime):
        return {"__dt": value.isoformat()}
    if isinstance(value, date):
        return {"__d": value.isoformat()}
    if isinstance(value, tuple):
        return {"__t": [_encode(item) for item in value]}
    if isinstance(value, list):
        return [_encode(item) for item in value]
    if isinstance(value, dict):
        if all(isinstance(key, str) for key in value):
            return {key: _encode(item) for key, item in value.items()}
        if all(isinstance(key, int) and not isinstance(key, bool) for key in value):
            return {"__i": [[key, _encode(item)] for key, item in value.items()]}
    raise TypeError(f"Cannot save a {type(value).__name__}")


def _decode(value):
    if isinstance(value, list):
        return [_decode(item) for item in value]
    if isinstance(value, dict):
        if set(value) == {"__dt"}:
            return datetime.fromisoformat(value["__dt"])
        if set(value) == {"__d"}:
            return date.fromisoformat(value["__d"])
        if set(value) == {"__t"}:
            return tuple(_decode(item) for item in value["__t"])
        if set(value) == {"__i"}:
            return {int(key): _decode(item) for key, item in value["__i"]}
        return {key: _decode(item) for key, item in value.items()}
    return value


def settings_snapshot(state, skip=lambda key: False):
    """The saveable part of ``state``: string keys not skipped, with values that can be saved (data frames cannot)."""

    snapshot = {}
    for key in sorted(key for key in state.keys() if isinstance(key, str)):
        if key.startswith("_") or skip(key):
            continue
        try:
            snapshot[key] = _encode(state[key])
        except TypeError:
            continue
    return snapshot


def pack(snapshot):
    """Snapshot -> list of cookie-sized text chunks; [] when it is too large to save."""

    raw = json.dumps(snapshot, separators=(",", ":"), sort_keys=True).encode("utf-8")
    text = base64.urlsafe_b64encode(zlib.compress(raw, 9)).decode("ascii")
    chunks = [text[start:start + CHUNK_SIZE] for start in range(0, len(text), CHUNK_SIZE)]
    return chunks if len(chunks) <= MAX_CHUNKS else []


def unpack(cookies):
    """Cookies (name -> value) -> saved settings; {} when none are saved or they cannot be read."""

    try:
        count = int(cookies.get(f"{COOKIE_PREFIX}n", 0))
        if not 0 < count <= MAX_CHUNKS:
            return {}
        text = "".join(cookies[f"{COOKIE_PREFIX}{index}"] for index in range(count))
        settings = json.loads(zlib.decompress(base64.urlsafe_b64decode(text.encode("ascii"))))
        return {key: _decode(value) for key, value in settings.items()} if isinstance(settings, dict) else {}
    except Exception:
        return {}


def cookie_script(chunks, max_age_days=365):
    """JavaScript that writes the chunks to cookies and deletes leftover chunks of an older, longer save."""

    max_age = max_age_days * 24 * 3600
    lines = [f'document.cookie = "{COOKIE_PREFIX}{index}={chunk}; path=/; max-age={max_age}; SameSite=Lax";'
             for index, chunk in enumerate(chunks)]
    lines.append(f'document.cookie = "{COOKIE_PREFIX}n={len(chunks)}; path=/; max-age={max_age}; SameSite=Lax";')
    lines += [f'document.cookie = "{COOKIE_PREFIX}{index}=; path=/; max-age=0";'
              for index in range(len(chunks), MAX_CHUNKS)]
    return "<script>\n" + "\n".join(lines) + "\n</script>"
