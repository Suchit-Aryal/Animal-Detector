"""Event grouping: merge repeated detections of one class into one event (no Flask, no model)."""

from datetime import datetime


def epoch_of(ts: str) -> float:
    try:
        return datetime.fromisoformat(ts).timestamp()
    except ValueError:
        return 0.0


def group_events(rows: list, gap: float) -> list:
    """Merge animal detections of the same class in the same source into one event when they are
    no more than gap seconds apart. Uploads use clip time; cameras use wall-clock time."""
    def clock(r):
        return r["clip_time"] if r["clip_time"] is not None else epoch_of(r["ts"])
    events = []
    for r in sorted(rows, key=lambda r: (r["source"], r["class_name"], clock(r))):
        c = clock(r)
        last = events[-1] if events else None
        if last and last["source"] == r["source"] and last["class"] == r["class_name"] and c - last["_end"] <= gap:
            last["_end"] = c
            last["end_ts"] = r["ts"]
            last["count"] += 1
            last["tracks"].add(r["track_id"])
            if r["snapshot"] and r["confidence"] > last["best_conf"]:
                last.update(snapshot=r["snapshot"], best_conf=r["confidence"])
            last["max_conf"] = max(last["max_conf"], r["confidence"])
        else:
            events.append({"source": r["source"], "class": r["class_name"], "_start": c, "_end": c,
                           "start_ts": r["ts"], "end_ts": r["ts"], "count": 1, "tracks": {r["track_id"]},
                           "max_conf": r["confidence"], "best_conf": r["confidence"],
                           "snapshot": r["snapshot"], "clip": r["clip_time"] is not None})
    for e in events:
        e["duration"] = round(e["_end"] - e["_start"], 1)
        e["tracks"] = len({t for t in e["tracks"] if t is not None})
    return sorted(events, key=lambda e: e["start_ts"], reverse=True)
