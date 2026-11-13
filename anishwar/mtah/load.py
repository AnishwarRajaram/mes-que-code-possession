"""Parse the provided ATD and Smart Tagging once, cache them, and load them back fast.

Tracking cache, one per source (cache/<source>_atd.npz):

    ball     float32 (n_frames, 3)        x, y, speed       -- row i is video frame i
    players  float32 (n_frames, slots, 3) x, y, speed per track slot
    team     int8    (slots,)             0 = home (H), 1 = away (A), fixed per slot

x and y stay in the file's normalised 0-1 units, the same convention the homography uses;
to_metres() converts. Missing observations are NaN, as in the file. Row i is video frame i
because the ATD counts from 1 (starter/load_atd.py): video_frame = frame_id - 1.

Smart Tagging cache (cache/smart_tagging.json) is starter/load_smart_tagging.load() as is:
intervals already in video frames, end exclusive.

The caches are derived from the match data: gitignored, never shared, never printed row by row.
"""

import json
import os
import sys
import xml.etree.ElementTree as ET

import numpy as np

from . import config

sys.path.insert(0, os.path.join(config.REPO_ROOT, "starter"))
import load_smart_tagging  # noqa: E402  (starter, standard library only)

HOME, AWAY = 0, 1
FIELD_SEP, GROUP_SEP, VALUE_SEP = ":", ";", ","
CACHE_VERSION = 1


def _cache_path(name):
    return os.path.join(config.CACHE_DIR, name)


def _stamp(path):
    """Size and mtime of a source file, so a changed export invalidates its cache."""
    info = os.stat(path)
    return [info.st_size, int(info.st_mtime)]


def team_of_slots(meta_path):
    """Team of every track slot, from the ATD metadata. First team listed is home.

    Same reading as starter/baseline.py: the FIFA/EPTS convention, and DATA.md confirms
    Barcelona is home.
    """
    root = ET.parse(meta_path).getroot()
    team_ids = [team.get("id") for team in root.iter("Team")]
    if len(team_ids) != 2:
        raise ValueError("{}: expected 2 teams, found {}".format(meta_path, len(team_ids)))
    slots = [player.get("teamId") for player in root.iter("Player")]
    return np.array([HOME if team_id == team_ids[0] else AWAY for team_id in slots], np.int8)


def parse_atd(source="tactical"):
    """Stream the raw ATD into arrays. One pass; the slow step the cache exists to skip.

    Each line is "<frame_id>:<slot>;<slot>;...:<ball>", every group "x,y,speed". Rather than
    splitting groups one by one, the separators are unified and numpy parses the whole line,
    which reads NaN natively. A line with the wrong number of values is an error, not a skip:
    a silently dropped frame would shift nothing but would leave a hole we'd never notice.
    """
    paths = config.asset_paths(source)
    team = team_of_slots(paths["atd_meta"])
    n_slots = len(team)
    n_frames = config.games.source_frames(source, config.GAME)

    ball = np.full((n_frames, 3), np.nan, np.float32)
    players = np.full((n_frames, n_slots, 3), np.nan, np.float32)
    seen = np.zeros(n_frames, bool)
    n_values = 3 * (n_slots + 1)

    with open(paths["atd_raw"]) as handle:
        for line_no, line in enumerate(handle, 1):
            head, _, body = line.partition(FIELD_SEP)
            frame = int(head) - 1
            values = np.fromstring(
                body.replace(FIELD_SEP, VALUE_SEP).replace(GROUP_SEP, VALUE_SEP), sep=VALUE_SEP)
            if values.size != n_values:
                raise ValueError("{} line {}: {} values, expected {}".format(
                    os.path.basename(paths["atd_raw"]), line_no, values.size, n_values))
            if not 0 <= frame < n_frames or seen[frame]:
                raise ValueError("{} line {}: frame_id out of range or repeated".format(
                    os.path.basename(paths["atd_raw"]), line_no))
            players[frame] = values[:-3].reshape(n_slots, 3)
            ball[frame] = values[-3:]
            seen[frame] = True

    return {"ball": ball, "players": players, "team": team, "seen": seen}


def load_atd(source="tactical", rebuild=False):
    """Tracking arrays for a source, from the cache, building it first if needed."""
    path = _cache_path("{}_atd.npz".format(source))
    paths = config.asset_paths(source)
    stamp = _stamp(paths["atd_raw"]) + _stamp(paths["atd_meta"])

    if not rebuild and os.path.exists(path):
        with np.load(path) as cached:
            if (int(cached["version"]) == CACHE_VERSION
                    and cached["stamp"].tolist() == stamp):
                return {key: cached[key] for key in ("ball", "players", "team", "seen")}

    data = parse_atd(source)
    os.makedirs(config.CACHE_DIR, exist_ok=True)
    np.savez(path, version=CACHE_VERSION, stamp=np.array(stamp), **data)
    return data


def load_tagging(rebuild=False):
    """Smart Tagging intervals (tactical only), from the cache, building it first if needed."""
    path = _cache_path("smart_tagging.json")
    source = config.asset_paths("tactical")["smart_tagging"]
    stamp = _stamp(source)

    if not rebuild and os.path.exists(path):
        with open(path) as handle:
            cached = json.load(handle)
        if cached.get("version") == CACHE_VERSION and cached.get("stamp") == stamp:
            return cached["events"]

    events = load_smart_tagging.load(config.GAME)
    os.makedirs(config.CACHE_DIR, exist_ok=True)
    with open(path, "w") as handle:
        json.dump({"version": CACHE_VERSION, "stamp": stamp, "events": events}, handle)
    return events


def to_metres(xy):
    """Normalised (..., 2) pitch coordinates to metres, using games.py's measured pitch."""
    return xy * np.asarray(config.PITCH, np.float32)


def graded_mask(n_frames=None):
    """Boolean mask over video frames: True on the graded halves."""
    mask = np.zeros(n_frames or config.N_FRAMES, bool)
    for _, first, last in config.half_blocks():
        mask[first:last + 1] = True
    return mask
