"""Other-match data with per-frame labels: the IDSSE dataset (Bassek et al., 2025, CC-BY 4.0).

Seven Bundesliga 2022/23 matches with official DFL position data at 25 fps. Every frame carries
the DFL's own BallStatus (alive/dead) and BallPossession (home/away). These are labels on *other*
games, which RULES.md allows in any amount; they calibrate our rules, never the provided game.

Files are downloaded into anishwar/external/idsse (gitignored; see README for the source) and read
with kloppy. Each match is reduced once to compact arrays in anishwar/external/arrays:

    ball_xy      (n, 2)   metres, NaN where missing
    alive        (n,)     DFL BallStatus == alive
    owner        (n,)     0 home, 1 away, -1 none
    players_xy   (n, p, 2) metres, NaN where the player is not on the pitch
    team         (p,)     0 home, 1 away
    period       (n,)     1 or 2

Nothing here prints a frame; callers report aggregates only.
"""

import os

import numpy as np

from . import config

MATCHES = ("J03WMX", "J03WN1", "J03WOH", "J03WOY", "J03WPY", "J03WQQ", "J03WR9")
EXTERNAL_DIR = os.path.join(config.WORKSPACE, "external")
IDSSE_DIR = os.path.join(EXTERNAL_DIR, "idsse")
ARRAYS_DIR = os.path.join(EXTERNAL_DIR, "arrays")


def _paths(match_id):
    from kloppy._providers.sportec import get_IDSSE_url
    return {kind: os.path.join(IDSSE_DIR, get_IDSSE_url(match_id, kind).rsplit("/", 1)[1])
            for kind in ("meta", "tracking")}


def convert(match_id):
    """Read one match with kloppy and write its arrays. Returns the output path."""
    from kloppy import sportec
    from kloppy.domain import BallState, Ground

    paths = _paths(match_id)
    ds = sportec.load_tracking(meta_data=paths["meta"], raw_data=paths["tracking"])
    md = ds.metadata
    length, width = md.pitch_dimensions.pitch_length, md.pitch_dimensions.pitch_width
    players = [p for t in md.teams for p in t.players]
    index = {p.player_id: i for i, p in enumerate(players)}
    team = np.array([0 if p.team.ground == Ground.HOME else 1 for p in players], np.int8)

    n = len(ds.frames)
    ball = np.full((n, 2), np.nan, np.float32)
    alive = np.zeros(n, bool)
    owner = np.full(n, -1, np.int8)
    period = np.zeros(n, np.int8)
    pxy = np.full((n, len(players), 2), np.nan, np.float32)
    for i, f in enumerate(ds.frames):
        if f.ball_coordinates is not None and f.ball_coordinates.x is not None:
            ball[i] = (f.ball_coordinates.x * length, f.ball_coordinates.y * width)
        alive[i] = f.ball_state == BallState.ALIVE
        if f.ball_owning_team is not None:
            owner[i] = 0 if f.ball_owning_team.ground == Ground.HOME else 1
        period[i] = f.period.id
        for player, data in f.players_data.items():
            c = data.coordinates
            if c is not None:
                pxy[i, index[player.player_id]] = (c.x * length, c.y * width)

    os.makedirs(ARRAYS_DIR, exist_ok=True)
    out = os.path.join(ARRAYS_DIR, "{}.npz".format(match_id))
    np.savez_compressed(out, ball_xy=ball, alive=alive, owner=owner, players_xy=pxy, team=team,
                        period=period, pitch=np.array([length, width], np.float32))
    return out


def load(match_id):
    path = os.path.join(ARRAYS_DIR, "{}.npz".format(match_id))
    if not os.path.exists(path):
        convert(match_id)
    with np.load(path) as data:
        return {k: data[k] for k in data.files}
