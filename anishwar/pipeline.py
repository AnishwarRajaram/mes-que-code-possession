"""The one entry command for the pipeline.

    python anishwar/pipeline.py status          # environment, paths, which assets are present
    python anishwar/pipeline.py cache           # parse tracking + tagging once, report aggregates
    python anishwar/pipeline.py features        # per-frame features, full-match summary
    python anishwar/pipeline.py census CSV...   # label-free plausibility checks of submissions
    python anishwar/pipeline.py cues            # dead-ball cues measured on other matches (IDSSE)
    python anishwar/pipeline.py calibrate       # fit the dead-ball rules on IDSSE, held out
    python anishwar/pipeline.py challenger      # logistic regression challenger, same test
    python anishwar/pipeline.py engine          # score engine: named evidence, learned points
    python anishwar/pipeline.py possession      # fit the possession rules on IDSSE, held out
    python anishwar/pipeline.py changes         # fit the change detector and rule 6 on IDSSE
    python anishwar/pipeline.py deadcheck       # dead-ball rules on our match, label-free checks
    python anishwar/pipeline.py run [--name N]  # build the submission file (results/N.csv)
    python anishwar/pipeline.py eval CSV...     # correctness check on the public window

The constants are re-derived in order -- calibrate, then possession, then changes -- each stage on
the constants stored by the one before it. The IDSSE commands take --workers N to search folds side
by side (same result for any N).

Every subcommand prints aggregates only -- never rows of tracking, tagging or labels -- so its
output is safe to paste anywhere.

The public window (`eval`) checks that a file is correct: frames aligned, file well-formed, labels
meaning what we think. Under RULES.md it is not a tuning set and never chooses between versions;
`census` and data from other matches do that.
"""

import argparse
import os
import platform
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mtah import config  # noqa: E402


def status(_args):
    import numpy
    import pandas

    print("python {}  numpy {}  pandas {}".format(
        platform.python_version(), numpy.__version__, pandas.__version__))
    print("game {}  frames {}  fps {}  pitch {} x {} m".format(
        config.GAME, config.N_FRAMES, config.FPS, *config.PITCH))
    for half, first, last in config.half_blocks():
        print("half {}  graded video frames {}-{}  ({} frames)".format(
            half, first, last, last - first + 1))

    print("\nassets (existence only, nothing is opened):")
    for source in ("tactical", "broadcast"):
        for name, path in sorted(config.asset_paths(source).items()):
            if not isinstance(path, str):
                continue
            mark = "ok     " if os.path.exists(path) else "missing"
            print("  {}  {:<9} {:<14} {}".format(
                mark, source, name, os.path.relpath(path, config.REPO_ROOT)))

    print("\nworkspace outputs (gitignored):")
    for path in (config.CACHE_DIR, config.RESULTS_DIR, config.EVAL_OUT_DIR):
        state = "present" if os.path.isdir(path) else "not yet created"
        print("  {:<18} {}".format(os.path.relpath(path, config.REPO_ROOT), state))
    print("public ground truth: {}".format(
        "ok" if os.path.exists(config.PUBLIC_GT) else "missing"))


def cache(args):
    import time
    from collections import Counter

    import numpy as np

    from mtah import load

    for source in args.source:
        started = time.time()
        atd = load.load_atd(source, rebuild=args.rebuild)
        ball_seen = ~np.isnan(atd["ball"][:, 0])
        objects = (~np.isnan(atd["players"][:, :, 0])).sum(axis=1)
        team = atd["team"]

        print("{} ATD  ({:.1f} s)".format(source, time.time() - started))
        print("  frames {}  all present: {}".format(len(atd["seen"]), bool(atd["seen"].all())))
        print("  slots {}  home {}  away {}".format(
            len(team), int((team == load.HOME).sum()), int((team == load.AWAY).sum())))
        print("  ball present  {} / {} frames ({:.1f}%)".format(
            int(ball_seen.sum()), len(ball_seen), 100.0 * ball_seen.mean()))
        if source == "tactical":
            graded = load.graded_mask(len(ball_seen))
            print("  ball present  {} / {} graded frames ({:.1f}%)".format(
                int(ball_seen[graded].sum()), int(graded.sum()),
                100.0 * ball_seen[graded].mean()))
            print("  tracked objects on graded frames  mean {:.1f}  max {}".format(
                objects[graded].mean(), int(objects[graded].max())))

    if "tactical" in args.source:
        events = load.load_tagging(rebuild=args.rebuild)
        teams = Counter(event["tags"].get("Team") for event in events)
        print("\nSmart Tagging  {} instances".format(len(events)))
        print("  Team tags  " + "  ".join(
            "{}: {}".format(name, n) for name, n in teams.most_common()))
        check = load.load_smart_tagging.check_time_base(events)
        print("  kickoff time-base check agrees: {}".format(check["agrees"]))


def features(args):
    import time

    import numpy as np

    from mtah import features as features_module, load

    started = time.time()
    f = features_module.load_features(rebuild=args.rebuild)
    print("features ready ({:.1f} s)\n".format(time.time() - started))

    graded = load.graded_mask()
    print("full match, graded frames: ball seen {:.1f}%  out of play {:.2f}%  "
          "median ball speed {:.1f} m/s".format(
              100 * f["ball_seen"][graded].mean(), 100 * f["out_of_play"][graded].mean(),
              np.nanmedian(f["ball_speed"][graded])))


def census(args):
    from mtah import census as census_module, harness

    for path in args.csv:
        name = os.path.splitext(os.path.basename(path))[0]
        codes = harness.load_pred(path)
        print(census_module.format_census(name + "  (full match)", census_module.census(codes)))
        caught, total = census_module.set_piece_catch(codes == harness.D)
        print("Smart Tagging set pieces with dead ball from {:.0f} s before to {:.0f} s after "
              "their tagged start: {}/{}\n".format(census_module.SET_PIECE_BEFORE_S,
                                                   census_module.SET_PIECE_AFTER_S, caught, total))


def cues(_args):
    import warnings

    from mtah import cues as cues_module

    warnings.simplefilter("ignore", RuntimeWarning)
    print(cues_module.report())


def calibrate(args):
    import warnings

    from mtah import calibrate as calibrate_module

    warnings.simplefilter("ignore", RuntimeWarning)
    print(calibrate_module.report(use_dropout=not args.full_ball))


def possession(_args):
    import warnings

    from mtah import poss_calibrate

    warnings.simplefilter("ignore", RuntimeWarning)
    print(poss_calibrate.report())


def changes(_args):
    import warnings

    from mtah import poss_calibrate

    warnings.simplefilter("ignore", RuntimeWarning)
    print(poss_calibrate.changes_shape_report())


def challenger(args):
    import warnings

    from mtah import challenger as challenger_module

    warnings.simplefilter("ignore", RuntimeWarning)
    print(challenger_module.report(use_dropout=not args.full_ball))


def deadcheck(_args):
    import warnings

    from mtah import calibrate as calibrate_module, census as census_module, deadball

    warnings.simplefilter("ignore", RuntimeWarning)
    cues, _ = calibrate_module.our_cues()
    dead = deadball.detect(cues)
    print(census_module.format_dead_check(
        "dead-ball rules on the provided game (label-free checks)",
        census_module.dead_check(dead, cues),
        reference="dead share 44.9%, 96 stoppages per 90, length p50 19.6 s, under 5 s 8%"))


def engine(args):
    import warnings

    from mtah import engine as engine_module

    warnings.simplefilter("ignore", RuntimeWarning)
    print(engine_module.report(use_dropout=not args.full_ball))


def run(args):
    import warnings

    from mtah import census as census_module, harness, submit

    warnings.simplefilter("ignore", RuntimeWarning)
    path, digest, codes = submit.write(args.name)
    problem = harness.pm.validate_submission(harness.pm.read_label_csv(path), config.N_FRAMES)
    print("wrote {}  sha256 {}  validation: {}".format(
        os.path.relpath(path, config.REPO_ROOT), digest[:16], problem or "ok"))
    print(census_module.format_census(args.name + "  (full match)", census_module.census(codes)))
    caught, total = census_module.set_piece_catch(codes == harness.D)
    print("Smart Tagging set pieces with dead ball near their tagged start: {}/{}".format(
        caught, total))


def evaluate(args):
    from mtah import harness

    gt = harness.load_gt("public")
    for path in args.csv:
        name = os.path.splitext(os.path.basename(path))[0]
        result, _ = harness.score(harness.load_pred(path), gt)
        print(harness.format_report(name + "  (public window: correctness check only)", result)
              + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("status", help="environment, paths and asset presence").set_defaults(
        run=status)
    cache_cmd = commands.add_parser("cache", help="parse tracking and tagging into the cache")
    cache_cmd.add_argument("--source", nargs="+", default=["tactical"],
                           choices=["tactical", "broadcast"])
    cache_cmd.add_argument("--rebuild", action="store_true", help="ignore an existing cache")
    cache_cmd.set_defaults(run=cache)
    features_cmd = commands.add_parser("features", help="build per-frame features, summarise")
    features_cmd.add_argument("--rebuild", action="store_true", help="ignore an existing cache")
    features_cmd.set_defaults(run=features)
    census_cmd = commands.add_parser("census", help="label-free plausibility checks")
    census_cmd.add_argument("csv", nargs="+", help="submission CSVs")
    census_cmd.set_defaults(run=census)
    commands.add_parser("cues", help="dead-ball cues measured on other matches (IDSSE)") \
        .set_defaults(run=cues)
    for name, fn, text in (("calibrate", calibrate, "fit the dead-ball rules on IDSSE"),
                           ("challenger", challenger, "logistic challenger on IDSSE"),
                           ("engine", engine, "score engine (named evidence points) on IDSSE")):
        cmd = commands.add_parser(name, help=text + ", nested leave-one-match-out")
        cmd.add_argument("--full-ball", action="store_true",
                         help="keep the IDSSE ball fully tracked instead of hiding it like ours")
        cmd.add_argument("--workers", type=int, default=0,
                         help="fold searches side by side (default: MTAH_WORKERS or 1)")
        cmd.set_defaults(run=fn)
    for name, fn, text in (("possession", possession, "fit the possession rules on IDSSE"),
                           ("changes", changes, "fit the change detector and rule 6 on IDSSE")):
        cmd = commands.add_parser(name, help=text + ", nested leave-one-match-out")
        cmd.add_argument("--workers", type=int, default=0,
                         help="fold searches side by side (default: MTAH_WORKERS or 1)")
        cmd.set_defaults(run=fn)
    commands.add_parser("deadcheck", help="dead-ball rules on the provided game, label-free") \
        .set_defaults(run=deadcheck)
    run_cmd = commands.add_parser("run", help="build the submission file from the tracking")
    run_cmd.add_argument("--name", default="Mes Que Code", help="file name, without .csv")
    run_cmd.set_defaults(run=run)
    eval_cmd = commands.add_parser("eval", help="correctness check on the public window")
    eval_cmd.add_argument("csv", nargs="+", help="submission CSVs")
    eval_cmd.set_defaults(run=evaluate)
    args = parser.parse_args()
    if getattr(args, "workers", None):
        from mtah import calibrate as calibrate_module
        calibrate_module.WORKERS = args.workers
    args.run(args)


if __name__ == "__main__":
    main()
