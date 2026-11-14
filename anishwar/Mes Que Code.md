# Ball Possession from Automatic Tracking

**More than a Hack 2027, Phase 1: Technical Document**
**Team:** Mes Que Code
**Organisers:** Metrica Sports, FC Barcelona Innovation Hub, Mobile World Capital Barcelona

---

## 1. Introduction

This document describes the method that produced the submission file `Mes Que Code.csv`. The method labels every graded frame of the provided match as home possession, away possession or dead ball. Its only input is the automatic tactical tracking.

The method has three stages:

1. Dead-ball detection from the Laws of the Game and from player tempo.
2. Possession while the ball is tracked, from touches and a control filter.
3. Possession while the ball is not tracked, from the shape of the two teams.

Each stage is a short set of rules with named constants. Every tuned constant is calibrated on other matches that carry official labels. The remaining constants are fixed in advance from physics or from the challenge text and are listed in Section 5.4. No constant is calibrated against labels of the provided match.

## 2. Methodology

### 2.1 Design Principle

The provided match supplies no training signal. Hand labels on it are prohibited, and the public five-minute window is reserved for correctness checks. The method therefore derives its rules from football knowledge and its constants from other matches with official labels.

The method uses the provided match in three ways only:

- The algorithm runs on its tracking.
- Label-free plausibility checks compare its output with norms measured on other matches.
- One correctness run on the public window confirms alignment and label meaning for each finished version. This run never selects between versions.

### 2.2 Rules and Learned Models

The team compared rules with learned models under the same test. A points-based score engine with sign-constrained weights scored below the rules. A logistic regression on the dead-ball cues scored above them on dead-ball measures, and Section 6 reports the comparison. Both learned models are harder to interpret than the rules.

The method retains one learned component. A small logistic model reads possession from team shape when the ball is not tracked. It improves the score on six of the seven held-out matches, and its weights have a direct football reading.

### 2.3 Evaluation Protocol

All design decisions use nested leave-one-match-out validation on seven labelled matches:

- The search for each match uses the other six matches. The chosen constants are then scored once on the held-out match.
- The search starts from spread-out seed points drawn by Latin hypercube sampling with a fixed random seed and refines them by coordinate ascent. The stored constants are never a starting point. Exact ties go to the smaller grid value, so a constant with no effect settles at the bottom of its range.
- The dead-ball rules also set the ball-hiding profile of Section 4.3, so the dead-ball search is repeated with the profile re-measured until it returns its own input. It did so in the second round. Re-running the search therefore returns exactly the stored values.
- The final constants are the best setting on all seven matches. The search reports how many one-step neighbours score within 0.01 of it and which constants sit at the edge of their search range. Section 6 states both for each stage.
- Possession is scored with the organisers' code in `evaluation/possession_metrics.py`.

## 3. Assets

Table 1 lists each provided asset and its use.

**Table 1. Use of the provided assets**

| Asset | Use | Purpose |
|---|---|---|
| Tactical tracking of players and ball | Used | The only input to the labels |
| Smart Tagging | Checks only | Automatic tagging supplied by the organisers. It counts tagged set pieces inside detected stoppages and guided which failure cases the team studied. It is never an input and never selects a version |
| Tactical and broadcast video | Viewing only | Viewed to understand the match and to design cues. No viewing produced a label, a correction or a constant |
| Broadcast tracking | Not used | The tactical view covers the whole pitch, so one source suffices |
| Homography | Not used | The tracking is already in pitch coordinates |
| Public five-minute ground truth | Correctness only | Confirms frame alignment and label meaning |

## 4. Training Data

The method uses no label from the provided match. All labelled data comes from the IDSSE dataset.

### 4.1 IDSSE Dataset

**Table 2. Labelled data used for calibration**

| Property | Value |
|---|---|
| Source | Bassek, Rein, Weber and Memmert (2025), *Scientific Data* 12, 195 |
| Content | Seven Bundesliga 2022/23 matches, 668 minutes of official DFL position data at 25 frames per second |
| Labels | Per-frame `BallStatus` and `BallPossession` |
| Labelled by | The DFL data provider, as part of its official tracking. The team labelled nothing |
| Licence | Data copyright DFL, released under CC-BY 4.0. The team converted the data to arrays and hid the ball for calibration as described in Section 4.3. The original data is unchanged |
| Access | Public mirror at huggingface.co/datasets/pysport/idsse-data, read with kloppy 3.19.1 |
| Volume | 709 stoppages of at least one second and 2,351 possession changes in live play |

### 4.2 Conversion to the Challenge Convention

The DFL possession record changes at the first touch of the new team. The median distance from the new team's nearest player to the ball at the change is 0.6 m. The challenge asks for controlled possession instead. Both the IDSSE target and the prediction therefore pass through the same control filter.

The filter returns any spell shorter than two seconds that lies between two spells of the other team. The two-second length follows the challenge text, which names "a player keeps the ball for a few seconds" as a proxy for control. The length is fixed and never tuned.

### 4.3 Matching the Ball Loss of the Provided Match

The ball is missing on 40.7% of the graded frames of the provided tracking. The loss follows a clear pattern:

- The ball disappears in long gaps with a median length of 11.6 seconds.
- It is missing on 63% of the frames that the dead-ball rules mark as dead and on 21% of the frames they mark as live.
- It is missing more often in both states when players move slowly.

This profile is the only statistic of the provided match that enters calibration. It is measured on the provided tracking alone, uses no labels and refers to no specific frame. The calibration hides the IDSSE ball to match the profile, separately in dead and live play, and averages over three random placements. After hiding, 63.3% of IDSSE dead frames and 21.3% of IDSSE live frames lack the ball. All results in this document use this hidden-ball condition.

## 5. Labelling Rules

### 5.1 Dead Ball

The rules follow Law 9 of the Laws of the Game. The ball is out of play when it wholly crosses a line or when the referee stops play.

1. **Ball past a line.** The ball is dead from the frame it crosses the line.
2. **Players stop.** Players at or below their usual pace for 9 seconds mark a stoppage. The usual pace is the local five-minute norm. The stoppage starts 0.5 seconds into the stretch below 1.15 times that pace and ends 1 second before the players accelerate for the restart.
3. **Ball-out linking.** A ball-out joins any stoppage that starts within 12 seconds. A ball-out with no following stoppage is dead for 8 seconds. This case covers quick throw-ins.
4. **Kick-off.** Law 8 places each team in its own half with the kicker on the centre spot. Each team at least 85% in its own half, with a player on the centre spot for half a second, marks a dead ball.

### 5.2 Possession with the Ball Tracked

1. **Touch.** A touch requires a player within 1.5 m of the ball, the nearest opponent at least 0.25 m further away, and a ball action. A ball action is a turn of at least 20 degrees or a speed change of at least 13 m/s.
2. **Loose balls and challenges.** Possession stays with the team of the last touch through loose play, contested balls and frames without the ball. One team therefore holds possession at every live frame, as in the framework of Link and Hoernig [3].
3. **Restarts.** The team of the first touch after a stoppage owns the restart. A live stretch with no touch goes to the team that did not hold the ball before the stoppage.
4. **Hidden changes.** A change between touches of different teams across a ball gap is placed at the end of the gap.
5. **Control.** A spell shorter than 2 seconds between two spells of the other team counts as a touch without control. Deflections, blocks and headers won without the ball fall in this case. The spell returns to the other team.

### 5.3 Possession with the Ball Not Tracked

The team-shape model can transfer possession to the other team inside a gap in the ball tracking. The transfer requires a model probability of at least 0.45 for that team over 3 seconds. On held-out matches the model names the owner correctly on 82.3% of live frames without the ball. Holding the last owner names it correctly on 73.8%.

The model is a logistic regression on seven features. Each feature is the home value minus the away value:

- Forward velocity towards the attacked goal.
- Forward position past the halfway line.
- Team depth and team width as interquartile ranges.
- Share of players in the opponent half.
- Last owner seen, faded over six seconds, and its product with forward velocity.

The features use medians and interquartile ranges because stray tracks in the automatic tracking inflate standard deviations. The model is trained on IDSSE live frames without the ball and uses the DFL owner as the label. Its weights are stored in `mtah/shape.py`. A team in possession advances and widens, and the weights reflect this.

### 5.4 Constants Fixed in Advance

The following constants are not tuned. Each follows from physics, from tracking practice or from the challenge text.

**Table 3. Fixed constants**

| Constant | Value | Basis |
|---|---|---|
| Control filter | 2 s | Challenge text: "a player keeps the ball for a few seconds" |
| Tempo norm window | 300 s | Long against any stoppage, short against changes of match tempo |
| Speed smoothing | 1 s | About one stride cycle |
| Maximum player speed | 12 m/s | Above any sprinter; faster values are tracking jumps |
| Centre-spot radius | 3 m | Law 8 places the kicker on the centre spot; the radius allows for tracking error |
| Ball direction window | 0.12 s each side | Long enough to suppress position jitter, short enough to isolate one touch |
| Last-owner fade | 6 s | Prior on the last owner in the team-shape model |

## 6. Results

Table 4 reports the held-out scores on IDSSE under the hidden-ball condition.

**Table 4. Nested leave-one-match-out results on IDSSE**

| Stage | Measure | Held-out score |
|---|---|---|
| Dead ball | Stoppage F1 / frame F1 | 0.756 / 0.865 |
| Possession, rules in 5.2 | spellF1 / macroF1 | 0.513 / 0.770 |
| Possession, rules in 5.2 and 5.3 | spellF1 / macroF1 | 0.531 / 0.781 |

The searches differ in how settled their results are:

- **Dead ball.** 17 of 18 one-step neighbours score within 0.01 of the chosen setting. The slow-pace threshold sits at the top of its range. The inferred ball-out rule is off, which leaves its duration without effect, and the kick-off duration made no difference on the labelled matches; the tie rule puts both at the bottom of their ranges. Every fold chose the same stoppage start, ball-out link and ball-out duration; the folds differ only in the minimum stoppage length of 9 or 10 seconds and, in one fold, the slow-pace threshold and the kick-off share and duration.
- **Search audit.** 200 further seed points with the best 20 refined, and 3,000 uniformly random grid points with the best 10 refined, found no setting above the chosen one. The best random point before refinement scored 0.804 against 0.818. By the rule of three, fewer than 0.11% of the grid would score higher, at 95% confidence.
- **Possession rules.** All 10 one-step neighbours score within 0.01. The ball-at-the-feet rule sits at zero, which disables it, and the hidden-change position at the end of the gap; both are range limits. All seven folds chose a turn of 20 degrees and a change at the end of the gap. The folds chose speed changes from 11 to 14 m/s and touch radii from 1.5 to 2.25 m.
- **Team shape.** Four folds chose a probability of 0.45 held for 3 seconds and three chose 0.5 held for 2 seconds.

Table 5 separates the remaining error by source. The figures use the rules in Section 5.2 without the team-shape model.

**Table 5. Score with each error source removed, all seven matches**

| Condition | spellF1 |
|---|---|
| Submitted pipeline, rules in 5.2 | 0.519 |
| True dead ball | 0.655 |
| Ball fully tracked | 0.654 |
| True dead ball and ball fully tracked | 0.826 |

The team tested the following alternatives:

- A logistic regression on the dead-ball cues scored 0.762 in stoppage F1 and 0.881 in frame F1. Both scores exceed the rules, and frame F1 is higher on all seven matches. Its weights are harder to read than the rules. The end-to-end possession test decides between the two.
- A score engine with sign-constrained evidence points scored 0.689 in stoppage F1 and 0.846 in frame F1. It remains a measuring tool. Sustained slow play and a recent ball-out carry almost all of its evidence.
- An inferred ball-out rule marked the ball dead from the frame it vanished near a line while moving towards it. Every fold switched it off.
- Three rules for the end of a stoppage gave no in-sample gain and a held-out stoppage F1 of 0.740, with less agreement between folds. They were rejected.
- A ball-at-the-feet touch rule added false touches in duels.
- Smoothing the team-shape probability gave no gain.

Table 6 compares the output on the provided match with norms from the labelled matches. These checks use no labels.

**Table 6. Label-free checks on the provided match**

| Check | Provided match | IDSSE norm |
|---|---|---|
| Dead share | 46.7% | 44.9% |
| Stoppages per 90 minutes | 112 | 96 |
| Median stoppage length | 20.6 s | 19.6 s |
| Possessions per 90 minutes | 194 | 304 |
| Tagged set pieces inside a detected stoppage | 51 of 58 | Not applicable |

## 7. Limitations

- **Quick restarts.** Throw-ins and goal kicks taken at full pace are missed when the tracking does not show the ball crossing the line.
- **Possession count.** The method finds fewer possessions than the target. Changes inside long ball gaps remain the largest loss.
- **Stoppage edges.** Errors in stoppage start and end times cost about 0.14 spellF1 on the labelled matches.
- **Transfer.** The constants come from Bundesliga matches with corrected tracking. The hidden-ball profile matches the provided match, but player-tracking noise still differs between the sources.

Further work would address these limitations as follows:

- A sequence model for possession inside long ball gaps.
- Restart detection from set-piece formations such as walls, corners and throw-in lines.
- Calibration on labelled matches from further leagues.

## 8. Reproduction

The environment uses Python 3.12.8, numpy 1.26.3, pandas 2.1.4 and kloppy 3.19.1. The provided data must be placed in the repository's `data/` folder.

```bash
pip install -r anishwar/requirements.txt
python anishwar/pipeline.py run --name "Mes Que Code"
```

The second command writes `anishwar/results/Mes Que Code.csv`. The build is deterministic and produces an identical file on every run. It requires only the provided data.

The IDSSE matches are required only to re-derive the constants:

```bash
python anishwar/pipeline.py calibrate
python anishwar/pipeline.py possession
python anishwar/pipeline.py shape
python anishwar/pipeline.py deadcheck
```

The commands run in this order. Each one prints the constants it finds; these are stored in the source before the next command runs, because each stage uses the constants of the stage before it.

## References

1. IFAB. *Laws of the Game 2025/26*. Law 8: The Start and Restart of Play. Law 9: The Ball In and Out of Play.
2. Bassek, M., Rein, R., Weber, H. and Memmert, D. (2025). An integrated dataset of spatiotemporal and event data in elite soccer. *Scientific Data* 12, 195. doi:10.1038/s41597-025-04505-y.
3. Link, D. and Hoernig, M. (2017). Individual ball possession in soccer. *PLOS ONE* 12(7), e0179953. doi:10.1371/journal.pone.0179953.
4. kloppy. Open-source Python library for football tracking and event data. BSD-3 licence.
