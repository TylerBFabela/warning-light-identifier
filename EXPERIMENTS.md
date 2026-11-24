# Experiment log: improving the warning-light classifier on real crops

**Summary.** The website runs v5b seed 1 (MobileNetV3-small, pretrained on ImageNet, fine-tuned on synthetic images) with a
`none` penalty of 1.0 and an "identified" cutoff of 0.9. On the pooled development sets (113 real lights, 134 spots that are
not warning lights) it names 87 of the lights, 39 of them as "identified" with none wrong, and shows 10 of the non-lights as a
low-confidence "possible" light. These sets guided the decisions, so the numbers are optimistic; a fresh test set is still
needed (section 10). Sections 1-6 cover v3 to v4, section 7 covers v5a and v5b, and section 8 covers the light finder.

Numbers come from the evaluation scripts in `src/` (`probe_no_none.py`, `compare_models.py`, `evaluate.py`,
`finder_eval.py`). Counts first, percentages second. Small samples: read every number with the sample size next to it.

## 1. Question and setup

**Question.** The first model (v3, a small CNN trained only on synthetic images) scored well on its own synthetic
validation set (99.9%) but poorly on real dashboard crops. What limits it, and can we fix it without taking new photos?

**Models compared**
| Name | What it is |
|---|---|
| v3 | `SmallCNN`, 242,861 weights, trained from scratch on synthetic images (`data_v3`, 8,100 train / 900 val) |
| v4_scratch | MobileNetV3-small (934,509 weights), random start, same data and settings |
| v4_pre (seed 0), v4_pre_s1, v4_pre_s2 | MobileNetV3-small started from ImageNet weights, fine-tuned on the same data. Seeds 0, 1, 2 |

Training settings for v4: AdamW, learning rate 1e-3 (OneCycle), 12 epochs, batch 64, label smoothing 0.1, the same
augmentation and the same pixel normalization as v3. About 18 s per epoch on the M1 Pro (mps).

**Real evaluation data.** 119 crops cut from a public Roboflow dataset ("Car Dashboard Detection v2 demo-1"), sorted by
hand into 7 light classes plus `none` (crops of real dashboard icons that are not one of our 12 classes: ESP, engine
overheating, master warning). 38 unusable crops were set aside. The crops are used **for evaluation only**: never
trained on, never committed, never published. The dataset lists an MIT license, but that is the uploader's claim and
some images derive from stock photos, so the crops stay private.

**Split by source photo** (31 photos, fixed seed, `src/split_external.py`) so crops of one photo never appear on both sides:
| Half | Real lights | `none` crops | Used for |
|---|---|---|---|
| dev | 41 | 9 | choosing models and settings |
| test | 52 | 17 | one final run, no decisions afterwards |

**Rules set before the final run.** Pass mark on dev: at least 11 of 41 real lights right (`none` allowed) with at most
1 of 9 `none` crops called a light. Penalty setting chosen on dev only: the largest `none` penalty keeping false alarms
at 1 of 9 or fewer (this gave penalty 2 for seeds 0 and 1; seed 2 alone would have allowed 3; one value, 2, was used for
all). Candidate to ship chosen on dev before the test run: seed 1.

## 2. How the probe reads the model (`src/probe_no_none.py`)

- **Right, `none` allowed:** the model's normal answer is compared with the true class.
- **Right, `none` ignored:** the `none` score is removed and the best of the other 12 classes is used.
- **In best 3:** the true class is among the model's three highest-scoring classes, `none` excluded.
- **Penalty p:** p is subtracted from the `none` score (log-probability units, same as subtracting p from the `none` logit)
  before choosing. "`none` called a light" is the share of true `none` crops that are then given a light name.
- **Baseline:** always answering the most common class scores 20% (dev) and 21% (test).

## 3. Results

### Dev half (41 real lights, 9 none crops)
| Model | Right, none allowed | Right, none ignored | In best 3 | None crops called a light (penalty 0) | At penalty 2: lights right / none called a light |
|---|---|---|---|---|---|
| v3 | 7 (17%) | 11 (27%) | 20 (49%) | 0 of 9 | 9 / 3 of 9 |
| v4_scratch | 7 (17%) | 9 (22%) | 19 (46%) | 0 of 9 | 7 / 0 of 9 |
| v4_pre seed 0 | 11 (27%) | 25 (61%) | 34 (83%) | 0 of 9 | 19 / 1 of 9 |
| v4_pre seed 1 | 15 (37%) | 29 (71%) | 36 (88%) | 0 of 9 | 18 / 1 of 9 |
| v4_pre seed 2 | 9 (22%) | 28 (68%) | 36 (88%) | 0 of 9 | 14 / 0 of 9 |
| v4_pre average | 11.7 (28%) | 27.3 (67%) | 35.3 (86%) | | 17 / 0.7 of 9 |

### Test half, run once (52 real lights, 17 none crops)
| Model | Right, none allowed | Right, none ignored | In best 3 | None crops called a light (penalty 0) | At penalty 2: lights right / none called a light |
|---|---|---|---|---|---|
| v3 | 4 (8%) | 13 (25%) | 31 (60%) | 0 of 17 | 9 / 5 of 17 |
| v4_pre seed 0 | 13 (25%) | 29 (56%) | 44 (85%) | 1 of 17 | 17 / 3 of 17 |
| v4_pre seed 1 (candidate) | 14 (27%) | 25 (48%) | 47 (90%) | 0 of 17 | 20 / 2 of 17 |
| v4_pre seed 2 | 7 (13%) | 30 (58%) | 47 (90%) | 0 of 17 | 12 / 0 of 17 |
| v4_pre average | 11.3 (22%) | 28.0 (54%) | 46.0 (88%) | | 16.3 (31%) / 1.7 of 17 (10%) |

v4_scratch was not run on the test half (it did not beat v3 on dev).

### The `none` penalty curve on test (lights right / none crops called a light)
| Penalty | Seed 0 | Seed 1 | Seed 2 | v3 |
|---|---|---|---|---|
| 0 | 13 / 1 | 14 / 0 | 7 / 0 | 4 / 0 |
| 1 | 17 / 1 | 16 / 1 | 11 / 0 | 5 / 4 |
| 2 | 17 / 3 | 20 / 2 | 12 / 0 | 9 / 5 |
| 3 | 22 / 4 | 22 / 5 | 21 / 0 | 12 / 8 |
| 4 | 27 / 9 | 25 / 10 | 26 / 10 | 13 / 13 |
| 6 | 29 / 17 | 25 / 17 | 30 / 17 | 13 / 16 |

All three seeds fall off a cliff between penalty 3 and 4 (false alarms jump to about 10 of 17). Penalty 2 sits safely before it.

### Per class: real lights right with `none` ignored (average of 3 seeds, dev and test combined)
| Class | Right | Of | Share |
|---|---|---|---|
| oil_pressure | 15.3 | 18 | 85% |
| airbag | 6.0 | 8 | 75% |
| tire_pressure | 5.7 | 8 | 71% |
| battery | 11.3 | 17 | 67% |
| check_engine | 11.0 | 19 | 58% |
| abs | 2.7 | 8 | 33% |
| brake_system | 3.3 | 15 | 22% |
| **all** | **55.3** | **93** | **59%** |

## 4. What the results support

1. **Pretraining is what helped, not the architecture.** The same MobileNet from a random start matches v3 (dev: 7 / 9 / 19). Started from ImageNet weights it gets 25-29 of 41 with `none` ignored on dev and 25-30 of 52 on test, against 11 and 13 for v3. Both models fit the synthetic validation set equally well (about 99%), so the synthetic validation score cannot predict real-crop performance.
2. **Across all three seeds, the right class is among the model's best 3 for 85-90% of real lights on test** (v3: 60%). Top-1 with `none` allowed is much lower (13-27%, v3: 8%), because `none` still takes many real lights.
3. **The `none` penalty buys lights at a good rate for v4 and a poor one for v3.** At penalty 2 on test: v4 averages 16.3 of 52 lights right (31%) with 1.7 of 17 `none` crops falsely called lights (10%); v3 gets 9 of 52 (17%) with 5 of 17 (29%).
4. **Dev numbers were optimistic.** Seed 1, chosen as best on dev, scored 71% (none ignored) on dev and 48% on test. The average over seeds fell from 67% to 54%. Part of the gap is selection, part is that the test photos differ. The test numbers are the ones to report.
5. **Seed matters.** With `none` allowed the three seeds range from 7 to 14 of 52 on test (9 to 15 of 41 on dev). The class ranking is stable (best-3 and none-ignored move less); the `none` versus light boundary is what shifts.

## 5. What did not work, or is not established

- **Blur is not the main cause of errors.** Crops split at median sharpness: the sharper half was not better. In the v4 runs the blurrier half was equal or better in every seed on test. The sharpness measure probably also tracks image style (3D renders, textures), so it is a weak proxy. Not investigated further.
- **A plain `none` threshold or penalty on v3 was not a fix:** each extra light cost about one false alarm (test: 5 right and 4 false at penalty 1).
- **`brake_system` is not solved.** In every model, 9-12 of its 15 crops are called `parking_brake`. This persisted across architectures, seeds and pretraining, so it is probably a property of how the two symbols are drawn in our data, not a training accident.
- **Small samples.** 41 and 52 lights; one crop is 2.4 or 1.9 points. The three seeds share the same test crops, so averaging them does not shrink the test-sampling noise (roughly +/-12 points at 95% for a share near 30%). Direction and size of the top-3 and none-ignored gains are large enough to trust; the "none allowed" gain is moderate.
- **The test half is now used up** for this model family. Any further model change needs a new evaluation set.
- **These are crop-level numbers.** The deployed app also depends on the light finder; end-to-end on whole photos still has to be re-measured with the new model.

## 6. Second source: crops from the author's own cars (same probe, v3 vs v4_pre seed 1)

Sets: `real_dev` (Mazda, 33 lights / 60 none), `real_test` (Toyota RAV4, 20 / 30), `real_night` (both cars at night, 20 / 34).
The Mazda crops flatter both models, because the synthetic generator was tuned on that car, and 11 of its 33 lights are
`parking_brake` (probably near-duplicates). RAV4 and night are the cleaner checks. Per-class counts here are 0-4 crops
(except parking_brake), so they are anecdotes.

| Set | Model | Right, none allowed | Right, none ignored | In best 3 | Always-most-common | None crops called a light (penalty 0) |
|---|---|---|---|---|---|---|
| Mazda dev (33 / 60) | v3 | 23 | 26 | 29 | 11 | 9 of 60 |
| | v4_pre_s1 | 26 | 29 | 32 | 11 | 1 of 60 |
| RAV4 (20 / 30) | v3 | 6 | 9 | 15 | 3 | 1 of 30 |
| | v4_pre_s1 | 12 | 15 | 17 | 3 | 0 of 30 |
| Night (20 / 34) | v3 | 3 | 5 | 10 | 3 | 4 of 34 |
| | v4_pre_s1 | 8 | 13 | 19 | 3 | 2 of 34 |
| **Pooled (73 / 124)** | v3 | 32 (44%) | 40 (55%) | 54 (74%) | | 14 of 124 (11%) |
| | **v4_pre_s1** | **46 (63%)** | **57 (78%)** | **68 (93%)** | | **3 of 124 (2%)** |

`none` penalty on the pooled own-car sets (lights right of 73 / none crops called a light of 124):
| Penalty | v3 | v4_pre_s1 |
|---|---|---|
| 0 | 32 / 14 | 46 / 3 |
| 1 | 32 / 24 | 51 / 6 |
| 2 | 34 / 34 | 53 / 11 |
| 3 | 36 / 59 | 54 / 25 |
| 4 | 38 / 88 | 56 / 43 |

What this adds:
1. The v3-to-v4 gain repeats on a second source of images: RAV4 6 to 12 of 20, night 3 to 8 of 20 (right, none allowed).
2. The `none` crops here are the light finder's real false detections. v4 calls 3 of 124 a light at default, v3 calls 14.
3. Penalty 1 buys 5 more lights for 3 more false alarms; penalty 2 buys 7 for 8; penalty 3 and above is poor. The penalty (1 or 2) and the confidence cutoffs should be chosen together, on all dev sets combined.
4. Weak spots that remain: `abs` 0 of 4 right for v4 (v3 got 3 of 4 with none ignored), so abs is a regression; `brake_system` 3 of 10 (v3 0 of 10), with the RAV4 crops all called `parking_brake`; `headlights` 0 of 3 with `none` allowed, but 3 of 3 with `none` ignored (the model calls them `none`); green arrows and headlights are still confused at night.
5. Improvements on the other classes: `check_engine` 5 of 6 (v3 1 of 6), `tire_pressure` 4 of 5 (v3 0 of 5), `turn_left` 5 of 7 (v3 3 of 7).

## 7. Unseen-drawing test on synthetic images (does the model memorize drawings?)

**Why.** The synthetic validation score (99.8%) cannot detect memorized artwork: all 9,000 training and validation
images come from the same 12 drawings, one per class. To test the idea that v4 learned its 12 drawings and not the
symbols, v4 (seed 1) was run on synthetic images generated from drawings it never saw.

**Setup.** 14 new drawings (SVGs converted to PNG; `turn_left` and `turn_right` are the left and right halves of one
two-arrow drawing). Same generator flags as the training data, seed 7: 100 images per class, 300 `none`, no extra `none` icons
(`--hard-negatives --none-factor 3 --lowres --centered --color-jitter --bright-bg --labels`). Scored with `src/evaluate.py`.

**Invalid first run (discarded).** The holdout folder still held the 12 original training drawings (plus `coolant_temp` and 4
`none` icons), so about half of each class's images used drawings v4 was trained on. Reported accuracy 93.2%; this
number is inflated and should not be quoted. Cause: the converted PNGs were written into a copy of the whole `icons/` folder.

**Clean run (new drawings only, 1,500 images).** Overall accuracy **87.1%** (synthetic validation: 99.8%).
| Class | Recall | Class | Recall |
|---|---|---|---|
| abs | 0.75 | oil_pressure | 0.99 |
| airbag | 0.99 | parking_brake | 0.97 |
| battery | 1.00 | tire_pressure | 1.00 |
| brake_system | 1.00 | turn_left | **0.61** |
| check_engine | **0.00** | turn_right | 0.77 |
| headlights | 1.00 | none | 0.99 (precision 0.61) |
| high_beam | 1.00 | | |

Mistakes: check_engine to none 100, turn_left to none 39, abs to none 24, turn_right to none 23, parking_brake to brake_system 3,
everything else 1 or 2. Every sizable error is "called it `none`"; the model almost never confuses one class with another.

**Reading it (with limits).**
- Eight of twelve classes stay at 97-100% on a different drawing, so v4 is not simply memorizing its icons.
- The drop is concentrated in four classes. The `check_engine` result (0 of 100) comes from one new drawing, so it says this
  particular engine drawing is not recognized, not that engine symbols in general fail. Most classes have only one or two new
  drawings, so each per-class number rests on very few drawings.
- Pending check: confirm in the mask preview sheet that the `check_engine` drawing is extracted correctly. A broken extraction
  (frame, caption, inverted shape) would also produce 0%.
- The prediction made before this run (overall between 70% and 90%, `check_engine` near 0%) matched.

**Why the failures happened (mask preview sheets).** The original training drawings are mostly outline or hollow (thin outline
engine, hollow arrows, thin ABS ring); the failing new drawings are solid or bolder (solid engine silhouette, solid block arrows,
bold ABS). The classes whose style stayed the same scored 97-100%. Counterexample: `oil_pressure` changed from an outline can to a
solid can and still scored 99%. Style change is the leading explanation, not a proven rule.

### v5a: automatic style variants (no new drawings)
**Change.** `make_synth.py --styles` (off by default; with the flag off the output is byte-for-byte the same as before, checked
by checksum). Each time a class drawing is used: 25% a solid version (enclosed gaps filled, marks inside the shape such as a "!",
a "P" or letters kept as cut-outs), 25% an outline version (a rim of 2-4.5% of the icon height), 50% unchanged, plus a small
random bolder or lighter stroke. Applied to the 12 class drawings only, not to the `none` negatives.
**Data and model.** `data_v5a`: the v3/v4 flags plus `--styles`, seed 2, only the 12 original drawings (the 14 new drawings
were not used). MobileNetV3-small pretrained, lr 1e-3, 12 epochs, seed 1 (best epoch 10, validation accuracy 99.8%).
**Result on the same unseen drawings (`data_holdout2`, 1,500 images):** accuracy **99.7%** (v4: 87.1%).
| Class | v4 recall | v5a recall |
|---|---|---|
| check_engine | 0.00 | 1.00 |
| turn_left | 0.61 | 1.00 |
| turn_right | 0.77 | 1.00 |
| abs | 0.75 | 1.00 |
| all others | 0.97-1.00 | 0.99-1.00 |
**Caveat.** The style variants were designed after seeing which holdout drawings failed, so `data_holdout2` is no longer an
independent test of unseen drawings in general. It shows that the augmentation covers those styles. Real photos are the judge.
**Criterion set before looking at real-photo results.** Pooled `external_dev` plus the three own-car sets (114 real lights; 133
`none` crops). v5a counts as better than v4 if its 3-seed average is at least 5 lights higher with `none` allowed, and the number
of `none` crops called a light (penalty 0) rises by no more than 3. `external_test` stays unused.

**Real-photo comparison, 3 seeds each** (lights right with `none` allowed, per set; totals of 114 lights and 133 `none` crops):
| Model | external_dev (41) | Mazda (33) | RAV4 (20) | Night (20) | Right, none allowed | Right, none ignored | None crops called a light |
|---|---|---|---|---|---|---|---|
| v4 seed 0 | 11 | 26 | 11 | 6 | 54 | 84 | 1 |
| v4 seed 1 | 15 | 26 | 12 | 8 | 61 | 86 | 3 |
| v4 seed 2 | 9 | 26 | 12 | 7 | 54 | 86 | 1 |
| v5a seed 0 | 16 | 29 | 13 | 10 | 68 | 93 | 8 |
| v5a seed 1 | 18 | 27 | 14 | 11 | 70 | 92 | 3 |
| v5a seed 2 | 15 | 27 | 13 | 11 | 66 | 87 | 7 |
| **v4 average** | 11.7 | 26.0 | 11.7 | 7.0 | **56.3** | **85.3** | **1.7** |
| **v5a average** | 16.3 | 27.7 | 13.3 | 10.7 | **68.0** | **90.7** | **6.0** |

**Verdict against the criterion set beforehand:** lights +11.7 (needed at least +5: met); false alarms +4.3 (allowed at most +3:
**not met**). v5a did not pass the rule as written.
**What the numbers show:**
- Every v5a seed finds more lights than every v4 seed (66-70 against 54-61), and v5a is ahead on all four sets.
- With `none` ignored, every v5a seed is also ahead (87-93 against 84-86). That measure does not depend on where the `none`
  boundary sits, so the gain is not only a model that says "light" more readily.
- All the extra false alarms are on the own-car `none` crops (the light finder's real false detections: Mazda +3.0, night +1.3);
  none on the external `none` crops.
- Seed 1 of v5a: 70 lights with 3 false alarms, the same false-alarm count as v4 seed 1 (61 lights).
**Decision (made after seeing the results, so weaker than a pre-set rule):** ship v5a seed 1, with no `none` penalty and the
"identified" cutoff kept at 0.9 until the cutoffs are re-checked. Report the 3-seed averages and the missed false-alarm limit.
**Per class, pooled over the four sets (`src/compare_models.py`; lights right with `none` allowed, averages over seeds):**
| Class | Crops | v3 (live site) | v4 avg | v5a avg | v5a ensemble |
|---|---|---|---|---|---|
| abs | 7 | 0.0 | 0.3 | 3.0 | 3.0 |
| airbag | 10 | 6.0 | 7.0 | 8.0 | 7.0 |
| battery | 12 | 3.0 | 3.3 | 4.7 | 4.0 |
| brake_system | 17 | 1.0 | 4.0 | 5.3 | 5.0 |
| check_engine | 14 | 2.0 | 8.3 | 9.7 | 11.0 |
| headlights | 3 | 2.0 | 0.3 | 0.0 | 0.0 |
| high_beam | 4 | 2.0 | 2.7 | 3.0 | 3.0 |
| oil_pressure | 13 | 6.0 | 6.3 | 6.0 | 5.0 |
| parking_brake | 13 | 11.0 | 11.3 | 12.0 | 12.0 |
| tire_pressure | 8 | 1.0 | 5.3 | 6.3 | 7.0 |
| turn_left | 7 | 3.0 | 4.0 | 4.7 | 5.0 |
| turn_right | 6 | 2.0 | 3.3 | 5.3 | 5.0 |
| **all lights** | **114** | **39.0** | **56.3** | **68.0** | **67.0** |
| none kept as none | 133 | 119.0 | 131.3 | 127.0 | 126.0 |

- False alarms at 0.9 confidence or more (the website's "identified" cutoff): **0 for every model**. On these sets, every
  false alarm would have been shown as "possible" or "unsure", never as an identified light.
- The 3-seed ensemble is not better than a single seed (67 against 68 lights, 7 against 6 false alarms): not worth a 3x download.
- Confusion matrix of v5a seed 1: of its 44 missed lights, **39 were called `none`**; only 5 were called another light
  (brake_system to abs 2 and to tire_pressure 1, check_engine to tire_pressure 1, turn_left to turn_right 1). The model rarely
  confuses lights with each other; it mostly fails to recognize a crop as a light.
- Weakest classes: brake_system (10 of 17 to `none`), oil_pressure (7 of 13 to `none`, no gain since v4), battery (6 of 12 to
  `none`), headlights (0 of 3; v3 had 2 of 3, but 3 crops is too few to read much into).

**`none` penalty for v5a seed 1 (pooled sets):**
| Penalty | Lights right (of 114) | False alarms (of 133) | Shown as "identified" (>= 0.9) | Identified but wrong |
|---|---|---|---|---|
| 0 | 70 | 3 | 19 | 0 |
| 0.5 | 72 | 6 | 21 | 0 |
| 1.0 | 76 | 9 | 24 | 0 |
| 1.5 | 81 | 10 | 24 | 0 |
| 2.0 | 84 | 15 | 24 | 0 |
| 3.0 | 88 | 26 | 24 | 0 |
Only about 1 in 5 lights reaches the 0.9 "identified" cutoff, and none of those is wrong at any penalty; most correct answers
fall in the "possible" band (0.4-0.9). Penalty 1.5 is the knee (1.0 to 1.5: +5 lights for +1 false alarm; 1.5 to 2.0: +3 for +5).
Chosen on the same sets, so the gain is optimistic.

**What the refused lights look like (`src/show_mistakes.py`, v5a seed 1).** Own-car sets, 21 missed lights:
| Cause | Count | Examples |
|---|---|---|
| Text-only lights (no symbol in training) | 5 | "BRAKE" (RAV4 x3, night x1), "ABS" (RAV4) |
| Labels or class definition | 4 | 3 "headlights" crops are actually side-light / "lights on" symbols, while the side-light drawing (ISO 0456) was in `icons/none`; 1 "HOLD" (auto hold) labeled parking_brake, but HOLD is one of the generator's `none` text decoys |
| Red brake "(!)" photographed as orange | 3 | called abs or tire_pressure (amber classes) |
| Clear symbol, but glow or a pale daytime background | 7 | night airbag, battery, oil can, glowing arrow; Mazda ABS x2 and arrow (50-76% none) |
| Crop problems | 2 | tiny high-beam icon inside a large crop; arrow called the wrong direction at 33% |
False alarms (3, all below 0.9): power-steering "!" called abs, red stripes called airbag, a whole-dashboard crop called battery.
External set (23 missed): mostly the dataset's own blur and salt-and-pepper noise, colored glow backgrounds and parts of
neighbouring icons; many refusals at 41-75% `none`, which a penalty partly recovers.

**Changes decided from this (v5b, "more drawings"):** add the 14 new drawings; add text-only "BRAKE" (brake_system) and "ABS" (abs)
drawings made with Pillow's built-in font; redefine `headlights` as "exterior lights on" (low beam or side lights) and move the
side-light drawings from `icons/none` to `icons/headlights`; relabel the "HOLD" crop from parking_brake to none (auto hold is not
the parking brake, as the class list already says). These were chosen after seeing the errors, and all of them go in one run,
so their separate effects cannot be told apart.

### v5b: more drawings (14 new, text-only BRAKE/ABS/PARK, side lights and bulb symbol in `headlights`)
Same flags as v5a (with `--styles`), seed 2, 3 training seeds. A word icon never gets a second word printed under it
(`is_word` check; with the old icons the generator's output is unchanged, checked by checksum). After the HOLD relabel the
pooled sets hold 113 real lights and 134 `none` crops; all models were re-scored on the corrected labels.

| Class | Crops | v5a avg | v5b avg | v5b ensemble |
|---|---|---|---|---|
| abs | 7 | 3.0 | 6.3 | 7.0 |
| airbag | 10 | 8.0 | 7.7 | 8.0 |
| battery | 12 | 4.7 | 3.7 | 4.0 |
| brake_system | 17 | 5.3 | 9.3 | 10.0 |
| check_engine | 14 | 9.7 | 9.0 | 11.0 |
| headlights | 3 | 0.0 | 1.0 | 1.0 |
| high_beam | 4 | 3.0 | 3.0 | 3.0 |
| oil_pressure | 13 | 6.0 | 7.3 | 7.0 |
| parking_brake | 12 | 12.0 | 12.0 | 12.0 |
| tire_pressure | 8 | 6.3 | 6.0 | 6.0 |
| turn_left | 7 | 4.7 | 5.0 | 6.0 |
| turn_right | 6 | 5.3 | 4.7 | 5.0 |
| **all lights** | **113** | **68.0** | **75.0** | **80.0** |
| none crops called a light | 134 | 6.0 | 8.3 | 7.0 |
| ... at 0.9 confidence or more | | 0 | 0 | 0 |

**Verdict against the rule set beforehand:** lights +7.0 (needed at least +3: met); false alarms +2.3 (allowed at most +2:
**missed by 0.3**, about one false alarm over three seeds); none at 0.9 confidence (met). Strictly, v5b did not pass.
**Penalty curve, v5b seed 1** (lights right of 113 / false alarms of 134 / shown as identified / identified but wrong):
0: 75 / 6 / 36 / 0; 0.5: 82 / 8 / 38 / 0; **1.0: 87 / 10 / 39 / 0**; 1.5: 88 / 13 / 41 / 0; 2.0: 89 / 20 / 41 / 0; 3.0: 92 / 37 / 42 / 0.
At the same number of false alarms v5b finds more lights than v5a (6 false alarms: 75 against 72; 10: 87 against 81), and
it is far more often confident when right (36 identified at penalty 0, against 19 for v5a), with no identified light wrong.
**A flaw in the rule:** it compared false alarms at penalty 0 only. A model that says "light" more readily is penalized
there even when it is better at every false-alarm level, and the penalty sets that level anyway. A better rule compares
models at equal false alarms. This is noted after seeing the results, so it is weaker than a rule set in advance.
**Decision:** ship v5b seed 1 with a `none` penalty of 1.0 (the knee: 0.5 to 1.0 gains 5 lights for 2 false alarms, 1.0 to
1.5 gains 1 light for 3) and the "identified" cutoff at 0.9. On the pooled development sets that means 87 of 113 lights
named (77%), 39 of them as "identified" with none wrong, and 10 of 134 non-lights shown as a low-confidence "possible" light.
**Caveat:** the text icons and the side-light change were chosen from errors in these same sets (for example the "BRAKE"
crops), so the brake_system and abs gains are optimistic. The ensemble (80 lights) was not chosen: 3x download and more code.

**Limit:** `external_dev` and the own-car sets have now guided several rounds of decisions (v3, v4, v5a, seed choice). They are
development sets, and their numbers are optimistic. A fresh, untouched test set is needed before quoting a final accuracy.

## 8. The light finder (the cropper)

**Why.** The model only sees what the finder boxes, and every accuracy number above is measured on crops the finder already
produced. A light the finder misses never reaches the model.

**How it was measured (`src/finder_eval.py`).** The finder was run on the Roboflow photos, whose lights carry hand-drawn
boxes. Each labeled light counts as *clean* (a finder box centered on it and fitting it, overlap 0.15 or more), *poor*
(covered but merged, oversized or cut short) or *missed*. Roboflow's `train` copies are rotated and cropped by its
augmentation and their labels can sit beside the icons, so decisions use the `valid` and `test` photos only (16 photos,
53 labeled lights; one photo per source). The author's own photos were checked by eye with `src/finder_compare.py`.

**Baseline (current finder):** valid/test: clean 19, poor 29, missed 5, 10.1 boxes per photo. All 54 photos: clean 90 (41%),
poor 83 (38%), missed 45 (21%). The main fault is over-merging: glow joins neighbouring lights, and gauge scales, needles
and lit displays become large boxes that swallow icons. Daylight photos (RAV4) lose the lights' color, so they are missed.

**Settings tried (`src/finder_sweep.py`, valid/test):**
| Setting | Clean | Poor | Missed | Boxes/photo |
|---|---|---|---|---|
| current (gap 1% of the short side) | 19 | 29 | 5 | 10.1 |
| gap 0.5% | 29 | 18 | 6 | 12.0 |
| gap 0.3% | 28 | 17 | 8 | 13.2 |
| merge cap 10% | 30 | 15 | 8 | 16.4 |
| gap 0.5% + merge cap 15% | 33 | 12 | 8 | 16.3 |
| **gap 0.5% + drop sparse shapes > 10%** | **31** | **15** | **7** | **12.6** |
| gap 0.5% + white core | 24 | 26 | 3 | 10.9 |
| gap 0.5% + sparse + white core | 30 | 18 | 5 | 12.1 |
| looser color thresholds 0.4/0.4 | 13 | 37 | 3 | 9.4 |
Rule set before round 2: the most clean boxes, with at most 2 more misses and at most 3 more boxes per photo than current.
**Chosen: gap 0.5% + drop big sparse shapes (needles, scale arcs).** On all 54 photos: clean 118, poor 52, missed 48.
"+ white core" is a near tie (one fewer clean, two fewer missed); with 53 lights the difference is noise. It was not chosen
because the rule picks the most clean boxes, and because it needs more browser code; it stays a candidate.
On the author's photos the chosen setting changes nothing on the Mazda (the finder already boxes its icons) and removes
needle and arc boxes on the RAV4 and on the key-fob message display. Washed-out daylight lights are still missed.
The browser port gives identical boxes to Python on all 36 test photos (20 own, 16 Roboflow valid/test).

**Display decisions (website).** Spots the model calls `none` are not hidden: at the site's penalty, hiding every `none`
spot would hide about 1 in 4 real lights. They are collapsed under "Other bright spots" only when the model is at least
80% sure (pooled dev sets: 43 of 134 junk spots collapsed, 1 of 113 real lights); below that they stay visible as
"Unsure, probably not a warning light". Candidate light types are only listed from 10% up.

**Draw-a-box fallback (website).** The visitor drags a box around a light the finder missed; only that area is classified.
Tested in headless Chromium at desktop and phone width (no sideways scrolling, no errors). On the RAV4 daylight photo,
three hand-drawn boxes around washed-out lights gave two low-confidence "possible" answers and one "probably not a
warning light", so drawing gets the light to the model but does not make washed-out lights easy to read.

## 9. Earlier results (v3, real photos from the author's two cars)
- Real crops split by car: Mazda dev 77.2% overall (22 of 34 real lights, `none` 84%); Toyota RAV4 held-out 70% (6 of 20 real lights, `none` 97%); night 61% (3 of 20). The RAV4 set was used once and is no longer a clean test.
- End to end on one labeled Mazda photo: the top guess was right for 4 of 7 lights at 40-79% confidence; at the 0.9 threshold two "identified" claims were made and both were wrong (clipped fragments).
- Synthetic-only improvements v1 to v3 raised Mazda dev from 18% to 77%.

## 10. Open items
Done since this log started: v5b is exported and live (`docs/model/`), the `none` penalty (1.0) and the confidence tiers are
in the app, the model's top candidates are shown when it is unsure, and the light finder was retuned (section 8).
1. **A fresh test set** (a new dataset through the same crop pipeline), never used for any decision, before quoting a final
   accuracy. `external_test` was used once for v4 and is no longer clean.
2. **Check the advice text.** Every entry in `docs/meanings.json` is marked `"source": "to verify"`: confirm each meaning and
   recommended action against owner's manuals and record the source.
3. **brake_system against parking_brake.** Text icons helped (5.3 to 9.3 of 17 between v5a and v5b), but this pair is still the
   most common confusion between two lights.
4. **Weak classes:** battery and oil pressure are still often called `none`; more distinct drawings for them are the next step.
5. **Queued one-change experiments:** v5c (overexposed white-hot core with bloom, aimed at the night set) and v5d (the printed
   word CHECK under the check-engine symbol). Both rest on few examples: lower priority.
6. **Light finder:** washed-out daylight lights are still missed (the draw-a-box fallback covers them); "white core" is a
   near-tie candidate (section 8); a trained detector needs box-labeled photos that may be used for training.
7. **Icon hygiene:** check `Kontrollleuchte_Parkbremse_black` (in brake_system), `A07_Extra_high_beam` and the `B03`/`B04`
   brake-circuit icons for misfiling; keep look-alikes such as `coolant_temp` and the warning triangle out of `icons/none`.
8. **Re-exporting the model:** `src/export_onnx.py` must keep the website settings in `docs/model/config.json`
   (threshold 0.9, none_penalty 1.0, possible_threshold 0.4, collapse_none_from 0.8).
