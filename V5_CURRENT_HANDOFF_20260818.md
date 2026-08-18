# HairFastGAN V5 Current Handoff

Updated: 2026-08-18

## 1. Scope of this handoff

This document records the state of the current V5 PP work after the latest
training results. It is intentionally limited to the unresolved face and
earring behaviour. It does not claim that an unverified explanation is a root
cause.

The user has stated that the latest trained outputs have shown no meaningful
improvement in the two central failures:

1. The face still shows a three-part composition rather than one continuous
   face with recovered source facial detail.
2. Earrings are still not properly recovered. This affects both long pendant
   earrings and short studs, not only long earrings.

Do not report either issue as solved until a newly trained model is evaluated
on fixed cases and the final images meet the acceptance criteria below.

## 2. Repository state

- Working directory:
  `E:\BaiduNetdiskDownload\HairFastGAN\hair_fast_fwq_bx_ppmodify`
- Git branch: `codex/earring-foreground-v18`
- Latest pushed commit:
  `e095d1bcfe44b3a6b9b92e20e5e7ad5d2a8e786f`
- GitHub branch:
  <https://github.com/zhouuu183/dniwndxsq21/tree/codex/earring-foreground-v18>
- Remote: `https://github.com/zhouuu183/dniwndxsq21.git`

The current worktree was clean immediately after the push. Local `.edge-*`
browser automation profiles were deliberately excluded and `.gitignore` now
contains `.edge-*/`; these directories contain local cache/profile data and
must never be committed.

## 3. Non-negotiable user requirements

1. `models/Embedding.py` is baseline code. Do not modify it for V5 work.
   V5-specific embedding changes belong in `models/Embedding_v5.py`.
2. Do not use or introduce anything named `superpower`.
3. The user retrains from scratch after a real policy/code change. Dataset
   generation, training inputs/losses, and inference/compositing must remain
   mutually consistent. Do not make an inference-only patch and call the
   model improved.
4. Do not keep adding threshold patches based only on final images. First
   identify the exact processing stage at which a known visible earring is
   rejected, truncated, or overwritten.
5. Existing unrelated worktree changes must not be reverted.
6. Do not modify V8 or unrelated baseline behaviour merely to address V5 PP.
7. Do not claim a cause as established unless it is supported by a comparison
   of the relevant masks/tensors or by a direct code-path comparison.

## 4. Reference implementation and its limits

The comparison implementation is:

`E:\BaiduNetdiskDownload\HairFastGAN\besthairfast\hairfast_pp修改完毕`

It is important because, according to the user, it can recover earrings very
well. It must not be copied wholesale. The user has already identified three
unacceptable behaviours in that implementation:

1. It produces a three-part face.
2. Broad earring recovery can create holes in transferred hair/background.
3. It can force an ear and earring even when the reference hairstyle truly
   covers that ear.

The next investigation must compare why that implementation recalls earrings
with the current implementation, while retaining the current project's
requirements for face continuity, no holes, and no hallucinated ear/earring.
Do not copy its facial restoration or broad ROI/background write-back logic.

## 5. Current unresolved visual failures

### 5.1 Face: three-part face remains unresolved

This is not merely a cosmetic smoothing issue. The final face contains
different image states in adjoining regions:

| Region | Current failure seen in trained output |
| --- | --- |
| Hairline and upper forehead | Often pale/white and overly smooth. Source skin texture, pores, and local tonal detail are absent. |
| Middle forehead | Does not blend continuously with either side. It can show colour blocks, grey haze, semi-transparent brush-like marks, or a visible change in brightness/sharpness. |
| Lower face | Relatively more source-like in tone and texture, but therefore visibly inconsistent with the upper face. |

Consequences that must be treated as failures:

- Skin tone and lighting change abruptly across the forehead/face.
- Texture and sharpness change abruptly rather than gradually.
- There can be horizontal or curved mask boundaries, grey halos, white lines,
  black lines, or coloured blocks.
- The source person's real facial detail is not recovered as one continuous
  facial region. Avoiding colour blocks by making the whole face smooth is
  also a failure.
- Near the hairline, source bangs, dark hair shadows, or unrelated dark
  residuals must not be returned to the skin.

Required visual target: the full face must read as one continuous face.
The hairline may be slightly smoother after shadow correction, but it must
blend progressively into source-consistent forehead and lower-face detail with
no visible processing boundary.

### 5.2 Earrings: recall remains unresolved for all major forms

The observed issue is not just weak fine detail. A visible source accessory is
frequently absent or reduced to a fragment in the final trained output even
when the relevant target ear/lobe is exposed.

#### Long pendant earrings

- The earlobe-side root, one rim, or a small bright fragment may appear.
- The pendant body below the lobe is often missing completely.
- The earring can be cut exactly where it reaches the transferred hair area;
  visually it behaves as if target hair has erased the earring rather than
  sitting behind it.
- The desired layer order in this valid case is: visible source earring in
  front, transferred hair behind the earring. Recovering the earring must not
  paste source background or source hair over the transferred hair.

#### Short studs and compact earrings

- This class fails too, so the issue cannot be described only as loss of the
  lower half of a long pendant.
- Even when the earlobe is exposed, the final result may contain no stud at
  all.
- In other samples the stud becomes only a generic bright spot, blob, or rough
  circular silhouette. Its real outline, small decorative geometry, material
  edge, and local contrast are not recovered.

#### Ear/lobe integrity

- Some outputs show a split, crack, gap, dark seam, or missing section in the
  earlobe around the accessory area.
- This is a separate failure from missing earring pixels: the skin anatomy
  itself must remain continuous while the earring is restored.

#### Required behaviour by visibility case

| Case | Required final result |
| --- | --- |
| Source has no earring | Do not create an earring, dark dot, metallic fragment, or false ear. |
| Source earring is visible and target ear/lobe is visible | Recover the complete real earring, including a short stud's identity/detail or a long pendant's body. |
| Visible earring overlaps the spatial area of transferred hair | Earring remains a foreground object; only its real instance is restored. The transferred hair stays behind it without a hole. |
| Target hairstyle genuinely covers the ear/lobe | Do not invent/reopen an ear solely to show a source earring. |
| Hoop earring | Restore the ring body only; keep target content inside the hole. Do not paste source background/hair into the hole. |

## 6. What is in the current V18 code

These changes exist in the checked-in code. They are implementation facts, not
proof that the final visual failures are fixed.

### Dataset generation: `scripts/pp_gen_v5.py`

- `DATASET_CONFIG_SCHEMA_VERSION = 18`.
- Small-profile output directory:
  `images/pp_dataset_v5_dual_ear_short_long_instance_v18_foreground_learning`
- Full-profile output directory:
  `images/pp_dataset_v5_dual_full_instance_v18_foreground_learning`
- Each dataset item now writes the additional fields:
  - `earring_learning_mask`
  - `earring_learning_reference`
  - `earring_learning_hole_mask`

### Training: `scripts/pp_train_v5.py`

- `PP_DATASET_SCHEMA_VERSION = 18`.
- The training loader reads the three V18 learning fields above.
- Horizontal flip and positive-sample selection include those fields.
- Small-profile run paths currently point to the V18 foreground-learning
  dataset/checkpoint directories.

### PP compositing: `models/postprocess_v5.py`

- The final output has a PP earring foreground mask named
  `output_pp_earring_foreground_mask` in auxiliary output.
- Earring foreground is excluded from target ear authority so the target ear
  preservation path does not overwrite that foreground mask.
- `_normal_target_face_preserve_mask()` is used during final evaluation to
  build a continuous target-face base before PP composition, excluding
  revealed-skin repair and earring areas.
- The target-output preservation path exposes
  `output_target_hair_earring_keep_mask`.

### Resolved runtime-only defect

The initial V18 training validation once failed with:

```text
NameError: name 'earring_keep' is not defined
models/postprocess_v5.py:_preserve_target_output
```

That local scope error was fixed by constructing the required PP foreground
mask in `_preserve_target_output`. The following checks passed after the fix:

```text
python -m py_compile models/postprocess_v5.py scripts/pp_gen_v5.py scripts/pp_train_v5.py
git diff --check -- models/postprocess_v5.py scripts/pp_gen_v5.py scripts/pp_train_v5.py
```

This runtime fix does not establish visual efficacy. The latest user report is
that the trained output remains effectively unchanged in the face and earring
failures described above.

## 7. Known V18 dataset/training facts

The most recent reported V18 training startup printed:

```text
Dataset items=128
earring_recovery_positive=52
covered_or_nonpositive=76
```

This says only 52/128 generated items are currently marked as positive
earring-recovery supervision. It is not, by itself, proof of the cause of
poor recall. It must be examined together with the actual masks/references
for fixed long-pendant and short-stud samples.

When code or generation policy changes, do not reuse a directory containing
parts generated under a different schema/policy. Use a new directory and
regenerate the full dataset, then retrain from scratch. For the existing V18
code without a generation-policy change, the existing complete V18 dataset
does not need regeneration merely because of the prior `NameError` fix.

## 8. Required investigation before another edit

Do not change thresholds first. Select fixed examples from all of these cases:

1. Source without earring.
2. Exposed ear/lobe with a short stud.
3. Exposed ear/lobe with a long pendant and hair behind/adjacent to it.
4. Ear truly hidden by the transferred hairstyle.
5. Hoop earring, if present in the dataset.
6. A face case with obvious three-part segmentation.

For each earring side, compare and save the following evidence in the current
and reference implementations:

```text
source image
target before PP
source parser/accessory evidence
source native instance mask
dataset earring_learning_mask (during training case)
target-ear visibility decision
earring_write_mask / PP foreground mask
output_target_hair_earring_keep_mask
final compositing alpha
final image
```

The point is to establish which first differs from the desired result:

1. Source earring not detected.
2. Detected instance is incomplete.
3. Target visibility decision closes a valid case.
4. The instance exists but a hair-overlap rule clips it.
5. PP training target/reference does not contain it.
6. The model predicts it but the final compositor overwrites it.

Only after this classification should code be modified.

For the face case, capture the target before PP, the face-preservation mask,
the PP restoration mask/output, and the final composite. Verify the exact
first mask/composite stage that separates the upper forehead, middle forehead,
and lower face. Do not treat a smooth face as an acceptable workaround for
the three-part face.

## 9. Acceptance criteria for a future claim of success

Face:

- No visible three-state split between hairline, forehead, and lower face.
- Continuous skin colour, brightness, texture, and sharpness.
- Source-consistent facial texture/detail is present across the face without
  reintroducing source hair, source bangs, or source background.
- No brush edge, grey halo, colour block, black line, or white line.

Earrings:

- Visible short studs retain actual identity and shape, not merely a blob.
- Visible long pendants are complete below the lobe when they should be
  visible, even with transferred hair behind them.
- No earlobe split, crack, or missing skin around the earring.
- No target-hair hole, pasted source background, or source-hair leak.
- No fabricated ear or earring where the transferred hairstyle genuinely
  hides the ear.
- No earring hallucination for negative/source-without-earring cases.

## 10. Commands and relevant files

From the project root:

```powershell
python scripts/pp_gen_v5.py
python scripts/pp_train_v5.py
```

Primary files to inspect together:

```text
models/Embedding_v5.py
models/ear_modules_v5.py
models/postprocess_v5.py
models/Blending_v5.py
scripts/pp_gen_v5.py
scripts/pp_train_v5.py
losses/pp_losses_v5.py
hair_swap_v5.py
```

Do not modify `models/Embedding.py`.
