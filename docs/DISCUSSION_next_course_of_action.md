# Discussion: Next Course of Action

Date: 2026-09-10
Founder-to-founder discussion doc, synthesizing three documents added to `docs/`
since the last discussion round:

- `tooth_shade_pipeline_raw_findings.md` — your raw working notes (brainstorm +
  experiment history + paper takeaways + rankings)
- `top_10_arc_segmentation_solutions.pdf` — a cleaned-up ranked summary of the
  same material
- `41598_2024_Article_69314.pdf` — Adel et al. 2024, *"Clinical audit of an
  artificial intelligence (AI) empowered smile simulation system,"* Scientific
  Reports 14:19385 — the actual paper behind the "smile-arc" references

This supersedes/extends `DISCUSSION_segmentation_next_steps.md` (2026-09-05) —
that file's threshold finding (0.9 > 0.5 for our use case) and root-cause
hypothesis (bad prompt, not bad model) both hold and are reconfirmed below.

---

## 1. What's actually new here vs. the 2026-09-05 discussion

The two new working documents (raw findings + top-10 PDF) are internally
consistent with each other and with our earlier conclusions — no contradiction,
just much more detail and a concrete ranked roadmap. Two genuinely new,
good ideas worth calling out:

1. **Adaptive Interior Enamel ROI** (§6 of the raw findings) — instead of a
   fixed "central 30%" crop (the base paper's approach) or the full mask
   (our current approach), use a **distance transform** on the predicted mask
   to keep only pixels deep inside the tooth interior, scaled to each tooth's
   own size. This is strictly better than a fixed 30% crop because it adapts
   to tooth geometry instead of applying one arbitrary fraction to every tooth
   regardless of size — worth implementing over the paper's simpler version.
2. **Our dataset already has YOLO-format bounding box labels that we are not
   using.** This is the single most actionable fact in the whole document —
   it means "train a tooth detector to fix the prompt problem" is not a new
   annotation effort, it's free supervision sitting unused in the existing
   NCT dataset. This directly de-risks Rank #1/#4 in the ranking below.

## 2. Reality-checking the third document (the smile-arc paper)

Worth being precise about what `41598_2024_Article_69314.pdf` actually shows,
since it's easy to over-read as support for the smile-arc QA idea:

**What it is**: a clinical trial measuring whether Invisalign's SmileView tool
*predicts* a patient's future post-treatment smile accurately, in 24
orthodontic patients. It compares AI-simulated smile vs. the smile that
patient actually had after ~18 months of treatment.

**What it is NOT**: it is not a study about detecting or measuring smile-arc
geometry from a photo algorithmically. No segmentation, no landmark detection,
no image processing pipeline — all ten measurements were done by hand with a
digital caliper and image tool.

**The important nuance**: for the *qualitative* variables, this paper's own
result is that predicted smile arc had **poor agreement** with actual outcome
(P=0.002, "cannot be used reliably for prediction") — along with most
posterior tooth display and lower incisor exposure. Only two things showed
reliable agreement: lip line, and five of the six *quantitative* geometric
ratios (philtrum height, commissure height, smile width, buccal corridor,
smile index — not maxillary inter-canine width).

This doesn't undermine our smile-arc idea, but it does tell us something
useful: **don't build one "ideal smile arc" template and score deviation from
it** — the paper's own data confirms real smile arcs vary a lot between
people (parallel/consonant, straight, reverse all occur naturally, and even
within one person pre/post treatment they can differ). This is exactly the
"patient-specific geometry consistency score" framing already in the top-10
PDF (§ "Key update"), not a fixed-arc-template score — good that our plan
already avoided the mistake this paper would have exposed. The five reliable
*quantitative ratios*, however, are a better foundation for a QA/consistency
signal than "arc shape" alone — things like smile width and buccal corridor
are simpler to measure from landmarks and this paper gives some real-world
expected ranges (e.g., smile width ~7.2mm normalized, buccal corridor
~1.5mm) to sanity-check against.

**Practical takeaway**: cite this paper for smile-geometry *definitions* and
*terminology* (it has good precise definitions of smile line, smile arc,
buccal corridor, smile index) and for the point that no single ideal arc
exists — not as evidence that arc-matching is a validated QA technique.

## 3. Does this change the recommended order of operations?

No — if anything it reinforces the plan already written down on 2026-09-05.
Both new documents converge on the same top-of-list move independently
derived from our own experiment history:

> Fix the prompt first (YOLO or landmarks), keep the existing MedSAM
> checkpoint, re-test — before touching architecture.

The ranking tables in both new documents (raw findings §34, PDF page 1) also
agree with each other exactly (same 10 items, same scores) — they're two
formats of the same ranking exercise, not independent validation.

## 4. Consolidated recommended order (supersedes the shorter 2026-09-05 list)

| # | Step | Cost | Status |
|---|------|------|--------|
| 1 | `MASK_THRESHOLD = 0.9` | None | **Done** — confirmed 0.9 cleaner than 0.5 on real 15-image comparison |
| 2 | Train a small YOLO detector on the **existing, unused YOLO labels** in the NCT dataset | Low | Not started — highest-leverage next step |
| 3 | Generic bbox vs YOLO bbox vs mouth-landmark bbox, same MedSAM checkpoint | Low | Not started |
| 4 | Small-blob cleanup (not blind largest-component — an anterior row is multiple components) | None/very low | Not started |
| 5 | Full-mask LAB vs central-30% LAB vs **adaptive interior enamel ROI** (distance-transform based) | None | Not started — do after mask quality is settled, since ROI choice depends on having a trustworthy mask |
| 6 | Patient-specific smile-geometry QA (landmarks → expected ratios, not a fixed ideal arc) | Low | Not started |
| 7 | Boundary-aware loss (Dice+BCE+boundary term) | Moderate | Only if step 3 still leaks after a real prompt |
| 8 | Frozen encoder + dental adapters | Moderate | Only after step 7, if still needed |
| 9 | Light HQ-SAM swap | Moderate | Only if MedSAM itself proves to be the ceiling, not the prompt |
| 10 | Distillation to a mobile student | Later | After a strong teacher exists |

Step 2 is the one concrete piece of new, actionable work this round of
documents surfaced — training a tooth detector was previously discussed only
abstractly; now it's clear the labels for it already exist in our dataset and
nothing new needs to be collected.

## 5. One thing to watch before going further

Both new documents are still **entirely unimplemented, unvalidated proposals**
— the ranking scores (9.3, 9.0, etc.) are our own prioritization judgment
called out explicitly as "not published benchmark scores," not measured
results. The only two things in this entire stack of documents that are
actually *verified on our data* so far are: (a) DSC 0.8838 on the frozen
checkpoint, and (b) the 0.5-vs-0.9 threshold comparison. Everything else —
YOLO detector performance, adapter parameter counts, boundary loss effect,
enamel ROI improvement — is a hypothesis until run. Worth keeping that
distinction sharp so the roadmap doesn't get treated as already-achieved
progress in future planning.

## 6. Suggested immediate next action

Given the dataset already has YOLO labels sitting unused, the highest-value,
lowest-cost next concrete step is **step 2**: train a small YOLO detector on
the existing NCT YOLO annotations, then re-run inference with `infer.py`
using the detector's box instead of the generic centered box, on the same
15-image sample used in the threshold comparison, so the result is directly
comparable to what's already been observed. No new labels, no MedSAM
retraining, and it directly tests the root-cause hypothesis that's been
consistent across every document so far.

## 7. Addendum (2026-09-10, same day): decision — Rank #4, with a richer prompt

Explicit answer to "which of the top-10 do we go with": **Rank #4 (YOLO/
landmarks → current frozen MedSAM checkpoint)**, not Rank #1 (Light HQ-SAM
swap). Rank #1 changes both the model and the prompt at once — if it works
we won't know which change actually fixed things. Rank #4 changes exactly
one variable (the box) against the same checkpoint we already have DSC and
threshold data for, so it's the only experiment that can confirm or kill the
"bad prompt is the root cause" hypothesis cleanly.

This addendum also folds in a refinement on *how* to use smile-line geometry
inside that step, framed as three separate jobs rather than one "detect arc"
step:

**(a) Localization — feeds directly into step 2/3 above.** Lip/mouth
landmarks → smile geometry (lip line, arc type, smile width/index,
lower-incisor exposure), computed once per photo, used as the source for the
box prompt. This is what "YOLO/landmarks" in step 3 concretely means.

**(b) Boundary guidance — a cheap addition on top of (a), not a separate
experiment.** Use the detected **lip line** (how much gingiva is visible) as
a post-processing constraint on MedSAM's output mask — erode/exclude a
margin band near the detected gingival landmark — rather than a training
signal. No retraining required; this targets gum leakage (our actual
observed failure) more directly than the box prompt alone, and stacks on top
of step 3 for free.

**(c) Quality control — independent of the segmentation-model decision,
build regardless of how (a)/(b) turn out.** Three pieces, all model-agnostic
so they don't need to wait on the Rank #4 vs Rank #1 outcome:
  - Normalized geometry features (`tooth_width/smile_width`,
    `mask_area/mouth_area`) — resolution/zoom-invariant, stored in
    `meta.json`, also useful later as module 6 input.
  - A composite **Dental Plausibility Score** (arc consistency + lip-line
    consistency + tooth-mask location + upper/lower separation + mask area +
    buccal-corridor consistency) replacing reliance on MedSAM's own
    confidence alone — this is step 6 in the table above, made concrete.
  - A **retake/quality gate** before segmentation even runs (blur, glare,
    exposure, extreme crop/pose, insufficient tooth exposure) — already
    listed as a QA idea in the raw findings; this makes it an explicit
    pipeline stage rather than a future metric.

**Important caveat carried over from §2**: the plausibility score should
compare the predicted mask against *that patient's own* observed geometry,
not one "ideal" arc — the smile-arc paper's own result (poor agreement on
arc-type prediction) is exactly why a fixed template would be the wrong
design here.

**Concrete next implementation task**: build landmark-derived smile geometry,
use it both as (a) the MedSAM box prompt and (b) a post-hoc gum-boundary
trim on the existing 0.8838 checkpoint, then re-run on the same 15-image
sample already used for the threshold comparison for a direct before/after.
Build the plausibility score and retake gate alongside it, since neither
depends on the Rank #4 vs Rank #1 decision.

## 8. Correction (2026-09-10, later same day): landmarks don't work on this dataset

Before implementing §7, checked whether MediaPipe FaceMesh (or any face
landmarker) is actually viable on the images we've been testing against.
Pulled 10 of the 15 val samples from the original threshold-comparison run
and looked at each "Original" panel directly.

**None of them contain a face.** These are clinical/stock intraoral photos —
cheek retractors, tight crops on just the mouth, sometimes only lips and
chin, one is a stock photo with hand-drawn smile-line reference marks
already on it, one has a clinic watermark. Zero out of 10 checked show eyes
or a nose. MediaPipe's face detector needs that context to find a face
bounding box at all — on this dataset it would fail to detect a face on
most/all images, which would have silently broken the landmark-based prompt
plan in §7(a)/(b) before it produced anything.

**This does not kill the smile-geometry idea, it relocates it.** Landmark-
based localization and lip-line boundary guidance only apply to genuine
full-face patient selfies — which is exactly module 4's (BiSeNet) assumed
input domain, not the NCT segmentation dataset we've been evaluating
against. The two modules have been implicitly built for two different photo
domains (module 3 tested on tight clinical crops, module 4 assumed full
face) and this is the first time that mismatch has mattered for a design
decision.

**Revised priority — YOLO moves up, landmarks move to a separate track:**

- The YOLO half of Rank #4 is unaffected and becomes the primary
  localization fix: it trains directly on the NCT dataset's own images (the
  same tight intraoral crops), so it has no face-visibility requirement.
  Confirmed the `labels/` YOLO `.txt` bbox files referenced in
  `docs/README.md` actually exist alongside the masks — nothing new to
  collect, same conclusion as §1 point 2, now sharpened to "YOLO, not
  landmarks, is the near-term fix for this dataset."
- Landmark-derived smile geometry (lip line, arc, normalized ratios,
  plausibility score) stays a real idea but only makes sense once real
  full-face patient photos are in play — i.e. it's a module-4/deployment-time
  concern, not something to build against the current NCT val set.

**Revised concrete next step**: train a small YOLO detector on the existing
NCT `labels/*.txt` annotations, swap `get_bbox_prompt()` in
`module_3_medsam_inference/infer.py` to use the detector's box instead of
the generic centered box (same 0.8838 checkpoint, nothing else changed), and
re-run on the same 15-image sample used in the threshold comparison for a
direct, comparable before/after.
