# FACE Corpus Sourcing Notes - 2026-05-08

## Goal

Prepare a production-scale FACE training pool ahead of the 10k gate. The target is eventually 350k usable, high-quality, FACE-tokenizable meshes, not merely 350k downloaded assets.

## Current Source

Primary source: `cindyxl/ObjaversePlusPlus` annotations.

Reason: it gives explicit quality and semantic flags, so we can avoid wasting GPU/disk on scenes, multi-object packs, transparent assets, and lower-confidence rows before asset download.

## Strict Pool A

Path:

```text
.codex_outputs/face_source_pool_objpp600k_20260508_061413_managed
```

Archive:

```text
.codex_outputs/face_source_pool_objpp600k_20260508_061413_managed.tar.gz
```

SHA-256:

```text
7b04e788297869201ceae44c20e0806be83d47d324c923fcac04207be7c6201f
```

Filter:

```text
quality >= 2
reject is_scene
reject is_multi_object / multi_object / multiple_objects / multi-object
reject is_transparent / transparent / transparency
```

Result:

```text
scanned: 789,195
selected: 383,397
quality 2: 156,142
quality 3: 227,255
estimated usable at 55% strict-token pass: 210,868
estimated usable at 65% strict-token pass: 249,208
```

Takeaway: Objaverse++ high/superior-only is a strong pool, but probably not enough by itself for 350k usable strict FACE samples.

## Reserve Pool B

Path:

```text
.codex_outputs/face_source_pool_objpp_minquality1_20260508_061630_managed
```

Archive:

```text
.codex_outputs/face_source_pool_objpp_minquality1_20260508_061630_managed.tar.gz
```

SHA-256:

```text
670782fb1b7ea1a0c765b1a57d986341a622d32d9b049d445d952d8248a383d4
```

Filter:

```text
quality >= 1
same semantic rejects as Pool A
selection_mode: quality_ranked
```

Result:

```text
scanned: 789,195
selected: 490,837
quality 1: 107,440
quality 2: 156,142
quality 3: 227,255
estimated usable at 55% strict-token pass: 269,960
estimated usable at 65% strict-token pass: 319,044
```

Takeaway: even medium-plus Objaverse++ may still fall short of 350k usable unless the strict-token pass rate is around 71% or higher.

## Current Implication

For a real 350k usable run, we likely need one of these:

```text
1. Higher pass rate from the current strict geometry pipeline.
2. Additional curated sources beyond Objaverse++.
3. A separate reserve lane from Objaverse raw/Objaverse-XL annotations, then our own geometry/visual filters.
4. A slightly lower final usable target for the first production run, such as 200k-300k, if the 10k gate gives strong quality.
```

The current 10k A100 gate should tell us actual pass rate and training signal. Do not commit to 350k until that gate reports real strict passing counts and AR quality.

## B2 State

The local environment currently exposes only:

```text
B2_TOKEN=set
B2_KEY_ID=unset
B2_APP_KEY=unset
B2_BUCKET=unset
```

`B2_TOKEN` is not JSON and not `key_id:application_key`, so `rclone` cannot authenticate yet. The upload/download scripts now support `B2_TOKEN` if it is provided as JSON with `keyId`/`applicationKey` or as `key_id:application_key`.

## Next Sourcing Steps

1. Harvest actual strict pass-rate from the active 10k A100 gate.
2. If pass rate is strong, use Pool A as the first production source.
3. If Pool A projected usable is short, add Pool B samples after ranking by quality and strict geometry success.
4. If still short of 350k, source a third reserve pool from Objaverse/Objaverse-XL or other curated 3D repositories, but keep the same strict gates.
5. Upload Pool A and Pool B archives to B2 once `B2_KEY_ID` plus `B2_APP_KEY`, or an equivalent JSON/colon `B2_TOKEN`, is available.
