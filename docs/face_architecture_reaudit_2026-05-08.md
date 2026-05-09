# FACE Architecture Re-Audit - 2026-05-08

## Verdict

Do not launch another A100 FACE run until the strict lane is aligned around the
paper method and cited implementation lineage.

The latest Opus-aligned run proved that `token_concat_project + paper_chain`
can drive selection loss down, but it did not prove autoregressive mesh quality.
It failed free-running AR badly, so loss alone is not a sufficient signal.

## Paper-Critical Components Checked

### Face Decoder Attention

FACE specifies causal self-attention over previous face tokens followed by
cross-attention to the VecSet at every decoder layer:

- `H'_l = CausalSelfAttn(H_l)`
- `H_{l+1} = CrossAttn(Q=H'_l, K=C, V=C)`

Implementation status:

- `FaceDecoderBlock` applies causal self-attention with a triangular future mask.
- Each block then cross-attends to the encoder VecSet as key/value.
- The implementation uses standard PyTorch attention rather than FlashAttention,
  which affects speed/memory, not the mathematical attention contract.
- The incremental path is intended to match full causal recompute; tests cover
  this but require a Torch-capable environment to run.

Verdict: structurally faithful. The attention path is not the highest-probability
cause of current AR collapse.

### Previous-Face Embedding

FACE text says each previous face `f_{i-1} in R^9` is projected by a lightweight
Face Pooling MLP. Figure 2/caption also frames each face as nine quantized
coordinate tokens.

Implementation status:

- `token_concat_project` preserves all nine coordinate-token identities and order
  before projection to one face token.
- `continuous_mlp` and `discrete_sum` are retained for checkpoint compatibility
  and ablation only.

Verdict: `token_concat_project` remains the best strict-lane default.

### CausalMLP

FACE says the CausalMLP predicts coordinate token `j` conditioned on latent face
vector `h_i` and previous coordinate tokens inside the same face. FACE does not
publish exact CausalMLP code, but it cites TreeMeshGPT for this component.

Implementation status:

- `legacy_concat` matches the public TreeMeshGPT pattern most closely: staged
  coordinate heads, with each later head conditioned on embeddings of previous
  coordinate tokens.
- `paper_chain` is a reasoned experimental fill-in with shared slot-aware prefix
  projection. It is not the closest public-code match to the cited CausalMLP.
- The Opus run used `paper_chain`; therefore it should be treated as an ablation,
  not as final paper-faithful evidence.

Verdict: strict FACE lane should default to `legacy_concat`; `paper_chain` stays
available for ablation.

## Repo Corrections Made

- Builder default: `causal_mlp_variant=legacy_concat`.
- Training CLI default: `--causal-mlp-variant legacy_concat`.
- Thunder launch/default wrappers: `CAUSAL_MLP_VARIANT=legacy_concat`.
- Scale-readiness paper knob: expects `legacy_concat`.
- Decode/incremental/optimizer tests now cover both `legacy_concat` and
  `paper_chain` where applicable.

## What This Means For The Next Run

The next A100 gate should not repeat the failed `paper_chain` configuration.
Run a bounded memorization/closure gate with:

- `face_embedding_variant=token_concat_project`
- `causal_mlp_variant=legacy_concat`
- strict paper knobs: 128 bins, 8192 points, 2048 VecSet tokens, latent 64,
  Muon, bf16
- no augmentation for the first micro-closure proof, then paper augmentation only
  after train AR closure is demonstrated

Promotion remains blocked unless free-running AR becomes coherent. Teacher-forced
loss and selection loss are useful diagnostics, not sufficient go signals.
