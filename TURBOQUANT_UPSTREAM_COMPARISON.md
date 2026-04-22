# TurboQuant Upstream Comparison

Date: 2026-04-21

## Scope

I fetched `origin/main` and checked the public upstream GitHub state before
writing this report.

- Fetched upstream tip: `origin/main` at `9d567497e`
- Local implementation under review: current `HEAD` at `1bbe2fafd`
- Public GitHub PR search found one TurboQuant PR in `facebookresearch/faiss`:
  PR [#5049, "Add TurboQuant (CPU)"](https://github.com/facebookresearch/faiss/pull/5049),
  authored by Mistobaan and still open as of 2026-04-14
- `git grep` on `origin/main` found no `TurboQuant`, `QT_tqmse`, or `SQtqmse`
  symbols

Conclusion: there is no separate public upstream TurboQuant implementation in
the fetched upstream refs today. This report therefore compares:

1. upstream `origin/main` as it exists today
2. your current TurboQuant branch/worktree implementation

If there is another upstream branch or internal diff you want compared, this
report should be rerun against that exact ref.

## Snapshot

| Area | Upstream `origin/main` | Your branch (`HEAD`) | Take |
| --- | --- | --- | --- |
| TurboQuant support | None visible in public upstream tip | Dedicated `IndexTurboQuantMSE` / `IndexTurboQuantProd` plus `SQtqmse4/8` | Your branch is the only end-to-end TurboQuant implementation I could verify |
| SQ integration | Keeps existing `QT_0bit` / `SQ0` path | Replaces `QT_0bit` with `QT_tqmse_4bit` / `QT_tqmse_8bit` | Strong feature add, but with a regression risk |
| API surface | No TurboQuant public API | C++, Python SWIG, index factory, benchmark harness, Metal backend | Much more usable, but much larger surface area |
| Product variant | None | `TurboQuantProdQuantizer` and `IndexTurboQuantProd` | Your branch covers more of the paper |
| Benchmarking | No TurboQuant benchmark | `benchs/bench_turboquant.py` on SIFT, GloVe, DBpedia | Good for evaluation and adoption |
| Apple Silicon path | None | Metal backend, cloner, tests, docs | Major product advantage, major upstreaming cost |
| Change footprint | Minimal, no TurboQuant | `53 files changed, 7878 insertions, 956 deletions` vs upstream | Review and maintenance burden is materially higher |

## What Your Implementation Does Well

- It is complete enough to use, not just to discuss. You added standalone index
  types, scalar-quantizer integration, serialization hooks, Python exposure,
  tests, and a benchmark harness.
- The core codec is faithful to the TurboQuant paper structure:
  - `TurboQuantMSEQuantizer` uses random rotation plus an analytical scalar
    codebook on `[-1, 1]`
  - `TurboQuantProdQuantizer` composes a `(b - 1)`-bit MSE stage with a 1-bit
    QJL residual stage
  - arbitrary-norm vectors are handled by storing the norm in the code
- The implementation is practical, not just theoretical. It exposes
  `IndexTurboQuantMSE`, `IndexTurboQuantProd`, `SQtqmse4`, `SQtqmse8`, and a
  dedicated `bench_turboquant.py` workflow.
- The test story is stronger than a typical first pass:
  - dedicated TurboQuant correctness and serialization tests
  - scalar-quantizer encode/decode/search coverage
  - deserialization validation for TurboQuant-trained payload sizes
- The Metal work is a genuine differentiator. For Apple Silicon users, your
  branch is meaningfully ahead of upstream.

## Main Advantages Of Upstream Mainline

- Upstream mainline is much easier to reason about and maintain because it does
  not carry TurboQuant-specific public API, SWIG, packaging, or Metal backend
  complexity.
- It preserves existing `QT_0bit` / `SQ0` behavior. That is already wired into
  the scalar-quantizer scanner path and test suite, so upstream has less user
  visible churn.
- It stays aligned with the current scalar-quantizer dynamic-dispatch structure.
  That makes it a cleaner base for incremental changes.

## Main Advantages Of Your Branch

- Users can actually use TurboQuant today.
- Your branch covers both MSE-oriented and inner-product-oriented TurboQuant.
- It reaches all the way to Python and benchmarking, which is what makes a new
  codec adoptable.
- It is already positioned for Apple Silicon acceleration rather than stopping
  at a CPU-only prototype.

## Main Risks / Cons In Your Branch

- The landing surface is large. Relative to `origin/main`, this is a broad
  cross-cutting change, not a narrow codec patch.
- The branch removes existing upstream `QT_0bit` / `SQ0` behavior and its
  tests, replacing that slot with `QT_tqmse_4bit` / `QT_tqmse_8bit`. That is a
  regression risk unrelated to TurboQuant itself.
- There are now two overlapping public faces for TurboQuant:
  - dedicated `IndexTurboQuant*` classes
  - scalar-quantizer flavors `SQtqmse4/8`
  This is good for usability, but it increases maintenance and documentation
  cost.
- The scalar-quantizer integration is narrower than the standalone codec API:
  - standalone MSE supports `1..8` bits
  - standalone Prod supports `1..9` bits
  - SQ integration currently exposes only 4-bit and 8-bit MSE
  That mismatch may be surprising to users.
- The analytical codebook path is elegant, but it is much less data-driven than
  the existing scalar-quantizer training model. That makes it a worse fit for
  users who expect SQ training to adapt directly to the observed value
  distribution.
- The SWIG and Metal pieces add real fragility:
  - wrapper ordering matters for downcasting
  - packaging/runtime loading gets harder
  - platform-specific failures become part of the support burden

## Main Risks / Cons In Upstream Mainline

- Upstream mainline does not provide TurboQuant at all right now.
- There is no public CPU index, no product variant, no benchmark harness, no
  Python-level TurboQuant surface, and no Apple Silicon path.
- From a product perspective, upstream simplicity is achieved partly by not
  solving the TurboQuant problem yet.

## Bottom Line

If the goal is immediate capability, your branch is substantially ahead of
public upstream. It is the only verified end-to-end TurboQuant implementation in
this comparison, and it adds real product value through Python exposure,
benchmarks, and Metal support.

If the goal is upstream acceptance, upstream currently has structural
advantages: less API churn, less packaging complexity, no `QT_0bit` regression,
and a much smaller review surface. The biggest issue in your branch is not the
TurboQuant math; it is the size and breadth of the integration.

## Recommended Landing Strategy

If you want the best path to upstreaming, I would split the work like this:

1. Land core CPU TurboQuant codec support first.
2. Preserve `QT_0bit` and avoid replacing existing SQ behavior.
3. Add scalar-quantizer factory integration as a separate step.
4. Add dedicated `IndexTurboQuantMSE` / `IndexTurboQuantProd` API after the
   codec layer is accepted.
5. Keep benchmarks, SWIG extras, and Metal backend as follow-up PRs.

That keeps the strongest parts of your implementation while reducing the parts
most likely to block acceptance.
