# v6 Projection Veto

Status: draft spec / branch seed.

## Motivation

v5 residual friction asked whether the attention write `a` and MLP write `m`
fight inside a transformer block. The same-Delta benign baseline showed that
most of that fight was residual norm budgeting: `a` and `m` can oppose each
other in the residual stream without carrying answer-relevant conflict.

v6 moves the question from residual-space friction to projection-space conflict:

> We do not care if `a` and `m` fight in the hallway. We care if they fight over
> the steering wheel.

That means: measure whether `a` and `m` oppose each other along directions that
actually hit the logits / answer contrast.

## Projection Choice

Use the unembedding/readout matrix `W_u` directly, but not as a single raw top-1
logit. The primary projection is a frozen bank of logit-contrast directions.

For a contrast `(i, j)`:

```text
c_ij = W_u[i] - W_u[j]
```

If the model applies a final norm before the unembedding, the first-order
residual-space readout direction at hidden state `h` is:

```text
u_ij(h) = J_norm(h)^T c_ij
```

where `J_norm(h)` is the Jacobian of the final norm. A cheaper first pass may use
`c_ij` directly as a logit-lens proxy, but the sealed version should use the
local norm-linearized direction when practical.

Recommended contrast ladder:

1. **Task contrast primary**: frozen YES-vs-NO / entail-vs-contradict token
   buckets. This is `W_u`-dependent but label-independent at scoring time.
2. **Commit contrast secondary**: model top-1 vs runner-up at the commit token.
   This is unsupervised but can be brittle because the axis changes per sample.
3. **Top-k contrast bank**: all pairwise contrasts among the top-k commit logits,
   summarized by max/mean signed conflict.

Do not use the true label to choose the projection axis on the scored fold.

## Core Statistic

For each block/layer `L` at the commit token:

```text
a_L = attention residual write
m_L = MLP residual write
d_L = a_L + m_L
p_c(x) = <u_c, x>
```

Projection-space veto:

```text
proj_veto_c(L) = -p_c(a_L) * p_c(m_L)
```

Interpretation:

```text
p_c(a_L) > 0 and p_c(m_L) < 0  => MLP counters attention on contrast c
p_c(a_L) < 0 and p_c(m_L) > 0  => attention counters MLP on contrast c
p_c(a_L), p_c(m_L) same sign   => projection-space cooperation
```

Useful companion features:

```text
proj_net_c       = p_c(a_L + m_L)
proj_path_c      = |p_c(a_L)| + |p_c(m_L)|
proj_trim_c      = proj_path_c - |proj_net_c|
proj_gain_a_c    = |p_c(a_L)| / (||a_L|| + eps)
proj_gain_m_c    = |p_c(m_L)| / (||m_L|| + eps)
proj_amplify_c   = proj_path_c / (||a_L|| + ||m_L|| + eps)
```

`proj_amplify_c` is the projection-amplification diagnostic: small residual
norms can still carry large readout effects if they land on a high-leverage
unembedding direction.

## Same-Delta Benign Control

v6 keeps the v5 correction as a mandatory floor.

Construct a benign split:

```text
a'_L + m'_L = a_L + m_L
```

while matching route/cancellation magnitudes as closely as possible, but rotating
the hidden disagreement channel off the projection direction `u_c`.

Then report:

```text
net_projection_veto =
    DeltaAUROC(real proj_veto | null + route + projection-budget)
  - DeltaAUROC(same-Delta benign proj_veto | null + route + projection-budget)
```

The statistic only promotes if it beats:

```text
random-u floor
same-Delta benign floor
projection-budget floor
shuffled-label control
```

## What This Catches

False residual friction:

```text
a and m oppose in residual space
but p_c(a) ~= p_c(m) ~= 0
```

Projection-visible conflict:

```text
p_c(a) is large positive
p_c(m) is large negative
p_c(a + m) is small or sign-flipped
```

Projection amplification:

```text
||a||, ||m|| small
but |p_c(a)|, |p_c(m)| large
```

This is the v6 target: a small-looking residual event that is large in the
answer/readout geometry.

## Promotion Rule

Do not promote a v6 signal unless:

1. Its locked primary interval clears zero.
2. Same-Delta benign and projection-budget controls are approximately zero.
3. The net statistic remains positive on at least two related model families or
   a pre-registered within-family replication.
4. The sealed run uses the nested-OOB calibrator path for the real inferential
   interval.

The first pilot should be run offline where possible from schema-v3 dumps. Full
model reruns are needed only if the dump lacks the projection features or the
final-norm-linearized readout directions.
