# v7 Attention Route

Status: first offline screen.

## Motivation

v6 showed that answer-axis attention/MLP conflict can be large while still
collapsing under the same-Delta projection baseline. v7 asks a cleaner upstream
question:

> Before the MLP answers back, where did attention route the model?

In the emoji shorthand:

```text
eye + compass = attention route
```

For ANLI YES/NO prompts, use the same frozen `W_u` token-bucket contrast as v6:

```text
u = mean(W_u[NO bucket]) - mean(W_u[YES bucket])
```

For each layer `L` at the commit token:

```text
a_L = attention residual write
m_L = MLP residual write
d_L = a_L + m_L
p_a(L) = <u, a_L>
p_m(L) = <u, m_L>
p_d(L) = <u, d_L>
```

Positive values point toward NO/contradiction; negative values point toward
YES/entailment.

## Primary Screen

The first locked route statistic is the incremental OOF AUROC of signed
attention-route features over existing null and route-size controls:

```text
attention-route | null + route-size
```

The attention-route panel summarizes only `p_a`:

```text
mean(p_a)
mean early/mid/late p_a over the captured layer window
p_a at the layer with largest |p_a|
```

It is paired with an absolute-attention budget floor:

```text
abs-attention-budget | null + route-size
attention-route | null + route-size + abs-attention-budget
```

If the signed route vanishes after absolute route strength, v7 is measuring
route intensity, not route direction.

## Override Screen

Once attention route is measured, ask what the MLP does with it:

```text
mlp-readout | null + route-size + attention-route
final-readout | null + route-size + attention-route
override | null + route-size + attention-route + abs-attention-budget
```

The override panel is descriptive until it survives controls. It contains:

```text
mean(-p_a * p_m)             # projection conflict
fraction(sign p_a != sign p_m)
fraction(final sign follows m rather than a)
mean signed override margin
```

The clean Knowledge Veto shape is:

```text
attention route points one way
MLP readout points the other way
final readout follows the MLP
```

## Promotion Rule

Do not promote v7 unless:

1. `attention-route | null + route-size + abs-attention-budget` clears zero.
2. The override statistic adds beyond attention route and budget.
3. Shuffled labels are approximately zero.
4. The effect replicates in a pre-registered Qwen/Llama panel.
5. The sealed run moves to nested-OOB calibration.

The first implementation can run offline from v6 projection-veto dumps because
those dumps already persist `p_a`, `p_m`, and `p_d` per layer. New model forwards
are needed only for head-level attention weights, attention entropy, or finer
component attribution.
