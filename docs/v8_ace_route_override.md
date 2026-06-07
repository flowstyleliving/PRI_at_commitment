# v8 ACE Route Override

Status: first Qwen2.5 screen complete.

## Motivation

v7 tested a tempting but too-literal route definition:

```text
attention route = <W_u[NO-YES], attention_write>
```

That signed attention-write projection did not add signal on Qwen2.5. The v4
vault points to a cleaner route sensor: sealed ACE at `t=0`.

ACE does not read the attention write's answer-axis projection. It reads the
attention channel itself: JS-radius, BOS/sink mass, and value-norm-weighted
attention cells at the prefill-last-position.

v8 therefore defines:

```text
route = sealed ACE t=0 winning attention cell
response = MLP/readout projection features from v6
override = response beyond route
```

For Qwen2.5 on ANLI, the sealed ACE route cell is:

```text
attention[final_v_norm_lastq_weighted] @ step 0, sign=+1
```

## Primary Questions

Use the same ANLI R1 n=200 slice as v4/v6.

```text
ACE-route | null + route-size
MLP-readout | null + route-size + ACE-route
final-readout | null + route-size + ACE-route
override | null + route-size + ACE-route
```

Then apply the v6 guardrail:

```text
projection-veto | null + route-size + ACE-route + projection-budget
projection-veto | null + route-size + ACE-route + same-Delta
```

Interpretation:

- If ACE absorbs the MLP/final/readout lift, v6/v7 were rediscovering ACE.
- If MLP/final adds beyond ACE but projection-veto dies under budget/same-Delta,
  the extra signal is readout/budget structure, not clean veto.
- If override survives ACE plus budget/same-Delta, then we have a genuine route
  override candidate worth promoting.

## Data Flow

The first implementation uses two persisted artifacts:

1. The sealed v4 ACE profile to score the per-sample `t=0` ACE route value.
2. The v6 projection-veto feature dump for per-sample `p_a`, `p_m`, `p_net`,
   budget, and same-Delta projection features.

The script aligns both by prompt SHA-256 before scoring combined statistics.

## First Qwen2.5 Result

ANLI R1 n=200, using the sealed Qwen2.5 ACE profile and the v6 projection dump:

```text
ACE-route | null+route-size                        Delta=+0.0623 [+0.0528,+0.0743]
MLP-readout | null+route-size+ACE                  Delta=+0.0672 [+0.0532,+0.0788]
final-readout | null+route-size+ACE                Delta=+0.0579 [+0.0467,+0.0677]
override | null+route-size+ACE                     Delta=+0.0704 [+0.0580,+0.0828]
projection-veto | null+route-size+ACE              Delta=+0.0662 [+0.0552,+0.0758]
same-Delta projection | null+route-size+ACE        Delta=+0.0635 [+0.0534,+0.0732]
budget | null+route-size+ACE                       Delta=+0.0579 [+0.0439,+0.0699]
projection-veto | null+route-size+ACE+budget       Delta=+0.0025 [-0.0008,+0.0064]
projection-veto | null+route-size+ACE+same-Delta   Delta=+0.0007 [-0.0014,+0.0028]
```

Interpretation: ACE is a real route sensor, and MLP/final readout adds beyond
ACE. But the projection-veto component is still explained by the same-Delta and
budget floors. The net veto after ACE is about `+0.0027`, not a promotion-level
Knowledge Veto.

Caveats:

- The sealed ACE profile warns on winner stability (`0.69`, just below the
  `0.70` deployability threshold).
- Current code hashes differ from the sealed profile provenance, so this is a
  drift-aware pilot screen, not a sealed reproduction.
- Shuffled-label ACE-route is warm (`Delta=+0.0354 [+0.0058,+0.0561]`), showing
  the repeated-CV increment remains anti-conservative for added columns.

## Promotion Rule

Do not promote v8 unless:

1. MLP/final/override adds beyond ACE route.
2. Projection-veto/override survives projection-budget and same-Delta controls.
3. Shuffled labels remain near zero.
4. The effect replicates beyond Qwen2.5 under a pre-registered Qwen/Llama panel.
