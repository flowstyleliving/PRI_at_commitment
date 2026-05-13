## Cross-model W_u top-1 right-singular-vector character

Per-model: σ-spectrum top-1, σ-gap to top-2, top-5 positive and top-5 negative tokens projecting onto V_raw[0], targeted projections of answer-relevant tokens, modal gen_step=1 commit token, and per-sample signed Δh_jn · V_raw[0] for ctrl vs contr (N=100 stratified, seed=20260423).

### Spectrum + commit token

| Model | V | σ_1 | σ_1/σ_2 | modal gen_step=1 | frac |
|---|---:|---:|---:|---|---:|
| 🦙 Llama-3.2-3B-Instruct-4bit | 128,256 | 153.45 | 7.84× | ` Answer` | 98% |
| 🌀 Mistral-7B-Instruct-v0.3-4bit | 32,768 | 17.26 | 5.00× | `\n` | 100% |
| 🐉 Qwen2.5-7B-Instruct-4bit | 152,064 | 86.97 | 4.01× | ` NO` | 52% |
| 🐲 Qwen3-8B-4bit | 151,936 | 155.89 | 4.76× | ` Answer` | 79% |
| 🪼 Phi-3.5-mini-instruct-4bit | 32,064 | 111.67 | 1.75× | `\n` | 100% |
| 🌸 gemma-3-4b-it-4bit | 262,208 | 102.95 | 1.44× | `\n` | 100% |

### V_raw[0] top-5 tokens (positive / negative)

| Model | top-5 positive | top-5 negative |
|---|---|---|
| 🦙 Llama-3.2-3B-Instruct-4bit | `,` (+0.88), ` ` (+0.78), ` (` (+0.67), `.` (+0.62), ` and` (+0.60) | `SCRI` (-0.65), `﻿using` (-0.64), `TRGL` (-0.64), `!\n\n\n\n` (-0.64), `oolStrip` (-0.64) |
| 🌀 Mistral-7B-Instruct-v0.3-4bit | `(` (+0.13), `\n` (+0.13), `and` (+0.12), `in` (+0.12), `a` (+0.11) | `qpoint` (-0.23), `ICENSE` (-0.23), `ityEngine` (-0.22), `ERCHANTABILITY` (-0.22), `listade` (-0.22) |
| 🐉 Qwen2.5-7B-Instruct-4bit | ` ` (+0.69), `,` (+0.60), `1` (+0.59), ` (` (+0.59), `.` (+0.55) | `.IsNullOr` (-0.69), ` volunte` (-0.64), `gnore` (-0.58), ` citiz` (-0.56), `ityEngine` (-0.56) |
| 🐲 Qwen3-8B-4bit | ` neighb` (+0.69), ` porno` (+0.68), ` somew` (+0.67), ` Palestin` (+0.67), `CLU` (+0.65) | ` ` (-1.48), `\n` (-1.33), `,` (-1.27), `.` (-1.21), `1` (-1.20) |
| 🪼 Phi-3.5-mini-instruct-4bit | `provin` (+1.54), `Wikip` (+1.25), `zna` (+1.17), `vess` (+1.15), `kö` (+1.15) | `(` (-0.49), `\n` (-0.47), `in` (-0.39), `"` (-0.36), `a` (-0.34) |
| 🌸 gemma-3-4b-it-4bit | `S` (+0.74), `(` (+0.74), `g` (+0.74), `U` (+0.74), `M` (+0.74) | ` (` (-0.36), `\n` (-0.33), `\n\n` (-0.33), `  ` (-0.27), ` and` (-0.26) |

### Targeted projections onto V_raw[0]

Compares signed projections of `' YES'` / `' NO'` / `' Answer'` / `'\n'` (newline). If `' YES'` and `' NO'` project with *opposite* signs and *similar magnitudes*, V_raw[0] is a content (YES/NO) axis. If both project with the *same* sign, V_raw[0] is something else (rupture-magnitude axis or unrelated).

| Model | ` YES` | ` NO` | ` Answer` | `\n` | axis interpretation |
|---|---:|---:|---:|---:|---|
| 🦙 Llama-3.2-3B-Instruct-4bit | -0.338 | -0.088 | — | +0.576 | non-content (same-sign) |
| 🌀 Mistral-7B-Instruct-v0.3-4bit | +0.034 | -0.028 | — | +0.127 | content (YES/NO bipolar) |
| 🐉 Qwen2.5-7B-Instruct-4bit | -0.104 | +0.051 | — | +0.543 | ambiguous |
| 🐲 Qwen3-8B-4bit | +0.352 | -0.072 | — | -1.335 | ambiguous |
| 🪼 Phi-3.5-mini-instruct-4bit | — | +0.344 | +0.483 | -0.469 | ? |
| 🌸 gemma-3-4b-it-4bit | — | +0.029 | — | -0.329 | ? |

### Per-sample signed Δh_jn · V_raw[0] (N=100, ctrl vs contr)

| Model | ctrl mean ± std | contr mean ± std | Δ (contr − ctrl) | ctrl frac>0 | contr frac>0 | axis usage |
|---|:---:|:---:|:---:|:---:|:---:|---|
| 🦙 Llama-3.2-3B-Instruct-4bit | +1.302 ± 0.147 | +1.364 ± 0.263 | +0.062 | 100% | 100% | monotone same-sign (rupture-magnitude axis) |
| 🌀 Mistral-7B-Instruct-v0.3-4bit | +3.011 ± 0.411 | +4.644 ± 0.520 | +1.632 | 100% | 100% | monotone same-sign (rupture-magnitude axis) |
| 🐉 Qwen2.5-7B-Instruct-4bit | -0.295 ± 9.941 | -18.696 ± 1.327 | -18.400 | 78% | 0% | mixed |
| 🐲 Qwen3-8B-4bit | +0.770 ± 0.395 | -1.566 ± 2.547 | -2.336 | 98% | 50% | mixed |
| 🪼 Phi-3.5-mini-instruct-4bit | -9.573 ± 0.591 | -8.139 ± 0.893 | +1.434 | 0% | 0% | monotone same-sign (rupture-magnitude axis, native −) |
| 🌸 gemma-3-4b-it-4bit | -6.087 ± 0.684 | -6.149 ± 0.487 | -0.061 | 0% | 0% | monotone same-sign (rupture-magnitude axis, native −) |
