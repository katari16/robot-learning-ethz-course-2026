# Video Script — HW1 Video Submission (~1 min)

---

## Slide 1: Ex3 — MNIST MLP (brief, ~15 sec)

**What to say:**

"For exercise 3 I implemented a full MNIST classification pipeline from scratch — including
a custom Linear layer with Xavier initialization, LayerNorm, RMSNorm, Dropout, an MLP, and
a training loop using my own cross-entropy implementation. As you can see in the loss curve,
the model converges smoothly over 5 epochs, reaching a final test accuracy of **97.75%**, well
above the required 70%."

**Show:** `plots/ex3_training_loss.png`

---

## Slide 2: Ex4 — Setup (brief, ~10 sec)

**What to say:**

"For exercise 4 I implemented a tiny ViT-style classifier for MNIST — patch tokenization
with 4×4 patches giving 49 tokens, a patch embedding, learnable positional embeddings, and
pre-LayerNorm transformer encoder blocks. I then ran a controlled ablation replacing the
standard FFN with two GLU variants from Shazeer 2020: GEGLU and SwiGLU, using the 2/3 width
rule to keep parameter counts comparable (~105k params each)."

---

## Slide 3: Ex4 — Results (main focus, ~25 sec)

**What to say:**

"As you can see in the accuracy plot, both GLU variants outperform the FFN baseline
significantly. After 5 epochs: FFN reaches 94.5%, GEGLU 96.3%, and SwiGLU 95.9%.

The GLU variants also converge faster — already after epoch 1 they are ~3 points ahead.

**On reproducibility and statistical significance:** All three runs used seed 0 and identical
hyperparameters, so we have a single measurement per variant. The observed differences
(~1.4–1.8 points between FFN and GLU variants) are consistent with what's reported in the
literature, but with only one run per variant we cannot claim statistical significance — a
single run could be a lucky or unlucky seed. To properly validate this, we'd want to run
multiple seeds and use a statistical test. The 2/3 width rule also only approximately
equalizes parameters (~105k FFN vs ~104.9k GLU), so compute is not perfectly matched either.

That said, the direction of the result — gating improving convergence speed and final accuracy
— is consistent and aligns with the Shazeer 2020 findings."

**Show:** `plots/ex4_accuracy_comparison.png`

---

## Slide 4: Ex4 — Conclusion (5 sec)

**What to say:**

"In summary: GLU variants improve both convergence speed and final accuracy in this setting,
though more runs would be needed to make a statistically rigorous claim."

---

## Key numbers to have ready

| Variant | Final acc | Best acc | Params |
|---------|-----------|----------|--------|
| FFN     | 94.48%    | 94.48%   | 104,970 |
| GEGLU   | 96.31%    | 96.31%   | 104,882 |
| SwiGLU  | 95.89%    | 95.89%   | 104,882 |

Ex3 final test accuracy: **97.75%**
