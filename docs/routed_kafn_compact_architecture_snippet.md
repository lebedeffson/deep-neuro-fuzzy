# Routed KAFN compact architecture snippet (for self-contained reading)

Routed Kolmogorov-Arnold Fuzzy Network (Routed KAFN) is a neuro-fuzzy architecture that learns an interpretable routed rule dictionary and a linear output head. The full model has three core parts: (1) feature-to-term fuzzy mapping, (2) routed rule activations `h_r(x)` for rules of IF-THEN type, and (3) a linear logit head
`z(x) = theta_0 + sum_r theta_r h_r(x)`.
After training, we obtain an active rule vocabulary with rule-wise activations and coefficients. The present work does not modify Routed KAFN training; it applies post-training budgeted reduction of this active vocabulary to a target size `B` and evaluates quality, fidelity, and stability of the resulting compact subdictionary.

If the full Routed KAFN paper is not yet in print, cite the preprint or accepted-manuscript reference in this section so the architecture is externally traceable.
