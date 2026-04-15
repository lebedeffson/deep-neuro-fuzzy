# IITI'26 Camera-Ready Sections (Abstract + Related Work + References Skeleton)

## Abstract (Camera-Ready Draft)
Deep neuro-fuzzy modeling aims to combine the predictive capacity of deep representations with the transparency of rule-based inference. In practice, this objective is challenging because naively deepening ANFIS-like systems may either cause rule-base explosion or produce hidden states with weak semantic interpretability. In this work, we implement and compare three architecture lines under a single protocol: Stacked ANFIS, Hierarchical ANFIS, and Deep Fuzzy Feature Learning (DFFL). We evaluate these models on five real tabular datasets (`diabetes`, `linnerud_weight`, `breast_cancer`, `wine_binary`, `digits_binary`) with three random seeds (`19`, `23`, `29`) and additionally verify behavior on `california_housing` (`20640` samples). Evaluation follows a multi-objective framework with three axes: predictive quality (RMSE/F1), structural complexity (total and active rules), and interpretation stability (inter-seed Jaccard overlap of active rules). Hierarchical and stacked variants provide the strongest average quality rank in the core benchmark configuration, while DFFL shows competitive performance on selected datasets and strong rule-activation stability among deep alternatives. On `california_housing`, the best fuzzy result is achieved by Stacked ANFIS (`RMSE = 0.1159`). The main practical result is that architecture selection should be treated as a quality-complexity-stability trade-off rather than single-metric optimization.

## Related Work (Camera-Ready Draft)
### 2.1 Foundations: Fuzzy Inference and Neuro-Fuzzy Learning
Classical fuzzy set theory established the mathematical basis for uncertainty-aware rule systems [zadeh1965fuzzy]. Early fuzzy control and rule-based inference frameworks demonstrated that linguistic rules can be mapped to operational control logic [mamdani1975fuzzy]. The Takagi-Sugeno model introduced a compact consequent parameterization that enabled more expressive and numerically convenient fuzzy inference [takagi1985fuzzy].

The ANFIS formulation connected these fuzzy systems with gradient-based learning, making it possible to optimize membership and consequent parameters in a unified differentiable pipeline [jang1993anfis]. This line remains central in interpretable fuzzy learning, but canonical ANFIS is typically shallow and can become difficult to scale in high-dimensional settings.

### 2.2 Deep and Hierarchical Neuro-Fuzzy Architectures
Subsequent work explored deeper and hierarchical fuzzy designs to address ANFIS scalability limits and to model complex feature interactions [deep_hierarchical_fuzzy_survey_tbd, hierarchical_anfis_tbd, deep_anfis_end2end_tbd]. The common engineering challenge in these designs is balancing three competing requirements:

1. preserving interpretable intermediate representations,
2. controlling rule combinatorics,
3. retaining enough decision-layer expressivity.

Hierarchical local-block decompositions are frequently used to limit rule growth by replacing a global flat rule base with staged aggregation [hierarchical_anfis_tbd]. More recent approaches also frame hidden fuzzy blocks as feature builders rather than only local predictors, aligning deep fuzzy systems with representation-learning principles [deep_fuzzy_feature_learning_tbd].

### 2.3 Explainability and Stability as Selection Criteria
Interpretability in modern ML is often evaluated with post-hoc explanation methods such as LIME and SHAP [ribeiro2016lime, lundberg2017shap]. For rule-based fuzzy systems, however, explanation can be embedded directly in model structure. Even then, model selection by predictive score alone is often insufficient for deployment-sensitive scenarios.

Recent methodology emphasizes robust comparison protocols and stability-oriented criteria across random seeds, folds, or perturbations [demsar2006statistical]. In this context, rule-set consistency metrics (e.g., Jaccard overlap of active rules) provide an additional axis for selecting among similarly accurate models [model_stability_tbd].

### 2.4 Positioning of the Present Work
Compared with prior lines, the present study contributes an implementation-level and experimental comparison of three deep neuro-fuzzy architecture families under one reproducible benchmark protocol. The comparison is explicitly multi-objective (quality, complexity, stability), and the paper argues that this framing is necessary for practical architecture selection in interpretable deep fuzzy modeling.

## References Skeleton Notes
Use the BibTeX skeleton file:

- `docs/iiti26_references_skeleton_en.bib`

Recommended workflow:

1. keep the foundational references as-is,
2. replace all `*_tbd` entries with domain-specific, indexed (WoS/Scopus) papers,
3. update in-text citations accordingly,
4. ensure final bibliography style follows IITI template.
