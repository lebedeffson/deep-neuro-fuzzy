# Case-level explainability for DFFL

- timestamp: 2026-04-22T02:21:19
- dataset: `breast_cancer`
- seed: `23`
- profile_resolved: `quality`
- mode: `refined`
- threshold_tuned: `0.7700`
- case_index_in_test: `32`
- y_true: `0`
- y_pred: `0`
- p_hat: `0.0001`
- logit: `-9.8283`

## Версия для пользователя (без формул)

### Что решила модель
- Для этого пациента модель выбрала `класс 0`.
- Уверенность высокая: `p_hat=0.0001` для альтернативного класса.

### Почему так (простыми словами)
- На первом уровне модель посмотрела на небольшие группы признаков (размер/периметр, текстура, concavity, worst-признаки и т.д.).
- Дальше эти локальные сигналы были объединены в 2 агрегирующих блока.
- В финальном слое сработал небольшой набор правил с отрицательным вкладом в альтернативный класс, поэтому итог ушел в `класс 0`.

### Какие группы признаков сыграли ключевую роль в этом кейсе
- `worst smoothness / worst compactness / worst concavity / worst concave points`
- `mean symmetry / mean fractal dimension / radius error / texture error`
- bridge-связи между группами:  
  `mean concave points <-> worst concave points`,  
  `mean perimeter <-> worst perimeter`,  
  `mean perimeter <-> worst radius`,  
  `mean radius <-> worst perimeter`.

### Как это читать человеку
- `w` у правила: насколько сильно правило участвовало в решении.
- `contrib`: направление и сила вклада правила в финальный логит.
- Чем меньше активных доминирующих правил, тем проще объяснить решение “по шагам”.

## Concept index mapping

### s1_* -> stage-1 concepts and original feature groups
- `s1_0` -> `s1_0_0` | block=`dffl_local_0` | features: mean radius [x0], mean texture [x1], mean perimeter [x2], mean area [x3]
- `s1_1` -> `s1_0_1` | block=`dffl_local_0` | features: mean radius [x0], mean texture [x1], mean perimeter [x2], mean area [x3]
- `s1_2` -> `s1_0_2` | block=`dffl_local_0` | features: mean radius [x0], mean texture [x1], mean perimeter [x2], mean area [x3]
- `s1_3` -> `s1_1_0` | block=`dffl_local_1` | features: mean smoothness [x4], mean compactness [x5], mean concavity [x6], mean concave points [x7]
- `s1_4` -> `s1_1_1` | block=`dffl_local_1` | features: mean smoothness [x4], mean compactness [x5], mean concavity [x6], mean concave points [x7]
- `s1_5` -> `s1_1_2` | block=`dffl_local_1` | features: mean smoothness [x4], mean compactness [x5], mean concavity [x6], mean concave points [x7]
- `s1_6` -> `s1_2_0` | block=`dffl_local_2` | features: mean symmetry [x8], mean fractal dimension [x9], radius error [x10], texture error [x11]
- `s1_7` -> `s1_2_1` | block=`dffl_local_2` | features: mean symmetry [x8], mean fractal dimension [x9], radius error [x10], texture error [x11]
- `s1_8` -> `s1_2_2` | block=`dffl_local_2` | features: mean symmetry [x8], mean fractal dimension [x9], radius error [x10], texture error [x11]
- `s1_9` -> `s1_3_0` | block=`dffl_local_3` | features: perimeter error [x12], area error [x13], smoothness error [x14], compactness error [x15]
- `s1_10` -> `s1_3_1` | block=`dffl_local_3` | features: perimeter error [x12], area error [x13], smoothness error [x14], compactness error [x15]
- `s1_11` -> `s1_3_2` | block=`dffl_local_3` | features: perimeter error [x12], area error [x13], smoothness error [x14], compactness error [x15]
- `s1_12` -> `s1_4_0` | block=`dffl_local_4` | features: concavity error [x16], concave points error [x17], symmetry error [x18], fractal dimension error [x19]
- `s1_13` -> `s1_4_1` | block=`dffl_local_4` | features: concavity error [x16], concave points error [x17], symmetry error [x18], fractal dimension error [x19]
- `s1_14` -> `s1_4_2` | block=`dffl_local_4` | features: concavity error [x16], concave points error [x17], symmetry error [x18], fractal dimension error [x19]
- `s1_15` -> `s1_5_0` | block=`dffl_local_5` | features: worst radius [x20], worst texture [x21], worst perimeter [x22], worst area [x23]
- `s1_16` -> `s1_5_1` | block=`dffl_local_5` | features: worst radius [x20], worst texture [x21], worst perimeter [x22], worst area [x23]
- `s1_17` -> `s1_5_2` | block=`dffl_local_5` | features: worst radius [x20], worst texture [x21], worst perimeter [x22], worst area [x23]
- `s1_18` -> `s1_6_0` | block=`dffl_local_6` | features: worst smoothness [x24], worst compactness [x25], worst concavity [x26], worst concave points [x27]
- `s1_19` -> `s1_6_1` | block=`dffl_local_6` | features: worst smoothness [x24], worst compactness [x25], worst concavity [x26], worst concave points [x27]
- `s1_20` -> `s1_6_2` | block=`dffl_local_6` | features: worst smoothness [x24], worst compactness [x25], worst concavity [x26], worst concave points [x27]
- `s1_21` -> `s1_7_0` | block=`dffl_local_7` | features: worst symmetry [x28], worst fractal dimension [x29]
- `s1_22` -> `s1_7_1` | block=`dffl_local_7` | features: worst symmetry [x28], worst fractal dimension [x29]
- `s1_23` -> `s1_7_2` | block=`dffl_local_7` | features: worst symmetry [x28], worst fractal dimension [x29]
- `s1_24` -> `s1b_0_0` | block=`dffl_bridge_0` | features: mean concave points [x7], worst concave points [x27]
- `s1_25` -> `s1b_1_0` | block=`dffl_bridge_1` | features: mean perimeter [x2], worst perimeter [x22]
- `s1_26` -> `s1b_2_0` | block=`dffl_bridge_2` | features: mean perimeter [x2], worst radius [x20]
- `s1_27` -> `s1b_3_0` | block=`dffl_bridge_3` | features: mean radius [x0], worst perimeter [x22]

### s2_* -> stage-2 concepts and upstream groups
- `s2_0` -> `s2_0` | block=`dffl_aggregate_0` | upstream groups: dffl_local_0: mean radius [x0], mean texture [x1], mean perimeter [x2], mean area [x3]; dffl_local_1: mean smoothness [x4], mean compactness [x5], mean concavity [x6], mean concave points [x7]; dffl_local_2: mean symmetry [x8], mean fractal dimension [x9], radius error [x10], texture error [x11]; dffl_local_3: perimeter error [x12], area error [x13], smoothness error [x14], compactness error [x15]; dffl_local_4: concavity error [x16], concave points error [x17], symmetry error [x18], fractal dimension error [x19]; dffl_local_5: worst radius [x20], worst texture [x21], worst perimeter [x22], worst area [x23]; dffl_local_6: worst smoothness [x24], worst compactness [x25], worst concavity [x26], worst concave points [x27]; dffl_local_7: worst symmetry [x28], worst fractal dimension [x29]; dffl_bridge_0: mean concave points [x7], worst concave points [x27]; dffl_bridge_1: mean perimeter [x2], worst perimeter [x22]; dffl_bridge_2: mean perimeter [x2], worst radius [x20]; dffl_bridge_3: mean radius [x0], worst perimeter [x22]
- `s2_1` -> `s2_1` | block=`dffl_aggregate_0` | upstream groups: dffl_local_0: mean radius [x0], mean texture [x1], mean perimeter [x2], mean area [x3]; dffl_local_1: mean smoothness [x4], mean compactness [x5], mean concavity [x6], mean concave points [x7]; dffl_local_2: mean symmetry [x8], mean fractal dimension [x9], radius error [x10], texture error [x11]; dffl_local_3: perimeter error [x12], area error [x13], smoothness error [x14], compactness error [x15]; dffl_local_4: concavity error [x16], concave points error [x17], symmetry error [x18], fractal dimension error [x19]; dffl_local_5: worst radius [x20], worst texture [x21], worst perimeter [x22], worst area [x23]; dffl_local_6: worst smoothness [x24], worst compactness [x25], worst concavity [x26], worst concave points [x27]; dffl_local_7: worst symmetry [x28], worst fractal dimension [x29]; dffl_bridge_0: mean concave points [x7], worst concave points [x27]; dffl_bridge_1: mean perimeter [x2], worst perimeter [x22]; dffl_bridge_2: mean perimeter [x2], worst radius [x20]; dffl_bridge_3: mean radius [x0], worst perimeter [x22]
- `s2_2` -> `s2_2` | block=`dffl_aggregate_0` | upstream groups: dffl_local_0: mean radius [x0], mean texture [x1], mean perimeter [x2], mean area [x3]; dffl_local_1: mean smoothness [x4], mean compactness [x5], mean concavity [x6], mean concave points [x7]; dffl_local_2: mean symmetry [x8], mean fractal dimension [x9], radius error [x10], texture error [x11]; dffl_local_3: perimeter error [x12], area error [x13], smoothness error [x14], compactness error [x15]; dffl_local_4: concavity error [x16], concave points error [x17], symmetry error [x18], fractal dimension error [x19]; dffl_local_5: worst radius [x20], worst texture [x21], worst perimeter [x22], worst area [x23]; dffl_local_6: worst smoothness [x24], worst compactness [x25], worst concavity [x26], worst concave points [x27]; dffl_local_7: worst symmetry [x28], worst fractal dimension [x29]; dffl_bridge_0: mean concave points [x7], worst concave points [x27]; dffl_bridge_1: mean perimeter [x2], worst perimeter [x22]; dffl_bridge_2: mean perimeter [x2], worst radius [x20]; dffl_bridge_3: mean radius [x0], worst perimeter [x22]
- `s2_3` -> `s2_3` | block=`dffl_aggregate_0` | upstream groups: dffl_local_0: mean radius [x0], mean texture [x1], mean perimeter [x2], mean area [x3]; dffl_local_1: mean smoothness [x4], mean compactness [x5], mean concavity [x6], mean concave points [x7]; dffl_local_2: mean symmetry [x8], mean fractal dimension [x9], radius error [x10], texture error [x11]; dffl_local_3: perimeter error [x12], area error [x13], smoothness error [x14], compactness error [x15]; dffl_local_4: concavity error [x16], concave points error [x17], symmetry error [x18], fractal dimension error [x19]; dffl_local_5: worst radius [x20], worst texture [x21], worst perimeter [x22], worst area [x23]; dffl_local_6: worst smoothness [x24], worst compactness [x25], worst concavity [x26], worst concave points [x27]; dffl_local_7: worst symmetry [x28], worst fractal dimension [x29]; dffl_bridge_0: mean concave points [x7], worst concave points [x27]; dffl_bridge_1: mean perimeter [x2], worst perimeter [x22]; dffl_bridge_2: mean perimeter [x2], worst radius [x20]; dffl_bridge_3: mean radius [x0], worst perimeter [x22]
- `s2_4` -> `s2_4` | block=`dffl_aggregate_0` | upstream groups: dffl_local_0: mean radius [x0], mean texture [x1], mean perimeter [x2], mean area [x3]; dffl_local_1: mean smoothness [x4], mean compactness [x5], mean concavity [x6], mean concave points [x7]; dffl_local_2: mean symmetry [x8], mean fractal dimension [x9], radius error [x10], texture error [x11]; dffl_local_3: perimeter error [x12], area error [x13], smoothness error [x14], compactness error [x15]; dffl_local_4: concavity error [x16], concave points error [x17], symmetry error [x18], fractal dimension error [x19]; dffl_local_5: worst radius [x20], worst texture [x21], worst perimeter [x22], worst area [x23]; dffl_local_6: worst smoothness [x24], worst compactness [x25], worst concavity [x26], worst concave points [x27]; dffl_local_7: worst symmetry [x28], worst fractal dimension [x29]; dffl_bridge_0: mean concave points [x7], worst concave points [x27]; dffl_bridge_1: mean perimeter [x2], worst perimeter [x22]; dffl_bridge_2: mean perimeter [x2], worst radius [x20]; dffl_bridge_3: mean radius [x0], worst perimeter [x22]
- `s2_5` -> `s2_5` | block=`dffl_aggregate_0` | upstream groups: dffl_local_0: mean radius [x0], mean texture [x1], mean perimeter [x2], mean area [x3]; dffl_local_1: mean smoothness [x4], mean compactness [x5], mean concavity [x6], mean concave points [x7]; dffl_local_2: mean symmetry [x8], mean fractal dimension [x9], radius error [x10], texture error [x11]; dffl_local_3: perimeter error [x12], area error [x13], smoothness error [x14], compactness error [x15]; dffl_local_4: concavity error [x16], concave points error [x17], symmetry error [x18], fractal dimension error [x19]; dffl_local_5: worst radius [x20], worst texture [x21], worst perimeter [x22], worst area [x23]; dffl_local_6: worst smoothness [x24], worst compactness [x25], worst concavity [x26], worst concave points [x27]; dffl_local_7: worst symmetry [x28], worst fractal dimension [x29]; dffl_bridge_0: mean concave points [x7], worst concave points [x27]; dffl_bridge_1: mean perimeter [x2], worst perimeter [x22]; dffl_bridge_2: mean perimeter [x2], worst radius [x20]; dffl_bridge_3: mean radius [x0], worst perimeter [x22]
- `s2_6` -> `s2_6` | block=`dffl_aggregate_0` | upstream groups: dffl_local_0: mean radius [x0], mean texture [x1], mean perimeter [x2], mean area [x3]; dffl_local_1: mean smoothness [x4], mean compactness [x5], mean concavity [x6], mean concave points [x7]; dffl_local_2: mean symmetry [x8], mean fractal dimension [x9], radius error [x10], texture error [x11]; dffl_local_3: perimeter error [x12], area error [x13], smoothness error [x14], compactness error [x15]; dffl_local_4: concavity error [x16], concave points error [x17], symmetry error [x18], fractal dimension error [x19]; dffl_local_5: worst radius [x20], worst texture [x21], worst perimeter [x22], worst area [x23]; dffl_local_6: worst smoothness [x24], worst compactness [x25], worst concavity [x26], worst concave points [x27]; dffl_local_7: worst symmetry [x28], worst fractal dimension [x29]; dffl_bridge_0: mean concave points [x7], worst concave points [x27]; dffl_bridge_1: mean perimeter [x2], worst perimeter [x22]; dffl_bridge_2: mean perimeter [x2], worst radius [x20]; dffl_bridge_3: mean radius [x0], worst perimeter [x22]
- `s2_7` -> `s2_7` | block=`dffl_aggregate_0` | upstream groups: dffl_local_0: mean radius [x0], mean texture [x1], mean perimeter [x2], mean area [x3]; dffl_local_1: mean smoothness [x4], mean compactness [x5], mean concavity [x6], mean concave points [x7]; dffl_local_2: mean symmetry [x8], mean fractal dimension [x9], radius error [x10], texture error [x11]; dffl_local_3: perimeter error [x12], area error [x13], smoothness error [x14], compactness error [x15]; dffl_local_4: concavity error [x16], concave points error [x17], symmetry error [x18], fractal dimension error [x19]; dffl_local_5: worst radius [x20], worst texture [x21], worst perimeter [x22], worst area [x23]; dffl_local_6: worst smoothness [x24], worst compactness [x25], worst concavity [x26], worst concave points [x27]; dffl_local_7: worst symmetry [x28], worst fractal dimension [x29]; dffl_bridge_0: mean concave points [x7], worst concave points [x27]; dffl_bridge_1: mean perimeter [x2], worst perimeter [x22]; dffl_bridge_2: mean perimeter [x2], worst radius [x20]; dffl_bridge_3: mean radius [x0], worst perimeter [x22]
- `s2_8` -> `s2_8` | block=`dffl_aggregate_0` | upstream groups: dffl_local_0: mean radius [x0], mean texture [x1], mean perimeter [x2], mean area [x3]; dffl_local_1: mean smoothness [x4], mean compactness [x5], mean concavity [x6], mean concave points [x7]; dffl_local_2: mean symmetry [x8], mean fractal dimension [x9], radius error [x10], texture error [x11]; dffl_local_3: perimeter error [x12], area error [x13], smoothness error [x14], compactness error [x15]; dffl_local_4: concavity error [x16], concave points error [x17], symmetry error [x18], fractal dimension error [x19]; dffl_local_5: worst radius [x20], worst texture [x21], worst perimeter [x22], worst area [x23]; dffl_local_6: worst smoothness [x24], worst compactness [x25], worst concavity [x26], worst concave points [x27]; dffl_local_7: worst symmetry [x28], worst fractal dimension [x29]; dffl_bridge_0: mean concave points [x7], worst concave points [x27]; dffl_bridge_1: mean perimeter [x2], worst perimeter [x22]; dffl_bridge_2: mean perimeter [x2], worst radius [x20]; dffl_bridge_3: mean radius [x0], worst perimeter [x22]
- `s2_9` -> `s2g_0` | block=`dffl_aggregate_global` | upstream groups: dffl_local_0: mean radius [x0], mean texture [x1], mean perimeter [x2], mean area [x3]; dffl_local_1: mean smoothness [x4], mean compactness [x5], mean concavity [x6], mean concave points [x7]; dffl_local_2: mean symmetry [x8], mean fractal dimension [x9], radius error [x10], texture error [x11]; dffl_local_3: perimeter error [x12], area error [x13], smoothness error [x14], compactness error [x15]; dffl_local_4: concavity error [x16], concave points error [x17], symmetry error [x18], fractal dimension error [x19]; dffl_local_5: worst radius [x20], worst texture [x21], worst perimeter [x22], worst area [x23]; dffl_local_6: worst smoothness [x24], worst compactness [x25], worst concavity [x26], worst concave points [x27]; dffl_local_7: worst symmetry [x28], worst fractal dimension [x29]; dffl_bridge_0: mean concave points [x7], worst concave points [x27]; dffl_bridge_1: mean perimeter [x2], worst perimeter [x22]; dffl_bridge_2: mean perimeter [x2], worst radius [x20]; dffl_bridge_3: mean radius [x0], worst perimeter [x22]
- `s2_10` -> `s2g_1` | block=`dffl_aggregate_global` | upstream groups: dffl_local_0: mean radius [x0], mean texture [x1], mean perimeter [x2], mean area [x3]; dffl_local_1: mean smoothness [x4], mean compactness [x5], mean concavity [x6], mean concave points [x7]; dffl_local_2: mean symmetry [x8], mean fractal dimension [x9], radius error [x10], texture error [x11]; dffl_local_3: perimeter error [x12], area error [x13], smoothness error [x14], compactness error [x15]; dffl_local_4: concavity error [x16], concave points error [x17], symmetry error [x18], fractal dimension error [x19]; dffl_local_5: worst radius [x20], worst texture [x21], worst perimeter [x22], worst area [x23]; dffl_local_6: worst smoothness [x24], worst compactness [x25], worst concavity [x26], worst concave points [x27]; dffl_local_7: worst symmetry [x28], worst fractal dimension [x29]; dffl_bridge_0: mean concave points [x7], worst concave points [x27]; dffl_bridge_1: mean perimeter [x2], worst perimeter [x22]; dffl_bridge_2: mean perimeter [x2], worst radius [x20]; dffl_bridge_3: mean radius [x0], worst perimeter [x22]

## Stage 1: `dffl_stage_1_local`

### Block `dffl_local_0`

Membership peaks (top terms):
- `mean radius [x0]`: low=0.2076, high=0.1251
- `mean texture [x1]`: low=0.0190, high=0.0001
- `mean perimeter [x2]`: mid=0.6743, low=0.1468
- `mean area [x3]`: mid=0.1566, high=0.0158

Top-3 rules by normalized weight:
1. `dffl_local_0_0` | `mean radius [x0] IS low AND mean perimeter [x2] IS low` | w=0.6909
2. `dffl_local_0_13` | `mean texture [x1] IS low AND mean perimeter [x2] IS mid` | w=0.1895
3. `dffl_local_0_4` | `mean radius [x0] IS low AND mean texture [x1] IS low` | w=0.0691

### Block `dffl_local_1`

Membership peaks (top terms):
- `mean smoothness [x4]`: high=0.4049, mid=0.0033
- `mean compactness [x5]`: high=0.1816, mid=0.0029
- `mean concavity [x6]`: mid=0.3784, high=0.0274
- `mean concave points [x7]`: mid=0.3018, high=0.0704

Top-3 rules by normalized weight:
1. `dffl_local_1_0` | `mean concavity [x6] IS mid AND mean concave points [x7] IS mid` | w=0.9721
2. `dffl_local_1_10` | `mean smoothness [x4] IS mid AND mean concave points [x7] IS mid` | w=0.0124
3. `dffl_local_1_13` | `mean compactness [x5] IS mid AND mean concave points [x7] IS mid` | w=0.0077

### Block `dffl_local_2`

Membership peaks (top terms):
- `mean symmetry [x8]`: high=0.9988, low=0.0001
- `mean fractal dimension [x9]`: mid=0.9166, high=0.0460
- `radius error [x10]`: mid=0.1432, low=0.1251
- `texture error [x11]`: mid=0.0326, high=0.0003

Top-3 rules by normalized weight:
1. `dffl_local_2_11` | `mean fractal dimension [x9] IS mid AND radius error [x10] IS low` | w=0.5471
2. `dffl_local_2_0` | `mean symmetry [x8] IS high AND radius error [x10] IS mid` | w=0.4409
3. `dffl_local_2_8` | `radius error [x10] IS low AND texture error [x11] IS mid` | w=0.0120

### Block `dffl_local_3`

Membership peaks (top terms):
- `perimeter error [x12]`: low=0.3916, mid=0.0551
- `area error [x13]`: low=0.0536, mid=0.0451
- `smoothness error [x14]`: mid=0.0341, high=0.0004
- `compactness error [x15]`: mid=0.9475, low=0.9049

Top-3 rules by normalized weight:
1. `dffl_local_3_2` | `perimeter error [x12] IS low AND compactness error [x15] IS low` | w=0.4918
2. `dffl_local_3_8` | `perimeter error [x12] IS low AND compactness error [x15] IS mid` | w=0.3129
3. `dffl_local_3_4` | `area error [x13] IS low AND compactness error [x15] IS low` | w=0.0652

### Block `dffl_local_4`

Membership peaks (top terms):
- `concavity error [x16]`: low=0.6791, mid=0.0556
- `concave points error [x17]`: high=0.0023, mid=0.0001
- `symmetry error [x18]`: high=0.3962, low=0.0385
- `fractal dimension error [x19]`: high=0.6041, mid=0.0839

Top-3 rules by normalized weight:
1. `dffl_local_4_12` | `concavity error [x16] IS low AND symmetry error [x18] IS high` | w=0.8211
2. `dffl_local_4_9` | `concavity error [x16] IS low AND symmetry error [x18] IS low` | w=0.1665
3. `dffl_local_4_6` | `concavity error [x16] IS low AND symmetry error [x18] IS mid` | w=0.0119

### Block `dffl_local_5`

Membership peaks (top terms):
- `worst radius [x20]`: mid=0.9660, high=0.5555
- `worst texture [x21]`: high=0.0913, low=0.0047
- `worst perimeter [x22]`: high=0.2787, low=0.0006
- `worst area [x23]`: mid=0.4246, high=0.2061

Top-3 rules by normalized weight:
1. `dffl_local_5_8` | `worst radius [x20] IS high AND worst perimeter [x22] IS high` | w=0.7540
2. `dffl_local_5_11` | `worst perimeter [x22] IS high AND worst area [x23] IS high` | w=0.2442
3. `dffl_local_5_5` | `worst radius [x20] IS mid AND worst perimeter [x22] IS mid` | w=0.0016

### Block `dffl_local_6`

Membership peaks (top terms):
- `worst smoothness [x24]`: high=0.3615, low=0.0688
- `worst compactness [x25]`: high=0.5759, mid=0.3458
- `worst concavity [x26]`: high=0.3780, mid=0.0040
- `worst concave points [x27]`: mid=0.8472, high=0.6755

Top-3 rules by normalized weight:
1. `dffl_local_6_6` | `worst smoothness [x24] IS high AND worst concave points [x27] IS mid` | w=0.9771
2. `dffl_local_6_13` | `worst smoothness [x24] IS mid AND worst concave points [x27] IS mid` | w=0.0154
3. `dffl_local_6_8` | `worst compactness [x25] IS mid AND worst concavity [x26] IS mid` | w=0.0052

### Block `dffl_local_7`

Membership peaks (top terms):
- `worst symmetry [x28]`: high=0.4305, mid=0.3274
- `worst fractal dimension [x29]`: mid=0.6876, high=0.0005

Top-3 rules by normalized weight:
1. `dffl_local_7_12` | `worst fractal dimension [x29] IS mid` | w=0.4616
2. `dffl_local_7_11` | `worst symmetry [x28] IS mid` | w=0.1883
3. `dffl_local_7_2` | `worst symmetry [x28] IS mid AND worst fractal dimension [x29] IS mid` | w=0.1755

### Block `dffl_bridge_0`

Membership peaks (top terms):
- `mean concave points [x7]`: mid=0.5200, high=0.1876
- `worst concave points [x27]`: high=0.3621, mid=0.1459

Top-3 rules by normalized weight:
1. `dffl_bridge_0_0` | `mean concave points [x7] IS mid AND worst concave points [x27] IS high` | w=0.5624
2. `dffl_bridge_0_2` | `mean concave points [x7] IS mid AND worst concave points [x27] IS mid` | w=0.2757
3. `dffl_bridge_0_5` | `mean concave points [x7] IS high AND worst concave points [x27] IS high` | w=0.0877

### Block `dffl_bridge_1`

Membership peaks (top terms):
- `mean perimeter [x2]`: high=0.8517, low=0.6559
- `worst perimeter [x22]`: mid=0.1386, high=0.0140

Top-3 rules by normalized weight:
1. `dffl_bridge_1_5` | `mean perimeter [x2] IS high AND worst perimeter [x22] IS mid` | w=0.9394
2. `dffl_bridge_1_2` | `mean perimeter [x2] IS high AND worst perimeter [x22] IS high` | w=0.0558
3. `dffl_bridge_1_0` | `mean perimeter [x2] IS low AND worst perimeter [x22] IS low` | w=0.0047

### Block `dffl_bridge_2`

Membership peaks (top terms):
- `mean perimeter [x2]`: high=0.5485, low=0.4471
- `worst radius [x20]`: mid=0.2988, high=0.1230

Top-3 rules by normalized weight:
1. `dffl_bridge_2_4` | `mean perimeter [x2] IS high AND worst radius [x20] IS mid` | w=0.7421
2. `dffl_bridge_2_2` | `mean perimeter [x2] IS high AND worst radius [x20] IS high` | w=0.2546
3. `dffl_bridge_2_1` | `mean perimeter [x2] IS low AND worst radius [x20] IS low` | w=0.0032

### Block `dffl_bridge_3`

Membership peaks (top terms):
- `mean radius [x0]`: high=0.8531, low=0.8473
- `worst perimeter [x22]`: mid=0.2921, high=0.0563

Top-3 rules by normalized weight:
1. `dffl_bridge_3_5` | `mean radius [x0] IS high AND worst perimeter [x22] IS mid` | w=0.8725
2. `dffl_bridge_3_2` | `mean radius [x0] IS high AND worst perimeter [x22] IS high` | w=0.1257
3. `dffl_bridge_3_0` | `mean radius [x0] IS low AND worst perimeter [x22] IS low` | w=0.0018

## Stage 2: `dffl_stage_2_aggregate`

### Block `dffl_aggregate_0`

Membership peaks (top terms):
- `s1_0 (s1_0_0 <- dffl_local_0)`: mid=0.3181, low=0.0064
- `s1_1 (s1_0_1 <- dffl_local_0)`: low=0.9231, mid=0.1182
- `s1_2 (s1_0_2 <- dffl_local_0)`: mid=0.3168, high=0.0063
- `s1_3 (s1_1_0 <- dffl_local_1)`: low=0.7604, mid=0.0552
- `s1_4 (s1_1_1 <- dffl_local_1)`: high=0.7913, mid=0.0631
- `s1_5 (s1_1_2 <- dffl_local_1)`: low=0.7834, mid=0.0610
- `s1_6 (s1_2_0 <- dffl_local_2)`: mid=0.0233, high=0.0057
- `s1_7 (s1_2_1 <- dffl_local_2)`: low=0.1096, mid=0.0263
- `s1_8 (s1_2_2 <- dffl_local_2)`: low=0.3380, mid=0.2000
- `s1_9 (s1_3_0 <- dffl_local_3)`: high=0.9825, mid=0.1407
- `s1_10 (s1_3_1 <- dffl_local_3)`: high=0.0011, mid=0.0000
- `s1_11 (s1_3_2 <- dffl_local_3)`: low=0.9740, mid=0.1657
- `s1_12 (s1_4_0 <- dffl_local_4)`: low=0.9838, mid=0.3316
- `s1_13 (s1_4_1 <- dffl_local_4)`: low=0.8275, mid=0.3373
- `s1_14 (s1_4_2 <- dffl_local_4)`: high=0.7274, mid=0.3314
- `s1_15 (s1_5_0 <- dffl_local_5)`: low=0.7423, mid=0.0511
- `s1_16 (s1_5_1 <- dffl_local_5)`: high=0.7531, mid=0.0535
- `s1_17 (s1_5_2 <- dffl_local_5)`: low=0.8559, mid=0.5408
- `s1_18 (s1_6_0 <- dffl_local_6)`: low=0.7675, high=0.1808
- `s1_19 (s1_6_1 <- dffl_local_6)`: mid=0.3992, high=0.0104
- `s1_20 (s1_6_2 <- dffl_local_6)`: high=0.8926, mid=0.1005
- `s1_21 (s1_7_0 <- dffl_local_7)`: mid=0.7798, high=0.1906
- `s1_22 (s1_7_1 <- dffl_local_7)`: low=0.9288, mid=0.1220
- `s1_23 (s1_7_2 <- dffl_local_7)`: mid=0.9997, low=0.2590
- `s1_24 (s1b_0_0 <- dffl_bridge_0)`: high=0.8153, mid=0.0701
- `s1_25 (s1b_1_0 <- dffl_bridge_1)`: high=0.7937, mid=0.0638
- `s1_26 (s1b_2_0 <- dffl_bridge_2)`: high=0.8522, mid=0.5454
- `s1_27 (s1b_3_0 <- dffl_bridge_3)`: high=0.9032, mid=0.5965

Top-3 rules by normalized weight:
1. `dffl_aggregate_0_11` | `s1_8 (s1_2_2 <- dffl_local_2) IS low AND s1_27 (s1b_3_0 <- dffl_bridge_3) IS high` | w=0.5554
2. `dffl_aggregate_0_10` | `s1_13 (s1_4_1 <- dffl_local_4) IS low AND s1_21 (s1_7_0 <- dffl_local_7) IS high` | w=0.2769
3. `dffl_aggregate_0_17` | `s1_8 (s1_2_2 <- dffl_local_2) IS low AND s1_21 (s1_7_0 <- dffl_local_7) IS high` | w=0.1221

### Block `dffl_aggregate_global`

Membership peaks (top terms):
- `s1_0 (s1_0_0 <- dffl_local_0)`: mid=0.3181, high=0.1032
- `s1_1 (s1_0_1 <- dffl_local_0)`: low=0.9231, mid=0.1182
- `s1_2 (s1_0_2 <- dffl_local_0)`: mid=0.3168, low=0.0938
- `s1_3 (s1_1_0 <- dffl_local_1)`: low=0.7604, mid=0.0552
- `s1_4 (s1_1_1 <- dffl_local_1)`: high=0.7913, mid=0.0631
- `s1_5 (s1_1_2 <- dffl_local_1)`: low=0.7834, mid=0.0610
- `s1_6 (s1_2_0 <- dffl_local_2)`: mid=0.9874, high=0.0377
- `s1_7 (s1_2_1 <- dffl_local_2)`: mid=0.9758, low=0.3519
- `s1_8 (s1_2_2 <- dffl_local_2)`: low=0.9919, mid=0.2000
- `s1_9 (s1_3_0 <- dffl_local_3)`: high=0.9520, mid=0.1407
- `s1_10 (s1_3_1 <- dffl_local_3)`: low=0.8692, high=0.0011
- `s1_11 (s1_3_2 <- dffl_local_3)`: low=0.9740, mid=0.1657
- `s1_12 (s1_4_0 <- dffl_local_4)`: low=0.9838, mid=0.3316
- `s1_13 (s1_4_1 <- dffl_local_4)`: low=0.9817, mid=0.3373
- `s1_14 (s1_4_2 <- dffl_local_4)`: high=0.9839, mid=0.3314
- `s1_15 (s1_5_0 <- dffl_local_5)`: low=0.7423, mid=0.0511
- `s1_16 (s1_5_1 <- dffl_local_5)`: high=0.7531, mid=0.0535
- `s1_17 (s1_5_2 <- dffl_local_5)`: low=0.8559, mid=0.5408
- `s1_18 (s1_6_0 <- dffl_local_6)`: low=0.7675, mid=0.7238
- `s1_19 (s1_6_1 <- dffl_local_6)`: low=0.4809, mid=0.3992
- `s1_20 (s1_6_2 <- dffl_local_6)`: high=0.8926, mid=0.1005
- `s1_21 (s1_7_0 <- dffl_local_7)`: mid=0.7798, low=0.0600
- `s1_22 (s1_7_1 <- dffl_local_7)`: low=0.9288, mid=0.1220
- `s1_23 (s1_7_2 <- dffl_local_7)`: mid=0.9997, low=0.2590
- `s1_24 (s1b_0_0 <- dffl_bridge_0)`: high=0.8153, mid=0.0701
- `s1_25 (s1b_1_0 <- dffl_bridge_1)`: high=0.7937, mid=0.0638
- `s1_26 (s1b_2_0 <- dffl_bridge_2)`: high=0.8522, mid=0.5454
- `s1_27 (s1b_3_0 <- dffl_bridge_3)`: mid=0.5965, high=0.3210

Top-3 rules by normalized weight:
1. `dffl_aggregate_global_0` | `s1_18 (s1_6_0 <- dffl_local_6) IS mid AND s1_19 (s1_6_1 <- dffl_local_6) IS low` | w=0.9457
2. `dffl_aggregate_global_4` | `s1_0 (s1_0_0 <- dffl_local_0) IS high AND s1_2 (s1_0_2 <- dffl_local_0) IS low` | w=0.0534
3. `dffl_aggregate_global_3` | `s1_14 (s1_4_2 <- dffl_local_4) IS low AND s1_27 (s1b_3_0 <- dffl_bridge_3) IS high` | w=0.0007

## Decision layer

Top-3 decision rules:
1. `decision_10` | `s2_2 (s2_2 <- dffl_aggregate_0) IS low AND s2_5 (s2_5 <- dffl_aggregate_0) IS low` | w=0.2109 | contrib=-1.6430
2. `decision_11` | `s2_2 (s2_2 <- dffl_aggregate_0) IS low AND s2_10 (s2g_1 <- dffl_aggregate_global) IS low` | w=0.2029 | contrib=-3.4563
3. `decision_8` | `s2_2 (s2_2 <- dffl_aggregate_0) IS low AND s2_6 (s2_6 <- dffl_aggregate_0) IS low` | w=0.1057 | contrib=-0.3912

## Short interpretation
Решение формируется через ограниченный набор активных правил в локальных блоках первого/второго уровня, после чего решающий слой агрегирует их в итоговую вероятность класса. Этот пример показывает, что можно проследить путь `вход -> скрытые правила -> решающие правила -> прогноз`.


## Feature index mapping (breast_cancer)

This mapping is used to decode `x_i` terms in Stage-1 rules into human-readable feature names.

- `x0` -> `mean radius`
- `x1` -> `mean texture`
- `x2` -> `mean perimeter`
- `x3` -> `mean area`
- `x4` -> `mean smoothness`
- `x5` -> `mean compactness`
- `x6` -> `mean concavity`
- `x7` -> `mean concave points`
- `x8` -> `mean symmetry`
- `x9` -> `mean fractal dimension`
- `x10` -> `radius error`
- `x11` -> `texture error`
- `x12` -> `perimeter error`
- `x13` -> `area error`
- `x14` -> `smoothness error`
- `x15` -> `compactness error`
- `x16` -> `concavity error`
- `x17` -> `concave points error`
- `x18` -> `symmetry error`
- `x19` -> `fractal dimension error`
- `x20` -> `worst radius`
- `x21` -> `worst texture`
- `x22` -> `worst perimeter`
- `x23` -> `worst area`
- `x24` -> `worst smoothness`
- `x25` -> `worst compactness`
- `x26` -> `worst concavity`
- `x27` -> `worst concave points`
- `x28` -> `worst symmetry`
- `x29` -> `worst fractal dimension`

## Мини-глоссарий (для читателя статьи)

- `локальный блок`: маленький набор правил на ограниченной группе признаков.
- `bridge-блок`: блок, который специально связывает разные группы признаков.
- `агрегирующий блок`: собирает сигналы из локальных и bridge-блоков.
- `top-k rules`: несколько самых влиятельных правил для данного объекта.
- `traceable transparency`: возможность пройти путь решения по блокам и правилам, а не только увидеть финальный класс.
