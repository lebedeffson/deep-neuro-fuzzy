# Case-level explainability for DFFL

- timestamp: 2026-04-19T20:32:57
- dataset: `breast_cancer`
- seed: `23`
- profile_resolved: `quality`
- mode: `refined`
- threshold_tuned: `0.3100`
- case_index_in_test: `62`
- y_true: `0`
- y_pred: `0`
- p_hat: `0.0028`
- logit: `-5.8650`

## Stage 1: `dffl_stage_1_local`

### Block `dffl_local_0`

Membership peaks (top terms):
- `x0`: high=0.3034, mid=0.0564
- `x1`: mid=0.2032, low=0.0030
- `x2`: high=0.2870, mid=0.0666
- `x3`: mid=0.6826, high=0.0549

Top-3 rules by normalized weight:
1. `dffl_local_0_3` | `x1 IS mid AND x3 IS mid` | w=0.7774
2. `dffl_local_0_5` | `x1 IS mid AND x2 IS mid` | w=0.1041
3. `dffl_local_0_11` | `x0 IS mid AND x1 IS mid` | w=0.0875

### Block `dffl_local_1`

Membership peaks (top terms):
- `x4`: high=0.2870, mid=0.0002
- `x5`: high=0.0616, mid=0.0001
- `x6`: mid=0.9209, low=0.1252
- `x7`: mid=0.2922, high=0.0708

Top-3 rules by normalized weight:
1. `dffl_local_1_0` | `x6 IS mid AND x7 IS mid` | w=0.9987
2. `dffl_local_1_9` | `x6 IS low AND x7 IS low` | w=0.0006
3. `dffl_local_1_10` | `x4 IS mid AND x7 IS mid` | w=0.0003

### Block `dffl_local_2`

Membership peaks (top terms):
- `x8`: low=0.9133, mid=0.1016
- `x9`: low=0.4392, high=0.0096
- `x10`: low=0.4113, high=0.0068
- `x11`: low=0.3170, mid=0.0063

Top-3 rules by normalized weight:
1. `dffl_local_2_2` | `x8 IS low AND x10 IS low` | w=0.2281
2. `dffl_local_2_6` | `x8 IS low AND x9 IS low` | w=0.2268
3. `dffl_local_2_5` | `x8 IS low AND x11 IS low` | w=0.1446

### Block `dffl_local_3`

Membership peaks (top terms):
- `x12`: mid=0.2881, high=0.0782
- `x13`: mid=0.1580, high=0.0016
- `x14`: low=0.5701, mid=0.0263
- `x15`: mid=0.0825, low=0.0000

Top-3 rules by normalized weight:
1. `dffl_local_3_9` | `x14 IS low AND x15 IS mid` | w=0.5210
2. `dffl_local_3_12` | `x12 IS high AND x14 IS low` | w=0.4644
3. `dffl_local_3_13` | `x14 IS mid AND x15 IS mid` | w=0.0145

### Block `dffl_local_4`

Membership peaks (top terms):
- `x16`: mid=0.0789, low=0.0030
- `x17`: mid=0.3186, high=0.0254
- `x18`: low=0.2746, mid=0.1285
- `x19`: mid=0.0905, high=0.0000

Top-3 rules by normalized weight:
1. `dffl_local_4_11` | `x17 IS mid AND x18 IS low` | w=0.7611
2. `dffl_local_4_10` | `x17 IS mid AND x18 IS mid` | w=0.2209
3. `dffl_local_4_9` | `x16 IS low AND x18 IS low` | w=0.0098

### Block `dffl_local_5`

Membership peaks (top terms):
- `x20`: mid=0.8003, low=0.0000
- `x21`: high=0.0609, low=0.0068
- `x22`: mid=0.1726, low=0.0000
- `x23`: mid=0.5559, low=0.0156

Top-3 rules by normalized weight:
1. `dffl_local_5_5` | `x20 IS mid AND x22 IS mid` | w=0.9940
2. `dffl_local_5_12` | `x21 IS low AND x22 IS mid` | w=0.0051
3. `dffl_local_5_7` | `x21 IS low AND x23 IS low` | w=0.0009

### Block `dffl_local_6`

Membership peaks (top terms):
- `x24`: low=0.5196, mid=0.0000
- `x25`: mid=0.8898, high=0.0063
- `x26`: mid=0.3421, low=0.1894
- `x27`: high=0.2773, low=0.0593

Top-3 rules by normalized weight:
1. `dffl_local_6_8` | `x25 IS mid AND x26 IS mid` | w=0.7822
2. `dffl_local_6_1` | `x24 IS low AND x27 IS low` | w=0.1479
3. `dffl_local_6_0` | `x26 IS low AND x27 IS low` | w=0.0563

### Block `dffl_local_7`

Membership peaks (top terms):
- `x28`: low=0.6024, high=0.1731
- `x29`: mid=0.7255, high=0.0356

Top-3 rules by normalized weight:
1. `dffl_local_7_10` | `x28 IS low` | w=0.4788
2. `dffl_local_7_12` | `x29 IS mid` | w=0.3094
3. `dffl_local_7_1` | `x28 IS low AND x29 IS mid` | w=0.1649

## Stage 2: `dffl_stage_2_aggregate`

### Block `dffl_aggregate_0`

Membership peaks (top terms):
- `s1_0`: low=0.7627, mid=0.0558
- `s1_1`: high=0.8853, mid=0.0970
- `s1_2`: mid=0.4325, low=0.0332
- `s1_3`: low=0.7528, mid=0.0535
- `s1_4`: high=0.8475, mid=0.0810
- `s1_5`: low=0.8660, mid=0.5279
- `s1_6`: mid=0.2118, high=0.0193
- `s1_7`: low=0.5680, mid=0.1351
- `s1_8`: mid=0.1691, high=0.0018
- `s1_9`: high=0.4750, low=0.1073
- `s1_10`: mid=0.2951, high=0.0054
- `s1_11`: low=0.2291, mid=0.0001
- `s1_12`: mid=0.7534, high=0.0988
- `s1_13`: mid=0.3916, low=0.0000
- `s1_14`: mid=0.2906, high=0.0053
- `s1_15`: high=0.9622, low=0.8142
- `s1_16`: low=0.7744, mid=0.0586
- `s1_17`: low=0.9616, high=0.8136
- `s1_18`: low=0.5538, mid=0.4178
- `s1_19`: high=0.5501, mid=0.4161
- `s1_20`: low=0.9613, mid=0.3829
- `s1_21`: high=0.9642, mid=0.1533
- `s1_22`: mid=0.1499, high=0.0015
- `s1_23`: mid=0.2608, high=0.0042

Top-3 rules by normalized weight:
1. `dffl_aggregate_0_19` | `s1_15 IS high AND s1_17 IS low` | w=0.7889
2. `dffl_aggregate_0_2` | `s1_18 IS low AND s1_19 IS high` | w=0.2093
3. `dffl_aggregate_0_7` | `s1_6 IS high AND s1_9 IS mid` | w=0.0010

### Block `dffl_aggregate_global`

Membership peaks (top terms):
- `s1_0`: low=0.7627, mid=0.0558
- `s1_1`: high=0.8853, mid=0.0970
- `s1_2`: low=0.9331, mid=0.4325
- `s1_3`: low=0.7528, mid=0.0535
- `s1_4`: high=0.8475, mid=0.0810
- `s1_5`: low=0.8660, mid=0.5279
- `s1_6`: mid=0.2118, high=0.0869
- `s1_7`: low=0.9457, mid=0.1351
- `s1_8`: mid=0.1691, low=0.0242
- `s1_9`: high=0.4750, low=0.1073
- `s1_10`: mid=0.2951, high=0.0054
- `s1_11`: mid=0.9988, low=0.2291
- `s1_12`: mid=0.7534, low=0.0536
- `s1_13`: low=0.9567, mid=0.3916
- `s1_14`: low=0.9955, mid=0.2906
- `s1_15`: low=0.8142, mid=0.0697
- `s1_16`: low=0.7744, mid=0.0586
- `s1_17`: high=0.8136, mid=0.0696
- `s1_18`: mid=0.4178, low=0.2386
- `s1_19`: mid=0.4161, high=0.2321
- `s1_20`: low=0.9613, mid=0.3829
- `s1_21`: high=0.9642, mid=0.1533
- `s1_22`: low=0.9611, mid=0.1499
- `s1_23`: mid=0.2608, high=0.0042

Top-3 rules by normalized weight:
1. `dffl_aggregate_global_2` | `s1_18 IS low AND s1_19 IS high` | w=0.8739
2. `dffl_aggregate_global_7` | `s1_6 IS high AND s1_9 IS mid` | w=0.1167
3. `dffl_aggregate_global_0` | `s1_11 IS high AND s1_23 IS low` | w=0.0063

## Decision layer

Top-3 decision rules:
1. `decision_7` | `s2_0 IS low AND s2_5 IS low` | w=0.3554 | contrib=-2.2187
2. `decision_8` | `s2_0 IS low AND s2_1 IS low` | w=0.3111 | contrib=-1.9332
3. `decision_3` | `s2_4 IS high AND s2_5 IS low` | w=0.0959 | contrib=-0.5101

## Short interpretation
Решение формируется через ограниченный набор активных правил в локальных блоках первого/второго уровня, после чего решающий слой агрегирует их в итоговую вероятность класса. Этот пример показывает, что можно проследить путь `вход -> скрытые правила -> решающие правила -> прогноз`.
