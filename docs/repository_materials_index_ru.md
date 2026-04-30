# Индекс материалов репозитория

Актуальный главный материал статьи сейчас находится в корне репозитория:

- `main(2).tex` — основной LaTeX-текст статьи про Routed KAFN.
- `docs/iiti26_references_skeleton_en.bib` — библиография для сборки.
- `docs/figures/fig5_kafn_architecture.png` — схема Routed KAFN.
- `docs/figures/fig6_kafn_explanation_flow.png` — схема объяснения Routed KAFN.
- `docs/figures/generate_kafn_figures.py` — генератор KAFN-рисунков.

## Текущий фокус статьи

Текущая версия статьи рассматривает Routed Kolmogorov--Arnold Fuzzy Network
(Routed KAFN) как основной метод и сравнивает его с:

- shallow fuzzy;
- stacked ANFIS;
- hierarchical ANFIS;
- DFFL;
- классическими sklearn baseline-моделями.

Главная формулировка результата: KAFN не заявляется как абсолютный лидер по
точности, но даёт сильную точку качества и стабильности интерпретируемой
структуры правил на `covtype_binary_20000`.

## Актуальные источники чисел KAFN

Сырые прогоны лежат локально в `runs/` и не коммитятся:

- `runs/kafn_paper_all_3s_2026-04-30`
- `runs/cov20k_deep_kanfis_teacher_grouped_q12_fan20_r936_3s_2026-04-30`
- `runs/cov20k_kaanifs_tg_q12_f20_r936_binaryterms_3s_2026-04-30`
- `runs/cov20k_kaanifs_proj_p8w4_fixed_3s_2026-04-30`
- `runs/cov20k_fuzzy4_fixcheck_3s_2026-04-29`
- `runs/cov20k_kaanifs_binary_q12_vs_sklearn_3s_2026-04-30`

## Текущие компактные артефакты

- `docs/artifacts/README.md` — политика и список paper-facing артефактов.
- `docs/artifacts/article_revision_results_2026-04-27.md` — предыдущий цикл ревизии.
- `docs/artifacts/article_support_tables_2026-04-27.md` — вспомогательные таблицы.
- `docs/artifacts/finalization_state_2026-04-27.md` — состояние перед переходом к KAFN.

## Устаревшие материалы

Файлы `docs/iiti26_*.md` и `docs/iiti26_*.docx` сохранены как исторические
рабочие версии. Они не являются текущим canonical paper source.

Старый пакет `docs/submission/iiti26_submission_pack_2026-04-16/` удалён из git,
потому что он относится к прежней версии статьи и больше не соответствует
KAFN-рукописи.

## Что не коммитить

- `runs/`
- `artifacts/`
- `data/`
- логи запусков;
- скачанные датасеты;
- LaTeX build-файлы.
