# Индекс материалов репозитория

Репозиторий хранит код, воспроизводимые настройки, компактные артефакты и
рисунки. Полные тексты статьи, LaTeX-исходники и Word-версии намеренно не
хранятся в git.

Актуальные материалы проекта:

- `src/ruanfis/` — библиотечный код fuzzy/KAFN моделей.
- `examples/` — entrypoint-скрипты для бенчмарков.
- `docs/artifacts/` — компактные paper-facing сводки и traceability notes.
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

Старые `docs/iiti26_*.md`, `docs/iiti26_*.docx`, `docs/article_current/` и
другие полные черновики статьи удалены из git. Репозиторий остаётся кодовым и
artifact-facing.

## Что не коммитить

- `runs/`
- `artifacts/`
- `data/`
- логи запусков;
- скачанные датасеты;
- LaTeX-исходники статей;
- LaTeX build-файлы;
- Word-версии статей.
