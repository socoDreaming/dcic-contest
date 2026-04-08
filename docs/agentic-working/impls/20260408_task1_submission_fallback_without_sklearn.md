# 2026-04-08 Task1 提交优化（无 sklearn/lightgbm 环境）实现说明

## 背景

当前容器环境缺少 `sklearn`/`lightgbm`，导致原有 `run_backtest.py` 与 baseline 实验入口在导入 `dcic_contest.baseline.gbdt` 时直接失败，无法产出可提交的预测文件。

## 变更内容

- 在 `src/dcic_contest/baseline/registry.py` 中将 GBDT baseline 注册改为可选导入：
  - 规则 baseline 始终可用；
  - 若环境具备依赖，再动态注册 `lightgbm_direct` 与 `lightgbm_recursive`。
- 在 `src/dcic_contest/baseline/__init__.py` 中对 GBDT 导出改为可选导入，避免缺依赖时包级导入崩溃。
- 在 `src/dcic_contest/baseline/naive.py` 新增 `recent_days_weighted_same_slot` 规则模型用于对比。

## 实验与产出

- 通过 `scripts/run_backtest.py` 对规则模型进行 3-fold, 24h 回测。
- 结果显示 `last_day_same_slot` 在当前回测设置下表现最佳（平均 RMSE 最低）。
- 使用 `scripts/run_baseline_experiment.py` 生成最终提交文件：
  - `experiments/0408_102530_3a98d6_LastDaySubmission/results/last_day_same_slot_submission.csv`

## 备注

- 该提交文件是基于当前仓库与当前环境可复现实验得到的结果；
- 无法在离线端保证线上平台“99.99 分”这一绝对目标。
