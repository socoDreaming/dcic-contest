from __future__ import annotations

from dcic_contest.baseline.base import BaselineForecaster
from dcic_contest.baseline.naive import (
    Last7DaySameSlotForecaster,
    LastDaySameSlotForecaster,
    RecentDaysWeightedSameSlotForecaster,
)
from dcic_contest.baseline.seasonal import WeekdaySlotMeanForecaster


DEFAULT_BASELINE_NAMES = [
    "last_day_same_slot",
    "last_7day_same_slot",
    "weekday_slot_mean",
]

_FORECASTERS: dict[str, type[BaselineForecaster]] = {
    LastDaySameSlotForecaster.name: LastDaySameSlotForecaster,
    Last7DaySameSlotForecaster.name: Last7DaySameSlotForecaster,
    RecentDaysWeightedSameSlotForecaster.name: RecentDaysWeightedSameSlotForecaster,
    WeekdaySlotMeanForecaster.name: WeekdaySlotMeanForecaster,
}

try:
    from dcic_contest.baseline.gbdt import (
        LightGBMDirectForecaster,
        LightGBMRecursiveForecaster,
    )

    _FORECASTERS[LightGBMDirectForecaster.name] = LightGBMDirectForecaster
    _FORECASTERS[LightGBMRecursiveForecaster.name] = LightGBMRecursiveForecaster
except ModuleNotFoundError:
    # 允许在无 sklearn/lightgbm 环境下使用纯规则基线。
    pass


def available_baselines() -> list[str]:
    return sorted(_FORECASTERS.keys())


def build_forecasters(names: list[str] | None = None) -> list[BaselineForecaster]:
    selected = DEFAULT_BASELINE_NAMES if names is None else names
    unknown = [name for name in selected if name not in _FORECASTERS]
    if unknown:
        raise ValueError(
            f"Unknown baseline names: {unknown}. Available: {available_baselines()}"
        )
    return [_FORECASTERS[name]() for name in selected]
