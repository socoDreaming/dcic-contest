from __future__ import annotations

import statistics

from dcic_contest.baseline.base import ForecastContext


class LastDaySameSlotForecaster:
    name = "last_day_same_slot"

    def predict(self, context: ForecastContext) -> list[float]:
        history = [float(row["V"]) for row in context.train_rows]
        if len(history) < context.horizon_steps:
            raise ValueError("history length is shorter than horizon_steps")
        return history[-context.horizon_steps :]


class Last7DaySameSlotForecaster:
    name = "last_7day_same_slot"

    def predict(self, context: ForecastContext) -> list[float]:
        history = [float(row["V"]) for row in context.train_rows]
        required = context.horizon_steps * 7
        if len(history) < required:
            raise ValueError("history length is shorter than seven days of slots")

        predictions: list[float] = []
        for slot in range(context.horizon_steps):
            slot_values = [
                history[-required + slot + day * context.horizon_steps]
                for day in range(7)
            ]
            predictions.append(statistics.fmean(slot_values))
        return predictions


class RecentDaysWeightedSameSlotForecaster:
    name = "recent_days_weighted_same_slot"

    def predict(self, context: ForecastContext) -> list[float]:
        history = [float(row["V"]) for row in context.train_rows]
        day_steps = context.horizon_steps
        required = day_steps * 14
        if len(history) < required:
            raise ValueError("history length is shorter than fourteen days of slots")

        # 最近 7 天赋予更高权重；更早 7 天作为稳定项。
        recent_weights = [1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7]
        older_weights = [0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.2]
        predictions: list[float] = []

        for slot in range(day_steps):
            weighted_sum = 0.0
            weighted_count = 0.0
            # day=0 表示昨天
            for day in range(7):
                idx_recent = -day_steps * (day + 1) + slot
                value_recent = history[idx_recent]
                weight_recent = recent_weights[day]
                weighted_sum += value_recent * weight_recent
                weighted_count += weight_recent

                idx_older = -day_steps * (day + 8) + slot
                value_older = history[idx_older]
                weight_older = older_weights[day]
                weighted_sum += value_older * weight_older
                weighted_count += weight_older

            predictions.append(weighted_sum / weighted_count)
        return predictions
