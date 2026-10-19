"""Epoch loss summaries with a separate denominator for each prediction head."""
from collections import defaultdict
import math


class LossAccumulator:
    def __init__(self, loss_fn, weights):
        self.normalizers = getattr(loss_fn, 'normalizers', None)
        self.weights = weights
        self.sums, self.counts = defaultdict(float), defaultdict(int)
        self.batch_totals = []

    def update(self, losses, graph):
        self.batch_totals.append(float(losses['total'].detach()))
        if self.normalizers is not None:
            for name, count in self.normalizers(graph).items():
                count = int(count)
                if count:
                    value = float(losses[name].detach())
                    if not math.isfinite(value):
                        raise ValueError(f'Nonfinite {name} loss')
                    self.sums[name] += count * value
                    self.counts[name] += count

    def result(self, aggregation='population'):
        if aggregation not in ('population', 'batch_mean'):
            raise ValueError(f'Unknown loss aggregation: {aggregation}')
        parts = {k: self.sums[k] / n for k, n in self.counts.items() if n}
        batch_mean = sum(self.batch_totals) / len(self.batch_totals) if self.batch_totals else float('nan')
        # A loss_fn with no `normalizers` cannot be aggregated per head, so the
        # population request degrades to the batch mean. That is a different
        # statistic, and silently substituting it would make a run's selector
        # disagree with its own configuration -- so the key below records which
        # one was actually used, and fit() reports a downgrade.
        population_ok = (self.normalizers is not None and bool(self.batch_totals)
                         and bool(parts))
        used = 'population' if (aggregation == 'population' and population_ok) \
            else 'batch_mean'
        total = (sum(getattr(self.weights, k) * v for k, v in parts.items())
                 if used == 'population' else batch_mean)
        return {'total': total, 'components': parts, 'counts': dict(self.counts),
                'batch_mean': batch_mean, 'aggregation': used}
