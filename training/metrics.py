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
        total = (sum(getattr(self.weights, k) * v for k, v in parts.items())
                 if aggregation == 'population' and self.normalizers is not None and self.batch_totals
                 else batch_mean)
        return {'total': total, 'components': parts, 'counts': dict(self.counts),
                'batch_mean': batch_mean}
