"""Identity must survive bookkeeping changes while detecting physical changes."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import pandas as pd
import pytest
from HGNDRecoGNN.scripts.content_census import event_signatures, signature, summarise


def test_exact_multiset_identity():
    x = np.array([[1., -0.], [2., 3.]])
    h = signature(x, ('a', 'b'))
    assert signature(x[::-1], ('a', 'b')) == h
    assert signature([[1., 0.], [2., 3.]], ('a', 'b')) == h
    assert signature(np.r_[x, x[:1]], ('a', 'b')) != h
    changed = x.copy(); changed[1, 1] = np.nextafter(3., 4.)
    assert signature(changed, ('a', 'b')) != h
    assert signature(x, ('b', 'a')) != h
    with pytest.raises(ValueError):
        signature([[float('nan'), 0]], ('a', 'b'))


def test_event_bookkeeping_and_overlap():
    frame = pd.DataFrame({'Row': [1, 2, 1], 'value': [3., 5., 4.]})
    first = list(event_signatures(frame, ('value',)))
    second = list(event_signatures(frame.assign(Row=frame.Row + 100).iloc[::-1], ('value',)))
    assert [v[-1] for v in first] == [v[-1] for v in second]
    records = [('a', 'f1', 1, 2, first[0][-1]), ('a', 'f1', 2, 2, first[0][-1]),
               ('a', 'f2', 1, 2, first[0][-1]), ('b', 'f1', 1, 2, first[0][-1]),
               ('b', 'f1', 2, 1, first[1][-1])]
    result = summarise(records)
    assert result['unique_contents'] == 2
    assert result['events_in_repeated_groups'] == 4
    assert result['cross_directory_groups'] == result['cross_file_groups'] == 1
    assert result['within_file_repeated_groups'] == 1
