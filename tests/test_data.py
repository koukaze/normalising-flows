'''Check deterministic generation, split independence and external CSV alignment.'''

import numpy as np
import pytest

from moonflow.data import load_moons_splits, make_moons_splits, save_moons_splits


def test_generated_splits_are_reproducible_and_independent():
    '''Same settings reproduce each split without reusing validation noise for test.'''
    first = make_moons_splits(n_train=40, n_val=20, n_test=20, seed=31)
    second = make_moons_splits(n_train=40, n_val=20, n_test=20, seed=31)
    for name, split in first.items():
        np.testing.assert_array_equal(split.points, second[name].points)
        np.testing.assert_array_equal(split.labels, second[name].labels)
        assert (split.labels == 0).sum() == len(split.points) // 2
    assert not np.array_equal(first['val'].points, first['test'].points)
    changed_train = make_moons_splits(n_train=100, n_val=20, n_test=20, seed=31)
    np.testing.assert_array_equal(first['val'].points, changed_train['val'].points)


def test_generated_data_csv_roundtrip(tmp_path):
    '''Coordinates and labels must retain their order across serialization.'''
    expected = make_moons_splits(n_train=10, n_val=8, n_test=8)
    save_moons_splits(expected, tmp_path)
    loaded = load_moons_splits(tmp_path)
    for name in expected:
        np.testing.assert_array_equal(loaded[name].points, expected[name].points)
        np.testing.assert_array_equal(loaded[name].labels, expected[name].labels)


def test_csv_labels_are_optional_and_nonfinite_coordinates_fail(tmp_path):
    '''Labels support plots only, while bad coordinates cannot enter training.'''
    for name in ('train', 'val', 'test'):
        (tmp_path / f'moons_{name}.csv').write_text('x1,x2\n0,1\n2,3\n')
    loaded = load_moons_splits(tmp_path)
    assert loaded['train'].labels is None
    (tmp_path / 'moons_train.csv').write_text('x1,x2\n0,nan\n')
    with pytest.raises(ValueError, match='non-finite'):
        load_moons_splits(tmp_path)
