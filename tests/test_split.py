import pytest

from app_opcoes_binarias.research.split import chronological_split


def test_chronological_split_preserves_order():
    split = chronological_split([1, 2, 3, 4, 5], train_fraction=0.6)
    assert split.train == [1, 2, 3]
    assert split.test == [4, 5]


def test_chronological_split_rejects_invalid_fraction():
    with pytest.raises(ValueError):
        chronological_split([1, 2], train_fraction=1.0)


def test_chronological_split_rejects_too_few_items():
    with pytest.raises(ValueError):
        chronological_split([1])
