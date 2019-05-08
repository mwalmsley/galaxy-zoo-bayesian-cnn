import pytest

import numpy as np

@pytest.fixture()
def typical_vote_frac():
    return 0.5


@pytest.fixture()
def n_subjects():
    return 1028  # more subjects decreases coverage variation between each confidence level


@pytest.fixture()
def n_samples():
    return 40  # more samples decreases systematic offset on coverage vs confidence (from pymc3)



