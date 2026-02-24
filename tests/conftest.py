"""Shared pytest fixtures for qc-sampler tests.

Provides reusable configuration objects, mock entropy sources, and
sample logit arrays that are used across multiple test modules.
"""

from __future__ import annotations

import numpy as np
import pytest

from qc_sampler.config import QCSamplingConfig
from qc_sampler.entropy_source import MockUniformSource, OsUrandomSource


@pytest.fixture
def default_config() -> QCSamplingConfig:
    """Return a QCSamplingConfig with all default values.

    This matches the dataclass defaults, which in turn match the
    spec's env var defaults.
    """
    return QCSamplingConfig()


@pytest.fixture
def edt_config() -> QCSamplingConfig:
    """Return a QCSamplingConfig with EDT temperature strategy enabled."""
    return QCSamplingConfig(
        temperature_strategy="edt",
        edt_base_temp=0.8,
        edt_exponent=0.5,
        edt_min_temp=0.1,
        edt_max_temp=2.0,
    )


@pytest.fixture
def diagnostic_config() -> QCSamplingConfig:
    """Return a config with diagnostic mode and full logging enabled."""
    return QCSamplingConfig(
        log_level="full",
        diagnostic_mode=True,
    )


@pytest.fixture
def silent_config() -> QCSamplingConfig:
    """Return a config with no logging for noise-free tests."""
    return QCSamplingConfig(log_level="none")


@pytest.fixture
def mock_entropy_source() -> MockUniformSource:
    """Return a MockUniformSource at the null-hypothesis mean (127.5).

    Uses a fixed seed for reproducibility across test runs.
    """
    return MockUniformSource(mean=127.5, seed=42)


@pytest.fixture
def biased_entropy_source() -> MockUniformSource:
    """Return a MockUniformSource with mean biased above null (128.0).

    Simulates a consciousness-biased QRNG. Fixed seed for reproducibility.
    """
    return MockUniformSource(mean=128.0, seed=42)


@pytest.fixture
def os_entropy_source() -> OsUrandomSource:
    """Return an OsUrandomSource for pseudo-random testing."""
    return OsUrandomSource()


@pytest.fixture
def sample_logits_uniform() -> np.ndarray:
    """Return logits that produce a roughly uniform probability distribution.

    All logits are equal (zero), so softmax gives equal probability to
    every token. Vocab size = 100.
    """
    return np.zeros(100, dtype=np.float64)


@pytest.fixture
def sample_logits_peaked() -> np.ndarray:
    """Return logits with one dominant token (index 0).

    Token 0 has logit 10.0; all others have logit 0.0.
    After softmax, token 0 has ~99.995% probability.
    Vocab size = 100.
    """
    logits = np.zeros(100, dtype=np.float64)
    logits[0] = 10.0
    return logits


@pytest.fixture
def sample_logits_large_vocab() -> np.ndarray:
    """Return random logits for a larger vocabulary (32000).

    Uses a fixed RNG seed for reproducibility.  Simulates a realistic
    LLM logit distribution.
    """
    rng = np.random.default_rng(seed=12345)
    return rng.standard_normal(32000).astype(np.float64)
