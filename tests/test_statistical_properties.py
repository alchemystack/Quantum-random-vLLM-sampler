"""Statistical property tests for the quantum consciousness sampling pipeline.

These tests validate mathematical invariants of the system rather than
individual code paths.  They use real (os.urandom) and biased (mock)
entropy sources to verify:

1. **Uniformity under null hypothesis** — u-values produced by the
   ZScoreMeanAmplifier with OsUrandomSource should be uniform on (0, 1).
   Validated via the Kolmogorov-Smirnov test.

2. **Consciousness bias simulation** — MockUniformSource with mean > 127.5
   should systematically shift u-values above 0.5, confirming that the
   z-score pipeline translates byte-level bias into CDF-position bias.

3. **EDT temperature-entropy correlation** — Higher-entropy logit
   distributions should produce higher temperatures under the EDT
   strategy, confirming the formula's monotonic coupling.

Dependencies:
    scipy (KS test) — listed in [project.optional-dependencies] dev.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import stats

from qc_sampler.config import QCSamplingConfig
from qc_sampler.entropy_source import MockUniformSource, OsUrandomSource
from qc_sampler.signal_amplifier import ZScoreMeanAmplifier
from qc_sampler.temperature_strategy import EDTTemperatureStrategy

# ---------------------------------------------------------------------------
# Configurable constants — no hardcoded iteration counts in test bodies.
# ---------------------------------------------------------------------------

# Number of u-values to generate for uniformity / bias tests.
# 10,000 gives good statistical power without making CI slow.
_NUM_SAMPLES: int = 10_000

# Significance level for the KS test.  p > this threshold means we
# cannot reject the null hypothesis of uniformity.
_KS_ALPHA: float = 0.01

# Sample count (bytes per u-value) used in the amplifier pipeline.
# Matches the production default.
_SAMPLE_COUNT: int = 20_480


# ---------------------------------------------------------------------------
# Group 1: Uniformity under null hypothesis
# ---------------------------------------------------------------------------


class TestUniformityUnderNullHypothesis:
    """Verify that u-values are uniform on (0, 1) when the entropy source
    has no bias (os.urandom is indistinguishable from uniform for our
    purposes).

    The probability integral transform guarantees this property when:
    - The raw bytes are i.i.d. uniform uint8.
    - The z-score is computed correctly.
    - The normal CDF is applied correctly.

    We use the Kolmogorov-Smirnov test against the continuous uniform
    distribution on [0, 1].
    """

    @pytest.fixture()
    def u_values(self) -> np.ndarray:
        """Generate u-values from the full pipeline under the null hypothesis.

        Uses OsUrandomSource → ZScoreMeanAmplifier with default config.
        Each u-value consumes _SAMPLE_COUNT bytes, so this fixture
        generates _NUM_SAMPLES × _SAMPLE_COUNT = ~200 MB of random data.
        """
        config = QCSamplingConfig(sample_count=_SAMPLE_COUNT)
        source = OsUrandomSource()
        amplifier = ZScoreMeanAmplifier(config)

        values = np.empty(_NUM_SAMPLES, dtype=np.float64)
        for i in range(_NUM_SAMPLES):
            raw = source.get_bytes(_SAMPLE_COUNT)
            result = amplifier.amplify(raw)
            values[i] = result.u
        return values

    def test_ks_test_against_uniform(self, u_values: np.ndarray) -> None:
        """KS test: u-values should follow Uniform(0, 1).

        We test at significance level _KS_ALPHA.  The null hypothesis
        is that the sample comes from a standard uniform distribution.
        Failing this test (p < _KS_ALPHA) would indicate a bug in the
        z-score → CDF pipeline.
        """
        ks_stat, p_value = stats.kstest(u_values, "uniform")

        assert p_value > _KS_ALPHA, (
            f"KS test rejected uniformity: statistic={ks_stat:.6f}, "
            f"p-value={p_value:.6f} (threshold: {_KS_ALPHA}). "
            f"This suggests a bug in the z-score → CDF pipeline."
        )

    def test_mean_near_half(self, u_values: np.ndarray) -> None:
        """Under the null hypothesis, E[u] = 0.5.

        We allow a generous tolerance because os.urandom is not a true
        QRNG, but its mean should still be close to 0.5 with N = 10,000.
        Standard error of the mean = 1/sqrt(12*N) ≈ 0.0029, so 5σ ≈ 0.015.
        """
        mean_u = float(np.mean(u_values))
        tolerance = 5.0 / math.sqrt(12.0 * _NUM_SAMPLES)  # ~5σ
        assert abs(mean_u - 0.5) < tolerance, (
            f"Mean u = {mean_u:.6f}, expected ~0.5 "
            f"(tolerance: ±{tolerance:.6f})"
        )

    def test_values_span_full_range(self, u_values: np.ndarray) -> None:
        """u-values should span roughly the full (0, 1) interval.

        With 10,000 samples, the minimum should be well below 0.1 and
        the maximum well above 0.9.  If all values cluster in a narrow
        band, the pipeline has a scaling bug.
        """
        assert float(np.min(u_values)) < 0.1, (
            f"Minimum u = {np.min(u_values):.6f} — expected < 0.1"
        )
        assert float(np.max(u_values)) > 0.9, (
            f"Maximum u = {np.max(u_values):.6f} — expected > 0.9"
        )


# ---------------------------------------------------------------------------
# Group 2: Consciousness bias simulation
# ---------------------------------------------------------------------------


class TestConsciousnessBiasSimulation:
    """Verify that a biased entropy source shifts u-values predictably.

    MockUniformSource(mean=128.0) produces bytes whose average is ~0.5
    above the null mean of 127.5.  Through the z-score pipeline, this
    should systematically push u > 0.5, confirming the directional
    coupling between QRNG bias and CDF-position bias.
    """

    @pytest.fixture()
    def biased_u_values(self) -> np.ndarray:
        """Generate u-values from a biased mock source (mean=128.0)."""
        config = QCSamplingConfig(sample_count=_SAMPLE_COUNT)
        # Use a fixed seed for reproducibility across CI runs.
        source = MockUniformSource(mean=128.0, seed=2024)
        amplifier = ZScoreMeanAmplifier(config)

        values = np.empty(_NUM_SAMPLES, dtype=np.float64)
        for i in range(_NUM_SAMPLES):
            raw = source.get_bytes(_SAMPLE_COUNT)
            result = amplifier.amplify(raw)
            values[i] = result.u
        return values

    def test_mean_u_above_half(self, biased_u_values: np.ndarray) -> None:
        """With upward-biased bytes, mean u should be significantly > 0.5.

        A mean shift of +0.5 in byte values, divided by SEM ≈ 0.514,
        gives z ≈ 0.97, so u ≈ Φ(0.97) ≈ 0.83.  We only require
        mean u > 0.5 (the spec's requirement), but in practice it
        should be much higher.
        """
        mean_u = float(np.mean(biased_u_values))
        assert mean_u > 0.5, (
            f"Mean u = {mean_u:.6f} — expected > 0.5 with biased source "
            f"(mean=128.0)"
        )

    def test_majority_of_values_above_half(
        self, biased_u_values: np.ndarray
    ) -> None:
        """A clear majority of u-values should be above 0.5.

        With a +0.5 byte-mean shift, essentially all u-values cluster
        around 0.83.  We check that at least 75% exceed 0.5 as a
        conservative lower bound.
        """
        fraction_above = float(np.mean(biased_u_values > 0.5))
        assert fraction_above > 0.75, (
            f"Only {fraction_above:.1%} of u-values > 0.5 — "
            f"expected > 75% with biased source"
        )

    def test_ks_test_rejects_uniformity(
        self, biased_u_values: np.ndarray
    ) -> None:
        """With biased bytes, the KS test should *reject* uniformity.

        This confirms that the bias propagates through the pipeline
        and produces a non-uniform u-distribution.  If the KS test
        still passes, the pipeline is ignoring the bias.
        """
        _, p_value = stats.kstest(biased_u_values, "uniform")
        assert p_value < _KS_ALPHA, (
            f"KS test did NOT reject uniformity (p={p_value:.6f}) — "
            f"the bias is not propagating through the pipeline"
        )

    def test_stronger_bias_gives_higher_mean_u(self) -> None:
        """Stronger byte-mean bias should produce higher mean u.

        Compare mean=128.0 (mild bias) vs mean=129.0 (stronger bias).
        The stronger bias should give a higher average u.
        """
        config = QCSamplingConfig(sample_count=_SAMPLE_COUNT)
        amplifier = ZScoreMeanAmplifier(config)
        # Use fewer samples for this comparison — we only need direction.
        num_comparison_samples = _NUM_SAMPLES // 5

        means_by_bias: dict[float, float] = {}
        for bias_mean in [128.0, 129.0]:
            source = MockUniformSource(mean=bias_mean, seed=42)
            values = np.empty(num_comparison_samples, dtype=np.float64)
            for i in range(num_comparison_samples):
                raw = source.get_bytes(_SAMPLE_COUNT)
                result = amplifier.amplify(raw)
                values[i] = result.u
            means_by_bias[bias_mean] = float(np.mean(values))

        assert means_by_bias[129.0] > means_by_bias[128.0], (
            f"Stronger bias (129.0) did not produce higher mean u: "
            f"u(128.0)={means_by_bias[128.0]:.6f}, "
            f"u(129.0)={means_by_bias[129.0]:.6f}"
        )


# ---------------------------------------------------------------------------
# Group 3: EDT temperature-entropy correlation
# ---------------------------------------------------------------------------


class TestEDTTemperatureEntropyCorrelation:
    """Verify that EDT produces temperature correlated with entropy.

    The EDT formula T = base × H_norm^θ is monotonically non-decreasing
    in H_norm (for θ > 0 and base > 0).  We verify this by creating
    logit distributions with known, increasing entropy and checking
    that the computed temperatures are non-decreasing.
    """

    # Vocabulary size for EDT tests.
    _VOCAB_SIZE: int = 32_000

    def _make_logits_with_n_active(self, n_active: int) -> np.ndarray:
        """Create logits with exactly n_active equally likely tokens.

        All other tokens are masked to -1e6 (effectively zero prob).
        Shannon entropy = ln(n_active).

        Args:
            n_active: Number of tokens with non-negligible probability.

        Returns:
            Logit array of shape (self._VOCAB_SIZE,).
        """
        logits = np.full(self._VOCAB_SIZE, -1e6, dtype=np.float64)
        logits[:n_active] = 0.0
        return logits

    def test_monotonicity_across_entropy_levels(self) -> None:
        """Temperature must be non-decreasing as entropy increases.

        We test with 1, 2, 10, 100, 1000, and vocab_size active tokens.
        Each step increases entropy, so temperature should increase
        (or stay equal if clamped).
        """
        config = QCSamplingConfig(
            edt_base_temp=1.0,
            edt_exponent=0.5,
            edt_min_temp=0.01,
            edt_max_temp=10.0,
        )
        strategy = EDTTemperatureStrategy(vocab_size=self._VOCAB_SIZE)

        active_counts = [1, 2, 10, 100, 1000, self._VOCAB_SIZE]
        temperatures: list[float] = []

        for n_active in active_counts:
            logits = self._make_logits_with_n_active(n_active)
            result = strategy.compute_temperature(logits, config)
            temperatures.append(result.temperature)

        for i in range(1, len(temperatures)):
            assert temperatures[i] >= temperatures[i - 1], (
                f"Monotonicity violated: T({active_counts[i]} active) = "
                f"{temperatures[i]:.6f} < T({active_counts[i-1]} active) = "
                f"{temperatures[i-1]:.6f}"
            )

    def test_single_token_gets_min_temp(self) -> None:
        """With only 1 active token, entropy = 0 → T clamped to min_temp."""
        config = QCSamplingConfig(
            edt_base_temp=1.0,
            edt_exponent=0.5,
            edt_min_temp=0.15,
            edt_max_temp=3.0,
        )
        strategy = EDTTemperatureStrategy(vocab_size=self._VOCAB_SIZE)
        logits = self._make_logits_with_n_active(1)

        result = strategy.compute_temperature(logits, config)

        assert result.temperature == pytest.approx(config.edt_min_temp)

    def test_uniform_gets_base_temp(self) -> None:
        """With all tokens equally likely, H_norm ≈ 1.0 → T ≈ base_temp.

        For any θ: base × 1.0^θ = base.
        """
        config = QCSamplingConfig(
            edt_base_temp=0.8,
            edt_exponent=0.5,
            edt_min_temp=0.01,
            edt_max_temp=3.0,
        )
        strategy = EDTTemperatureStrategy(vocab_size=self._VOCAB_SIZE)
        logits = self._make_logits_with_n_active(self._VOCAB_SIZE)

        result = strategy.compute_temperature(logits, config)

        assert result.temperature == pytest.approx(0.8, abs=0.01)

    def test_entropy_temperature_correlation_with_random_logits(
        self,
    ) -> None:
        """Over many random logit distributions, higher entropy should
        predict higher temperature on average.

        We generate 200 random logit arrays, compute entropy and
        temperature for each, and verify positive Spearman correlation.
        """
        num_distributions = 200
        rng = np.random.default_rng(seed=9999)

        config = QCSamplingConfig(
            edt_base_temp=1.0,
            edt_exponent=0.5,
            edt_min_temp=0.01,
            edt_max_temp=10.0,
        )
        strategy = EDTTemperatureStrategy(vocab_size=self._VOCAB_SIZE)

        entropies: list[float] = []
        temperatures: list[float] = []

        for _ in range(num_distributions):
            # Random logits with varying sparsity: multiply standard
            # normal by a random scale to get diverse entropy levels.
            scale = rng.uniform(0.1, 10.0)
            logits = rng.standard_normal(self._VOCAB_SIZE) * scale
            result = strategy.compute_temperature(logits, config)
            entropies.append(result.shannon_entropy)
            temperatures.append(result.temperature)

        # Spearman rank correlation: monotonic relationship.
        correlation, p_value = stats.spearmanr(entropies, temperatures)

        assert correlation > 0.5, (
            f"Spearman correlation between entropy and temperature = "
            f"{correlation:.4f} — expected > 0.5 (p={p_value:.6f})"
        )
        # The p-value should be very small (correlation is essentially 1.0
        # for a monotonic formula, modulo clamping).
        assert p_value < 0.01, (
            f"Correlation not significant: p={p_value:.6f}"
        )

    def test_different_exponents_same_ordering(self) -> None:
        """Different exponents (θ) should preserve temperature ordering.

        The EDT formula T = base × H_norm^θ is monotonic in H_norm
        for any θ > 0.  Different θ values change the *shape* of the
        curve but not the *ordering* of temperatures by entropy.
        """
        config_template = QCSamplingConfig(
            edt_base_temp=1.0,
            edt_min_temp=0.01,
            edt_max_temp=10.0,
        )
        strategy = EDTTemperatureStrategy(vocab_size=self._VOCAB_SIZE)

        active_counts = [2, 50, 500, 5000, self._VOCAB_SIZE]

        for exponent in [0.3, 0.5, 1.0, 2.0, 3.0]:
            config = QCSamplingConfig(
                edt_base_temp=config_template.edt_base_temp,
                edt_exponent=exponent,
                edt_min_temp=config_template.edt_min_temp,
                edt_max_temp=config_template.edt_max_temp,
            )

            temps: list[float] = []
            for n in active_counts:
                logits = self._make_logits_with_n_active(n)
                result = strategy.compute_temperature(logits, config)
                temps.append(result.temperature)

            for i in range(1, len(temps)):
                assert temps[i] >= temps[i - 1], (
                    f"Monotonicity violated with θ={exponent}: "
                    f"T({active_counts[i]})={temps[i]:.6f} < "
                    f"T({active_counts[i-1]})={temps[i-1]:.6f}"
                )
