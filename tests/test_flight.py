"""Coupled single-trajectory simulator (Paper I §3.5; Paper II §7.2)."""

import inspect

import numpy as np
import pytest

from passes.batch import cuda_available
from passes.flight import (
    FlightConfiguration,
    FlightSimulator,
    GlobalState,
    StateLayout,
)
from passes.flight import simulator as simulator_module

ENTRY = {"altitude": 120e3, "speed": 6500.0, "flight_path_angle": np.deg2rad(-8.0)}


@pytest.fixture(scope="module")
def sim():
    return FlightSimulator(FlightConfiguration(n_modes=4))


class TestStateLayout:
    def test_slices_tile_the_vector_without_overlap(self):
        layout = StateLayout(n_modes=5, n_thermal=9)
        covered = np.zeros(layout.size, dtype=int)
        for sl in (layout.position, layout.velocity, layout.quaternion,
                   layout.angular_rate, layout.modal_displacement,
                   layout.modal_velocity, layout.temperature, layout.densities):
            covered[sl] += 1
        covered[layout.mass] += 1
        covered[layout.recession] += 1
        assert np.all(covered == 1), "layout must partition the state exactly"

    def test_size_depends_only_on_configuration(self):
        layout = StateLayout(n_modes=6, n_thermal=11)
        assert layout.size == 15 + 2 * 6 + 4 * 11

    def test_pack_unpack_roundtrip(self):
        layout = StateLayout(n_modes=3, n_thermal=7)
        rng = np.random.default_rng(0)
        y = rng.standard_normal(layout.size)
        assert np.array_equal(GlobalState.unpack(y, layout).pack(layout), y)

    def test_validation(self):
        with pytest.raises(ValueError, match="n_thermal"):
            StateLayout(n_modes=2, n_thermal=1)
        with pytest.raises(ValueError, match="shape"):
            GlobalState.unpack(np.zeros(3), StateLayout(n_modes=2, n_thermal=5))


class TestNoRegimeBranching:
    def test_rhs_source_contains_no_altitude_branch(self):
        """Paper II §7.2: the atmosphere decays smoothly and no branch is
        taken at the Kármán line. Asserted against the source, because a
        branch is exactly the kind of thing that creeps back in."""
        source = inspect.getsource(simulator_module.FlightSimulator.rhs)
        lowered = source.lower()
        for forbidden in ("if altitude", "if alt ", "karman", "kármán"):
            assert forbidden not in lowered

    def test_rhs_is_continuous_across_the_karman_line(self, sim):
        """The right-hand side must not jump anywhere near 100 km."""
        base = sim.initial_state(**ENTRY)
        layout = sim.layout
        rates = []
        altitudes = np.linspace(95e3, 105e3, 41)
        for alt in altitudes:
            y = base.copy()
            y[layout.position] = np.array([6378137.0 + alt, 0.0, 0.0])
            rates.append(sim.rhs(0.0, y)[layout.velocity])
        rates = np.asarray(rates)
        jumps = np.linalg.norm(np.diff(rates, axis=0), axis=1)
        assert np.max(jumps) < 10.0 * np.median(jumps) + 1e-9

    def test_aero_terms_vanish_high_and_dominate_low(self, sim):
        layout = sim.layout
        base = sim.initial_state(**ENTRY)

        def drag_magnitude(alt):
            y = base.copy()
            y[layout.position] = np.array([6378137.0 + alt, 0.0, 0.0])
            accel = sim.rhs(0.0, y)[layout.velocity]
            gravity_only = 3.986004418e14 / (6378137.0 + alt) ** 2
            return float(np.linalg.norm(accel)) / gravity_only

        # at 400 km only gravity acts; the residual is the J2 term, which is
        # a ~1e-3 relative correction, not drag
        assert drag_magnitude(400e3) == pytest.approx(1.0, abs=5e-3)
        assert drag_magnitude(30e3) > 5.0


class TestCoupling:
    def test_recession_grows_the_effective_radius(self, sim):
        assert sim.effective_radius(0.0) == pytest.approx(sim.config.nose_radius)
        assert sim.effective_radius(0.01) == pytest.approx(sim.config.nose_radius + 0.01)

    def test_blunting_is_self_limiting(self, sim):
        """Paper II §4.1: recession grows R_eff, which reduces convective
        heating as R_eff^{-1/2}. The feedback must have the right sign."""
        from passes.aerothermal import sutton_graves

        q_sharp = float(sutton_graves(1e-3, sim.effective_radius(0.0), 6000.0))
        q_blunt = float(sutton_graves(1e-3, sim.effective_radius(0.05), 6000.0))
        assert q_blunt < q_sharp

    def test_recession_is_monotone_and_bounded(self, sim):
        y0 = sim.initial_state(**ENTRY)
        result = sim.propagate(y0, 60.0, n_output=31)
        recession = result.recession
        # the rate is non-negative by construction; the sampled output carries
        # interpolation noise at the integrator tolerance
        tol = 1e-8 * float(recession[-1]) + 1e-12
        assert np.min(np.diff(recession)) >= -tol, "recession is irreversible"
        assert 0.0 <= recession[-1] < sim.config.tps_thickness

    def test_surface_heats_and_stays_physical(self, sim):
        y0 = sim.initial_state(**ENTRY)
        result = sim.propagate(y0, 60.0, n_output=31)
        wall = result.surface_temperature
        assert wall[-1] > wall[0]
        assert np.all(wall > 0.0) and np.all(wall < 6000.0)

    def test_structural_response_tracks_dynamic_pressure(self, sim):
        y0 = sim.initial_state(**ENTRY)
        result = sim.propagate(y0, 60.0, n_output=31)
        modal = result.states[sim.layout.modal_displacement, :]
        elastic = np.max(np.abs(modal[2:, :]), axis=0)
        assert elastic[-1] > elastic[0], "aero loading must excite the structure"


class TestSingleIntegration:
    def test_state_dimension_is_constant(self, sim):
        y0 = sim.initial_state(**ENTRY)
        result = sim.propagate(y0, 60.0, n_output=31)
        assert result.states.shape == (sim.layout.size, 31)
        assert sim.layout.size == len(y0)

    def test_trajectory_descends_and_decelerates(self, sim):
        """Entry accelerates first — gravity dominates while the air is thin —
        and only decelerates once dynamic pressure builds, so the test looks
        for the peak rather than assuming monotone slowing."""
        y0 = sim.initial_state(**ENTRY)
        result = sim.propagate(y0, 110.0, n_output=56)
        altitude = result.altitude
        assert altitude[-1] < altitude[0]
        speeds = np.linalg.norm(result.states[sim.layout.velocity, :], axis=0)
        assert speeds.max() > speeds[0], "gravity accelerates the early arc"
        assert speeds[-1] < speeds.max(), "drag must eventually decelerate it"
        assert result.dynamic_pressure[-1] > result.dynamic_pressure[0]

    def test_quaternion_norm_preserved(self, sim):
        y0 = sim.initial_state(**ENTRY)
        y0[sim.layout.quaternion] = np.array([1.02, 0.0, 0.0, 0.0])
        result = sim.propagate(y0, 20.0, n_output=11)
        assert result.quaternion_norm_error[-1] < result.quaternion_norm_error[0]

    def test_implicit_method_is_insensitive_to_retained_modes(self):
        """Paper I Prop. 2 / Remark 4: the structural stiffness sets the
        explicit step. A fully implicit method removes the constraint, so
        the cost must not blow up as stiffer modes are retained."""
        costs = []
        for n_modes in (2, 4):
            simulator = FlightSimulator(FlightConfiguration(n_modes=n_modes))
            y0 = simulator.initial_state(**ENTRY)
            costs.append(simulator.propagate(y0, 30.0, n_output=11).n_rhs_evaluations)
        assert costs[1] < 4 * costs[0], f"cost blew up with mode count: {costs}"

    def test_validation(self, sim):
        with pytest.raises(ValueError, match="initial_state"):
            sim.propagate(np.zeros(3), 10.0)
        with pytest.raises(ValueError, match="duration"):
            sim.propagate(sim.initial_state(**ENTRY), 0.0)
        with pytest.raises(ValueError, match="beam_order"):
            FlightConfiguration(beam_order=3)
        with pytest.raises(ValueError, match="thermal_baumgarte_gain"):
            FlightConfiguration(thermal_baumgarte_gain=0.0)


@pytest.mark.skipif(not cuda_available(), reason="no CUDA device")
class TestOccupancy:
    @staticmethod
    def _kernel():
        import cupy

        kernel = cupy.RawKernel(
            r"""
extern "C" __global__ void probe(const double* y, double* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) { out[i] = y[i] * 2.0 + 1.0; }
}""",
            "probe",
        )
        kernel.compile()
        return kernel

    def test_occupancy_is_a_fraction_and_consistent(self):
        from passes.batch import device_limits, theoretical_occupancy

        limits = device_limits()
        for threads in (128, 256, 512):
            report = theoretical_occupancy(self._kernel(), threads)
            assert 0.0 < report.occupancy <= 1.0
            assert report.active_warps_per_sm <= report.max_warps_per_sm
            assert report.max_warps_per_sm == (
                limits["max_threads_per_sm"] // limits["warp_size"]
            )
            assert report.limiter in ("registers", "shared_memory", "warps", "blocks")

    def test_occupancy_falls_at_pathological_block_sizes(self):
        """1024 threads/block does not divide the 1536-thread SM limit, so
        occupancy must drop — a real effect the model has to capture."""
        from passes.batch import theoretical_occupancy

        good = theoretical_occupancy(self._kernel(), 256).occupancy
        awkward = theoretical_occupancy(self._kernel(), 1024).occupancy
        assert awkward < good

    def test_validation(self):
        from passes.batch import theoretical_occupancy

        with pytest.raises(ValueError, match="threads_per_block"):
            theoretical_occupancy(self._kernel(), 0)
        with pytest.raises(ValueError, match="exceeds the device limit"):
            theoretical_occupancy(self._kernel(), 4096)
