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
from passes.flight.ballistic_entry import (
    EXPONENTIAL_ATMOSPHERE_EARTH,
    MINIMUM_BALLISTIC_ENTRY_ANGLE,
    BallisticEntry,
    ExponentialAtmosphere,
    allen_eggers_velocity,
    ballistic_entry_range,
    peak_deceleration,
    peak_deceleration_altitude,
)

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


class TestBallisticEntry:
    """Allen-Eggers, checked against numerical integration of the
    unapproximated point-mass equations rather than against itself."""

    @staticmethod
    def _integrate(v0, gamma, beta, h0=120.0e3, gravity=True):
        """Full 2-D point-mass entry. ``gamma`` is positive downward, so
        gravity *steepens* the descent: d(gamma)/dt = +g cos(gamma)/V.

        That sign is the whole content of the check. Getting it backwards
        shallows the trajectory instead, and produces ranges of thousands
        of kilometres for a steep entry -- which is how it was caught.
        """
        from scipy.integrate import solve_ivp

        g0 = 9.80665

        def rhs(_t, y):
            v, gam, h, _s = y
            drag = 0.5 * float(EXPONENTIAL_ATMOSPHERE_EARTH.density(h)) * v * v / beta
            g = g0 if gravity else 0.0
            return [
                -drag + g * np.sin(gam),
                g * np.cos(gam) / max(v, 1e-3),
                -v * np.sin(gam),
                v * np.cos(gam),
            ]

        def ground(_t, y):
            return y[2]

        ground.terminal = True
        ground.direction = -1
        return solve_ivp(
            rhs,
            [0.0, 4000.0],
            [v0, gamma, h0, 0.0],
            events=ground,
            rtol=1e-10,
            atol=1e-10,
            max_step=0.5,
        )

    def test_velocity_matches_numerical_integration_without_gravity(self):
        """The closed form neglects gravity, so switching gravity off in the
        reference isolates the integration itself. Agreement should be at
        the level of the integrator, not of the physics."""
        for gamma_deg, beta in ((20.0, 7500.0), (45.0, 1000.0), (60.0, 7500.0)):
            gamma = np.deg2rad(gamma_deg)
            sol = self._integrate(6500.0, gamma, beta, gravity=False)
            altitudes = np.linspace(1.0e3, 119.0e3, 40)
            numerical = np.interp(altitudes, sol.y[2][::-1], sol.y[0][::-1])
            closed = allen_eggers_velocity(altitudes, 6500.0, gamma, beta)
            assert np.max(np.abs(closed - numerical) / numerical) < 1.0e-3

    def test_peak_deceleration_is_independent_of_the_vehicle(self):
        """The Allen-Eggers result that inverts intuition: a_max contains no
        vehicle property. Verified by integrating a 100x span of ballistic
        coefficient and finding the same peak.

        The span stops at 5000 kg/m^2 deliberately -- beyond that the peak
        falls below sea level, the maximum becomes an endpoint rather than a
        stationary point, and the formula correctly stops applying.
        """
        gamma = np.deg2rad(45.0)
        peaks = []
        for beta in (50.0, 500.0, 5000.0):
            sol = self._integrate(6500.0, gamma, beta, gravity=False)
            density = np.asarray(EXPONENTIAL_ATMOSPHERE_EARTH.density(sol.y[2]))
            peaks.append(float((0.5 * density * sol.y[0] ** 2 / beta).max()))
        spread = (max(peaks) - min(peaks)) / float(np.mean(peaks))
        assert spread < 1.0e-3, f"a_max moved by {spread:.2%} across 100x in beta"
        assert float(np.mean(peaks)) == pytest.approx(
            peak_deceleration(6500.0, gamma), rel=1.0e-3
        )

    def test_peak_deceleration_altitude_matches_numerical(self):
        """This is the assertion that caught a factor of two. The altitude
        was written with rho_crit = 2 beta sin(gamma)/H, which places every
        peak exactly H ln 2 = 4.85 km too low -- a constant offset, and
        therefore invisible in any single case taken alone."""
        for gamma_deg in (20.0, 45.0, 60.0):
            for beta in (65.0, 1000.0, 7500.0):
                gamma = np.deg2rad(gamma_deg)
                sol = self._integrate(6500.0, gamma, beta, gravity=False)
                density = np.asarray(EXPONENTIAL_ATMOSPHERE_EARTH.density(sol.y[2]))
                accel = 0.5 * density * sol.y[0] ** 2 / beta
                observed = float(sol.y[2][int(np.argmax(accel))])
                predicted = peak_deceleration_altitude(gamma, beta)
                assert abs(predicted - observed) < 250.0, (
                    f"gamma={gamma_deg} beta={beta}: predicted {predicted:.0f} m, "
                    f"observed {observed:.0f} m"
                )

    def test_range_is_an_upper_bound_and_by_the_documented_amount(self):
        """h/tan(gamma) always overstates, because the real trajectory
        steepens. The docstring publishes an error table; this checks the
        corner of it that the mission budget actually uses -- a heavy
        vehicle on a steep arc -- and checks the sign everywhere else."""
        for gamma_deg in (15.0, 20.0, 30.0, 45.0, 60.0):
            gamma = np.deg2rad(gamma_deg)
            for beta in (65.0, 500.0, 2000.0, 7500.0):
                sol = self._integrate(6500.0, gamma, beta, gravity=True)
                true_range = float(sol.y[3][-1])
                bound = ballistic_entry_range(gamma)
                assert bound > true_range, "the geometric range must be an upper bound"
                if beta >= 2000.0 and gamma_deg >= 30.0:
                    assert (bound - true_range) / true_range < 0.08

    def test_refuses_a_shallow_entry_rather_than_approximating_it(self):
        """Below 15 degrees the constant-gamma assumption fails badly enough
        that a returned number would be misleading. Refusing is the point:
        the previous behaviour, a flat 300 km constant, never refused
        anything."""
        with pytest.raises(ValueError, match="entry_angle"):
            ballistic_entry_range(np.deg2rad(9.0))
        with pytest.raises(ValueError, match="entry_angle"):
            BallisticEntry(6500.0, np.deg2rad(5.0))
        assert ballistic_entry_range(MINIMUM_BALLISTIC_ENTRY_ANGLE) > 0.0

    def test_replaces_the_old_constant_and_shows_what_it_assumed(self):
        """The hardcoded 300 km corresponded to one particular entry angle.
        Recovering that angle is what makes the replacement auditable rather
        than merely different."""
        implied = np.arctan(120.0e3 / 300.0e3)
        assert np.rad2deg(implied) == pytest.approx(21.8, abs=0.1)
        assert ballistic_entry_range(implied) == pytest.approx(300.0e3, rel=1e-9)
        # And the constant understated a steep entry by up to 2.5x.
        assert 300.0e3 / ballistic_entry_range(np.deg2rad(60.0)) > 4.0

    def test_impact_velocity_separates_a_warhead_from_a_capsule(self):
        """The number that distinguishes the two ballistic-entry regimes: a
        dense vehicle arrives still hypersonic, a light one does not."""
        steep = np.deg2rad(30.0)
        warhead = BallisticEntry(6500.0, steep, ballistic_coefficient=7500.0)
        capsule = BallisticEntry(6500.0, steep, ballistic_coefficient=65.0)
        assert warhead.impact_velocity == pytest.approx(1776.0, rel=1e-3)
        assert warhead.impact_velocity / 6500.0 == pytest.approx(0.273, abs=0.002)
        assert capsule.impact_velocity < 1.0

    def test_a_zero_impact_velocity_is_flagged_as_out_of_validity(self):
        """The closed form neglects gravity, so for a light vehicle it
        decelerates toward zero where a real one settles at terminal
        velocity. Reporting that zero as an impact speed would be wrong by
        the whole value, so the object says when it does not apply.

        This is the failure mode worth guarding: the formula does not blow
        up or return a NaN, it returns a plausible-looking small number."""
        steep = np.deg2rad(30.0)
        capsule = BallisticEntry(6500.0, steep, ballistic_coefficient=65.0)
        warhead = BallisticEntry(6500.0, steep, ballistic_coefficient=7500.0)
        assert not capsule.allen_eggers_applicable_at_impact
        assert warhead.allen_eggers_applicable_at_impact
        # The physical answer the closed form is missing.
        assert capsule.terminal_velocity == pytest.approx(30.3, rel=0.01)
        assert capsule.impact_velocity < capsule.terminal_velocity
        # And where it does apply, it is far above terminal.
        assert warhead.impact_velocity > 5.0 * warhead.terminal_velocity

    def test_entry_interface_is_an_exact_boundary_condition(self):
        """Retaining rho(h_E) rather than dropping it as small keeps
        V(h_E) = V_E exactly. Dropping it would make the boundary condition
        approximate for no benefit."""
        entry = BallisticEntry(6500.0, np.deg2rad(30.0))
        assert float(entry.velocity(entry.entry_altitude)) == pytest.approx(6500.0, rel=1e-15)

    def test_peak_below_the_ground_is_reported_not_clamped(self):
        """A heavy, steep entry never peaks: it hits the ground still
        accelerating. A negative altitude says so; clamping to zero would
        hide it."""
        assert peak_deceleration_altitude(np.deg2rad(60.0), 25000.0) < 0.0

    def test_descent_time_matches_numerical_integration(self):
        """The closed form neglects gravity, so the gravity-free reference
        is the like-for-like check; it should agree to the integrator.
        Gravity-on is then reported for scale: it is 2-11% shorter, because
        gravity accelerates the descent, which is the right direction."""
        for gamma_deg, beta in ((30.0, 7500.0), (45.0, 7500.0), (60.0, 20000.0)):
            gamma = np.deg2rad(gamma_deg)
            entry = BallisticEntry(6500.0, gamma, ballistic_coefficient=beta)
            reference = self._integrate(6500.0, gamma, beta, gravity=False).t[-1]
            assert entry.descent_time == pytest.approx(reference, rel=1e-3)
            with_gravity = self._integrate(6500.0, gamma, beta, gravity=True).t[-1]
            assert with_gravity < entry.descent_time
            assert (entry.descent_time - with_gravity) / with_gravity < 0.15

    def test_descent_time_refuses_where_the_integrand_diverges(self):
        """The integrand goes as 1/V, so a vehicle the closed form
        decelerates toward zero gives a divergent integral dominated by the
        regime the model omits. Returning a large float would look like an
        answer; raising says it is not one."""
        shallow = BallisticEntry(6500.0, np.deg2rad(30.0), ballistic_coefficient=2000.0)
        assert not shallow.allen_eggers_applicable_at_impact
        with pytest.raises(ValueError, match="does not reach the ground"):
            _ = shallow.descent_time

    def test_descent_time_shortens_with_steepness(self):
        """A steeper entry covers the same altitude over a shorter path at
        higher speed, so it must take less time."""
        times = [
            BallisticEntry(6500.0, np.deg2rad(d), ballistic_coefficient=20000.0).descent_time
            for d in (20.0, 30.0, 45.0, 60.0)
        ]
        assert times == sorted(times, reverse=True)

    def test_validation(self):
        with pytest.raises(ValueError, match="entry_velocity"):
            BallisticEntry(-1.0, np.deg2rad(30.0))
        with pytest.raises(ValueError, match="ballistic_coefficient"):
            BallisticEntry(6500.0, np.deg2rad(30.0), ballistic_coefficient=0.0)
        with pytest.raises(ValueError, match="scale_height"):
            ExponentialAtmosphere(1.39, 0.0)
        with pytest.raises(ValueError, match="sea_level_density"):
            ExponentialAtmosphere(0.0, 7000.0)
        with pytest.raises(ValueError, match="entry_angle"):
            peak_deceleration(6500.0, np.deg2rad(120.0))
