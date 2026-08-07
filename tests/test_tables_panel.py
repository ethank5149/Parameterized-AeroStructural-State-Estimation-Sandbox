"""The panel solver against the reference vehicle's saved meshes."""

from pathlib import Path

import numpy as np
import pytest

from passes.aerodynamics.tables import PanelSolver, SweepGrid, SweepRun
from passes.geometry import load_stl

_VEHICLE = Path(__file__).resolve().parents[1] / "data" / "vehicles" / "rs28"
_HAVE = (_VEHICLE / "rs28-configuration-full-stack.stl").is_file()

_DIAMETER = 3.0
_AREA = np.pi * _DIAMETER**2 / 4.0

pytestmark = pytest.mark.skipif(not _HAVE, reason="vehicle meshes not built")


class TestPanelSolver:
    @pytest.fixture(scope="class")
    @classmethod
    def solver(cls):
        mesh = load_stl(_VEHICLE / "rs28-configuration-full-stack.stl")
        return PanelSolver(mesh, _AREA, _DIAMETER)

    def test_subsonic_input_is_refused_with_the_reason(self, solver):
        """Not a limitation to be remembered — a limitation that says so.
        Newtonian impact theory and Prandtl-Meyer expansion both have no
        subsonic branch."""
        with pytest.raises(ValueError, match="supersonic theory"):
            solver.solve(0.6, 0.0)

    def test_the_stack_axial_force_is_positive_and_settles(self, solver):
        """Newtonian pressure asymptotes as Mach grows, so the coefficient
        must flatten. For the *full stack* it also falls, because the
        expansion side dominates at low supersonic; that direction is a
        property of this configuration, not of the theory."""
        machs = [1.5, 2.0, 3.0, 5.0, 10.0, 20.0]
        axial = [solver.solve(m, 0.0).axial for m in machs]
        assert all(a > 0.0 for a in axial), "drag cannot be negative"
        assert axial == sorted(axial, reverse=True)
        assert abs(axial[-1] - axial[-2]) < 0.1 * abs(axial[0] - axial[1])

    def test_normal_force_is_near_zero_at_zero_incidence_and_grows_with_it(self, solver):
        # Not exactly zero: the mesh's circular sections are 40-gons, so the
        # body is not perfectly axisymmetric and the cancellation is only as
        # good as the faceting. Judged against the axial force, which is the
        # scale that matters.
        axial = solver.solve(5.0, 0.0).axial
        assert abs(solver.solve(5.0, 0.0).normal) < 0.02 * axial
        normals = [solver.solve(5.0, np.deg2rad(a)).normal for a in (2.0, 6.0, 10.0)]
        assert normals == sorted(normals)
        assert all(n > 0.0 for n in normals)

    def test_the_hypersonic_limit_is_physically_sized(self):
        """A slender cone-cylinder at high Mach: modified Newtonian on a
        24.6-degree cone alone gives 2 sin^2(24.6) = 0.35, and the body adds
        base and shoulder terms. An answer far from order-unity would mean
        the integration or the reference area is wrong."""
        mesh = load_stl(_VEHICLE / "rs28-configuration-full-stack.stl")
        axial = PanelSolver(mesh, _AREA, _DIAMETER).solve(20.0, 0.0).axial
        assert 0.3 < axial < 2.0

    def test_the_payload_alone_has_less_drag_area_than_the_stack(self):
        """Sanity across the saved configurations: a re-entry vehicle is a
        smaller object than the stack that launched it."""
        stack = load_stl(_VEHICLE / "rs28-configuration-full-stack.stl")
        payload = load_stl(_VEHICLE / "rs28-configuration-payload-only.stl")
        at_ten = [
            PanelSolver(m, _AREA, _DIAMETER).solve(10.0, 0.0).axial
            for m in (stack, payload)
        ]
        assert at_ten[1] < at_ten[0]


class TestEndToEnd:
    def test_a_small_sweep_completes_and_is_ordered(self, tmp_path):
        mesh = load_stl(_VEHICLE / "rs28-configuration-payload-only.stl")
        grid = SweepGrid(
            mach=np.array([2.0, 5.0, 10.0]), alpha=SweepGrid.default_alpha(4.0, 2.0)
        )
        run = SweepRun(
            "payload-only", grid, PanelSolver(mesh, _AREA, _DIAMETER),
            tmp_path / "t.jsonl", _AREA, _DIAMETER,
        )
        table = run.run()
        assert table.complete
        assert np.all(table.axial > 0.0)
        # C_A *approaches a limit* with Mach; it does not have to fall. The
        # direction depends on which term dominates: a nose-dominated body
        # like this one follows Cp_max, which rises toward the Newtonian
        # limit, so the payload's axial coefficient climbs 0.278 -> 0.307
        # while the full stack's falls. Asserting a direction encodes the
        # wrong invariant; asserting convergence encodes the right one.
        steps = np.abs(np.diff(table.axial, axis=0))
        assert np.all(steps[-1] < steps[0])
        # C_N grows with incidence at every Mach.
        assert np.all(np.diff(table.normal, axis=1) > 0.0)

    def test_drag_area_is_a_sane_number_for_the_simulator(self, tmp_path):
        """This is what should replace a hand-set `drag_area`. The stack's
        frontal area is 7.07 m2, so a C_D near one puts it in the same
        range — which is the check that the reference area and the
        coefficient agree."""
        mesh = load_stl(_VEHICLE / "rs28-configuration-full-stack.stl")
        grid = SweepGrid(mach=np.array([5.0, 10.0]), alpha=np.array([0.0]))
        run = SweepRun(
            "full-stack", grid, PanelSolver(mesh, _AREA, _DIAMETER),
            tmp_path / "t.jsonl", _AREA, _DIAMETER,
        )
        table = run.run()
        assert 3.0 < table.drag_area(10.0) < 15.0
