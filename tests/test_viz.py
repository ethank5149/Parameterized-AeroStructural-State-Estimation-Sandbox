"""Ray-traced globe rendering (presentation layer)."""

from pathlib import Path

import numpy as np
import pytest

from passes.geodesy import WGS84_MEAN_RADIUS
from passes.viz import (
    DEFAULT_TEXTURE,
    Camera,
    SimulationHistory,
    load_texture,
    look_at,
    project,
    render,
    sun_direction,
)

_R = WGS84_MEAN_RADIUS


def _checker(rows: int = 64, cols: int = 128) -> np.ndarray:
    """A synthetic texture whose value encodes position.

    Red carries longitude and green carries latitude, so a sampled colour
    says exactly where on the sphere it came from. That turns "does the
    texture land in the right place" from an eyeball question into an
    arithmetic one.
    """
    lat = np.linspace(0.5, -0.5, rows)[:, None] + 0.5
    lon = np.linspace(0.0, 1.0, cols)[None, :]
    return np.stack(
        [np.broadcast_to(lon, (rows, cols)), np.broadcast_to(lat, (rows, cols)),
         np.zeros((rows, cols))],
        axis=-1,
    )


class TestCamera:
    def test_basis_is_right_handed_and_orthonormal(self):
        camera = look_at(np.zeros(3), 3.0 * _R, 0.3, 0.2)
        right, up, forward = camera.basis()
        for vector in (right, up, forward):
            assert float(np.linalg.norm(vector)) == pytest.approx(1.0, abs=1e-12)
        assert right @ up == pytest.approx(0.0, abs=1e-12)
        assert right @ forward == pytest.approx(0.0, abs=1e-12)
        assert np.cross(right, up) @ forward == pytest.approx(-1.0, abs=1e-12)

    def test_look_at_places_the_eye_where_asked(self):
        target = np.array([1.0e6, -2.0e6, 3.0e5])
        camera = look_at(target, 5.0e6, np.deg2rad(35.0), np.deg2rad(20.0))
        offset = camera.position - target
        assert float(np.linalg.norm(offset)) == pytest.approx(5.0e6, rel=1e-12)
        assert np.rad2deg(np.arcsin(offset[2] / 5.0e6)) == pytest.approx(20.0, abs=1e-9)
        assert np.rad2deg(np.arctan2(offset[1], offset[0])) == pytest.approx(35.0, abs=1e-9)

    def test_degenerate_up_does_not_produce_nan(self):
        """Looking straight down the world up axis leaves roll undefined.
        A fallback axis is substituted rather than dividing by zero."""
        camera = Camera(
            position=np.array([0.0, 0.0, 3.0 * _R]),
            target=np.zeros(3),
            up=np.array([0.0, 0.0, 1.0]),
        )
        for vector in camera.basis():
            assert np.all(np.isfinite(vector))

    def test_coincident_position_and_target_is_refused(self):
        camera = Camera(position=np.zeros(3), target=np.zeros(3), up=np.array([0.0, 0.0, 1.0]))
        with pytest.raises(ValueError, match="no view direction"):
            camera.basis()


class TestProjection:
    def test_the_target_lands_at_the_frame_centre(self):
        target = np.array([2.0e6, 1.0e6, 5.0e5])
        camera = look_at(target, 4.0e6, 0.6, 0.3, width=641, height=481)
        px, py, visible = project(target[None, :], camera)
        assert visible[0]
        assert px[0] == pytest.approx(320.0, abs=1e-6)
        assert py[0] == pytest.approx(240.0, abs=1e-6)

    def test_points_behind_the_camera_are_not_visible(self):
        camera = look_at(np.zeros(3), 3.0 * _R, 0.0, 0.0)
        behind = camera.position + np.array([1.0e6, 0.0, 0.0])
        _, _, visible = project(behind[None, :], camera)
        assert not visible[0]

    def test_the_far_side_of_the_globe_is_occluded(self):
        """The depth test Matplotlib's painter's algorithm cannot do, and
        the reason trajectories used to appear in front of the planet they
        were behind."""
        camera = look_at(np.zeros(3), 4.0 * _R, 0.0, 0.0)
        near = np.array([[1.2 * _R, 0.0, 0.0]])
        far = np.array([[-1.2 * _R, 0.0, 0.0]])
        assert project(near, camera, radius=_R)[2][0]
        assert not project(far, camera, radius=_R)[2][0]

    def test_occlusion_is_only_applied_when_a_radius_is_given(self):
        camera = look_at(np.zeros(3), 4.0 * _R, 0.0, 0.0)
        far = np.array([[-1.2 * _R, 0.0, 0.0]])
        assert project(far, camera)[2][0]
        assert not project(far, camera, radius=_R)[2][0]

    def test_a_point_just_off_the_limb_stays_visible(self):
        """The occlusion test must not clip geometry that merely passes
        near the limb, which is exactly where an entry trajectory sits."""
        camera = look_at(np.zeros(3), 6.0 * _R, 0.0, 0.0)
        grazing = np.array([[0.0, 1.02 * _R, 0.0]])
        assert project(grazing, camera, radius=_R)[2][0]

    def test_rejects_malformed_input(self):
        camera = look_at(np.zeros(3), 3.0 * _R, 0.0, 0.0)
        with pytest.raises(ValueError, match=r"shape \(n, 3\)"):
            project(np.zeros((4, 2)), camera)


class TestRender:
    def test_the_disc_covers_the_expected_solid_angle(self):
        """The rendered disc must subtend arcsin(R/d), which is geometry
        rather than taste — a wrong field of view or a wrong ray
        construction shows up here immediately."""
        # Far enough that the whole disc fits inside the frame; at 3 R with
        # a 35 degree field of view it overflows, and the measured fraction
        # would be the frame rather than the disc.
        distance = 8.0 * _R
        camera = look_at(np.zeros(3), distance, 0.0, 0.0, width=400, height=400)
        _, depth = render(camera, _checker(), _R)
        hit_fraction = float(np.isfinite(depth).mean())

        half_angle = np.arcsin(_R / distance)
        half_frame = np.tan(0.5 * camera.fov)
        disc_radius = np.tan(half_angle) / half_frame
        assert disc_radius < 1.0, "the disc must fit inside the frame"
        expected = np.pi * disc_radius**2 / 4.0
        assert hit_fraction == pytest.approx(expected, rel=0.02)

    def test_texture_lands_where_it_should(self):
        """A synthetic texture encoding longitude in red and latitude in
        green means the centre pixel reports the sub-camera point. This is
        the check that catches a flipped or rolled texture, which is
        otherwise only visible as a mirrored continent."""
        for longitude_deg, latitude_deg in ((0.0, 0.0), (90.0, 0.0), (-120.0, 30.0)):
            camera = look_at(
                np.zeros(3),
                4.0 * _R,
                np.deg2rad(longitude_deg),
                np.deg2rad(latitude_deg),
                width=201,
                height=201,
            )
            image, _ = render(
                camera, _checker(256, 512), _R, sun=None, atmosphere=0.0, specular=0.0
            )
            red, green, _ = image[100, 100]
            # Red encodes longitude on [0, 1] over -180..180.
            assert red * 360.0 - 180.0 == pytest.approx(longitude_deg, abs=2.0)
            assert (green - 0.5) * 180.0 == pytest.approx(latitude_deg, abs=2.0)

    def test_the_terminator_falls_where_the_sun_puts_it(self):
        """With the sun 90 degrees from the sub-camera point, exactly half
        the visible disc is lit. Checked by brightness rather than by eye."""
        camera = look_at(np.zeros(3), 8.0 * _R, 0.0, 0.0, width=300, height=300)
        flat = np.ones((32, 64, 3))
        image, depth = render(
            camera,
            flat,
            _R,
            sun=sun_direction(np.deg2rad(90.0)),
            atmosphere=0.0,
            night=0.0,
            specular=0.0,
        )
        on_disc = np.isfinite(depth)
        brightness = image[..., 0]
        # With no night term the dark side is exactly zero, so any
        # illumination marks the lit hemisphere. Thresholding at half
        # brightness would instead find the cos = 0.33 contour, which is
        # not the terminator.
        lit = (brightness > 0.02) & on_disc
        assert lit.sum() / on_disc.sum() == pytest.approx(0.5, abs=0.06)
        # ...and it is the half the sun is on. With the camera on +x looking
        # inward, the basis puts world +y along image +x, so a sun at
        # hour angle 90 degrees lights the right of the frame.
        right_axis = camera.basis()[0]
        assert right_axis @ sun_direction(np.deg2rad(90.0)) > 0.9
        columns = np.nonzero(lit.any(axis=0))[0]
        assert columns.mean() > image.shape[1] / 2

    def test_lighting_from_the_camera_removes_the_terminator(self):
        camera = look_at(np.zeros(3), 5.0 * _R, 0.4, 0.2, width=200, height=200)
        flat = np.ones((32, 64, 3))
        image, depth = render(camera, flat, _R, sun=None, atmosphere=0.0, specular=0.0)
        on_disc = np.isfinite(depth)
        assert (image[..., 0][on_disc] > 0.5).mean() > 0.98

    def test_background_is_composited_where_the_ray_misses(self):
        camera = look_at(np.zeros(3), 3.0 * _R, 0.0, 0.0, width=120, height=120)
        background = np.zeros((120, 120, 3))
        background[..., 2] = 1.0
        image, depth = render(camera, _checker(), _R, background=background)
        missed = ~np.isfinite(depth)
        assert missed.any()
        assert np.allclose(image[missed], np.array([0.0, 0.0, 1.0]))

    def test_rejects_a_mismatched_background(self):
        camera = look_at(np.zeros(3), 3.0 * _R, 0.0, 0.0, width=64, height=48)
        with pytest.raises(ValueError, match="background must have shape"):
            render(camera, _checker(), _R, background=np.zeros((10, 10, 3)))

    def test_rejects_a_non_positive_radius(self):
        camera = look_at(np.zeros(3), 3.0 * _R, 0.0, 0.0, width=32, height=32)
        with pytest.raises(ValueError, match="radius must be positive"):
            render(camera, _checker(), 0.0)


class TestTexture:
    @pytest.mark.skipif(not Path(DEFAULT_TEXTURE).is_file(), reason="texture not present")
    def test_the_packaged_texture_is_equirectangular(self):
        texture = load_texture()
        rows, cols = texture.shape[:2]
        assert texture.shape[2] == 3
        assert cols == 2 * rows, "an equirectangular image must be 2:1"
        assert texture.min() >= 0.0 and texture.max() <= 1.0

    def test_a_missing_texture_says_what_was_expected(self):
        with pytest.raises(FileNotFoundError, match="equirectangular"):
            load_texture("does-not-exist.jpg")

    def test_the_default_path_does_not_depend_on_the_working_directory(self):
        """Resolved by walking up from the module, not from the process CWD.
        A CWD-relative default works from the repository root and fails from
        `notebooks/`, which is where the notebooks actually run."""
        assert Path(DEFAULT_TEXTURE).is_absolute()

    def test_the_default_texture_loads_from_any_directory(self, tmp_path, monkeypatch):
        if not Path(DEFAULT_TEXTURE).is_file():
            pytest.skip("texture not present")
        monkeypatch.chdir(tmp_path)
        assert load_texture().shape[2] == 3


class TestSimulationHistory:
    """The canonical run record the renderer samples instead of rebuilding.

    The point of this class is that a picture cannot disagree with the
    physics, so the tests are about *agreement with the producer* rather
    than about appearance.
    """

    @staticmethod
    def _trajectory():
        from passes.geodesy import GeodeticPosition
        from passes.orbital.scenario import fobs_trajectory

        return fobs_trajectory(
            GeodeticPosition.from_degrees(51.0, 59.0, 120.0, "launch"),
            GeodeticPosition.from_degrees(38.87, -77.06, 24.0, "target"),
            samples=300,
            earth_rotation=True,
        )

    def test_positions_reproduce_the_trajectory_altitudes_exactly(self):
        """The conversion must be lossless in the quantity it carries. If
        the radius comes back different from the altitude that produced it,
        the history is a second model rather than a view of the first."""
        from passes.geodesy import WGS84_MEAN_RADIUS

        trajectory = self._trajectory()
        history = SimulationHistory.from_trajectory(trajectory, WGS84_MEAN_RADIUS)
        recovered = history.altitudes(WGS84_MEAN_RADIUS)
        assert np.allclose(recovered, trajectory.altitudes, atol=1e-6)
        assert np.array_equal(history.times, trajectory.times)

    def test_a_scenario_trajectory_reports_that_it_has_no_attitude(self):
        """A point mass on a great circle has no attitude. Reporting that
        is the whole design: substituting an identity quaternion would draw
        an oriented vehicle from something never computed."""
        from passes.geodesy import WGS84_MEAN_RADIUS

        history = SimulationHistory.from_trajectory(self._trajectory(), WGS84_MEAN_RADIUS)
        assert not history.has_attitude
        assert history.quaternions is None
        assert "quaternion" not in history.sample(history.times[5])

    def test_sampling_at_a_recorded_time_returns_that_sample(self):
        from passes.geodesy import WGS84_MEAN_RADIUS

        trajectory = self._trajectory()
        history = SimulationHistory.from_trajectory(trajectory, WGS84_MEAN_RADIUS)
        for index in (0, 37, len(trajectory.times) - 1):
            state = history.sample(float(history.times[index]))
            assert np.allclose(state["position"], history.positions[index], atol=1e-6)

    def test_sampling_between_samples_lands_between_them(self):
        from passes.geodesy import WGS84_MEAN_RADIUS

        history = SimulationHistory.from_trajectory(self._trajectory(), WGS84_MEAN_RADIUS)
        midpoint = 0.5 * (history.times[10] + history.times[11])
        state = history.sample(midpoint)
        expected = 0.5 * (history.positions[10] + history.positions[11])
        assert np.allclose(state["position"], expected, rtol=1e-9)

    def test_times_outside_the_run_are_held_not_extrapolated(self):
        """A camera asking beyond the physics should see the last real
        state, not a linear guess past the end of it."""
        from passes.geodesy import WGS84_MEAN_RADIUS

        history = SimulationHistory.from_trajectory(self._trajectory(), WGS84_MEAN_RADIUS)
        before = history.sample(history.times[0] - 1000.0)
        after = history.sample(history.times[-1] + 1000.0)
        assert np.allclose(before["position"], history.positions[0])
        assert np.allclose(after["position"], history.positions[-1])

    def test_velocity_falls_back_to_finite_differences(self):
        """A producer without velocity still drives a chase camera, because
        the direction of travel is recoverable from the positions."""
        from passes.geodesy import WGS84_MEAN_RADIUS

        history = SimulationHistory.from_trajectory(self._trajectory(), WGS84_MEAN_RADIUS)
        assert history.velocities is None
        state = history.sample(float(history.times[40]))
        step = history.positions[41] - history.positions[40]
        assert float(np.dot(state["velocity"], step)) > 0.0

    def test_quaternion_interpolation_stays_on_the_unit_sphere(self):
        """Componentwise interpolation of a quaternion leaves the sphere,
        which both shrinks and shears the rotation it represents. Slerp
        does not."""
        times = np.array([0.0, 1.0])
        positions = np.array([[7e6, 0.0, 0.0], [7e6, 1e5, 0.0]])
        start = np.array([1.0, 0.0, 0.0, 0.0])
        end = np.array([np.cos(0.6), 0.0, 0.0, np.sin(0.6)])
        history = SimulationHistory(
            label="spin", times=times, positions=positions,
            quaternions=np.stack([start, end]),
        )
        assert history.has_attitude
        for blend in np.linspace(0.0, 1.0, 11):
            quaternion = history.sample(float(blend))["quaternion"]
            assert float(np.linalg.norm(quaternion)) == pytest.approx(1.0, abs=1e-12)

    def test_attitude_sampling_yields_a_proper_rotation(self):
        """The DCM handed to a renderer must be orthonormal with unit
        determinant, or a vehicle drawn with it is sheared."""
        times = np.array([0.0, 1.0])
        positions = np.array([[7e6, 0.0, 0.0], [7e6, 1e5, 0.0]])
        quaternions = np.stack(
            [np.array([1.0, 0.0, 0.0, 0.0]), np.array([np.cos(0.4), np.sin(0.4), 0.0, 0.0])]
        )
        history = SimulationHistory(
            label="spin", times=times, positions=positions, quaternions=quaternions
        )
        dcm = history.sample(0.5)["dcm"]
        assert np.allclose(dcm @ dcm.T, np.eye(3), atol=1e-12)
        assert float(np.linalg.det(dcm)) == pytest.approx(1.0, abs=1e-12)

    def test_slerp_takes_the_short_way_round(self):
        """q and -q are the same rotation. Without the sign fix the
        interpolation sweeps the long way and the vehicle visibly spins
        backwards through most of a revolution."""
        times = np.array([0.0, 1.0])
        positions = np.array([[7e6, 0.0, 0.0], [7e6, 1e5, 0.0]])
        start = np.array([1.0, 0.0, 0.0, 0.0])
        history = SimulationHistory(
            label="short way", times=times, positions=positions,
            quaternions=np.stack([start, -start]),
        )
        midpoint = history.sample(0.5)["quaternion"]
        assert abs(float(np.dot(midpoint, start))) == pytest.approx(1.0, abs=1e-9)

    def test_validation_rejects_mismatched_and_non_monotone_input(self):
        times = np.array([0.0, 1.0, 2.0])
        good = np.zeros((3, 3))
        with pytest.raises(ValueError, match="strictly increasing"):
            SimulationHistory(label="x", times=np.array([0.0, 0.0, 1.0]), positions=good)
        with pytest.raises(ValueError, match="positions must have shape"):
            SimulationHistory(label="x", times=times, positions=np.zeros((2, 3)))
        with pytest.raises(ValueError, match="quaternions must have shape"):
            SimulationHistory(
                label="x", times=times, positions=good, quaternions=np.zeros((3, 3))
            )
        with pytest.raises(ValueError, match="at least two samples"):
            SimulationHistory(label="x", times=np.array([0.0]), positions=np.zeros((1, 3)))
