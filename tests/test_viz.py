"""Ray-traced globe rendering (presentation layer)."""

from pathlib import Path

import numpy as np
import pytest

from passes.batch.backend import cuda_available
from passes.dynamics.attitude import dcm_from_quaternion
from passes.geodesy import WGS84_MEAN_RADIUS, GeodeticPosition
from passes.viz import (
    DEFAULT_TEXTURE,
    NOSE_AXIS,
    Camera,
    ChaseRig,
    SimulationHistory,
    TrajectoryAnimator,
    ease,
    geodetic_to_cartesian,
    glyph_polylines,
    glyph_world,
    horizon_ring,
    load_texture,
    look_at,
    project,
    render,
    site_status,
    starfield,
    sun_direction,
    to_device,
    video_writer,
)

_R = WGS84_MEAN_RADIUS

pytest.importorskip("matplotlib")
import matplotlib  # noqa: E402

matplotlib.use("Agg")


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


class TestBackend:
    """The GPU path must be the same renderer, not a second one."""

    @pytest.mark.skipif(not cuda_available(), reason="no CUDA device")
    def test_the_cupy_path_reproduces_the_numpy_path(self):
        """A backend that disagreed with the CPU would make every frame a
        function of which machine drew it. Not bit-identical — the reduction
        order in einsum differs — but far inside anything visible: 1e-11 in
        a colour channel is 3e-9 of one 8-bit level."""
        texture = _checker(256, 512)
        camera = look_at(np.zeros(3), 4.0 * _R, 0.4, 0.3, width=160, height=120)
        sun = sun_direction(np.deg2rad(40.0), np.deg2rad(10.0))
        host, host_depth = render(camera, texture, _R, sun=sun)
        device, device_depth = render(
            camera, to_device(texture, "cupy"), _R, sun=sun, backend="cupy"
        )
        assert float(np.max(np.abs(host - device))) < 1e-9
        finite = np.isfinite(host_depth)
        assert np.array_equal(finite, np.isfinite(device_depth))
        assert float(np.max(np.abs(host_depth[finite] - device_depth[finite]))) < 1e-3

    def test_a_host_texture_on_a_device_backend_is_refused(self):
        """Silently uploading per frame would cost more than the render it
        was meant to accelerate, so it is an error rather than a courtesy."""
        camera = look_at(np.zeros(3), 4.0 * _R, 0.0, 0.0, width=32, height=32)
        with pytest.raises(TypeError, match="device-resident texture"):
            render(camera, _checker(), _R, backend="cupy")


class TestSceneGeometry:
    def test_geodetic_to_cartesian_round_trips(self):
        for lat_deg, lon_deg in ((0.0, 0.0), (51.0, 59.0), (-33.9, 18.4), (78.0, -68.0)):
            position = GeodeticPosition.from_degrees(lat_deg, lon_deg)
            xyz = geodetic_to_cartesian(position, _R)
            assert float(np.linalg.norm(xyz)) == pytest.approx(_R, rel=1e-12)
            assert np.rad2deg(np.arcsin(xyz[2] / _R)) == pytest.approx(lat_deg, abs=1e-9)
            assert np.rad2deg(np.arctan2(xyz[1], xyz[0])) == pytest.approx(lon_deg, abs=1e-9)

    def test_an_iterable_gives_one_row_per_point(self):
        points = [GeodeticPosition.from_degrees(a, b) for a, b in ((0, 0), (10, 20), (-5, 90))]
        assert geodetic_to_cartesian(points, _R).shape == (3, 3)
        assert geodetic_to_cartesian(points[0], _R).shape == (3,)

    def test_lift_raises_the_marker_clear_of_the_surface(self):
        position = GeodeticPosition.from_degrees(40.0, -75.0)
        plain = geodetic_to_cartesian(position, _R)
        lifted = geodetic_to_cartesian(position, _R, lift=15.0e3)
        assert float(np.linalg.norm(lifted) - np.linalg.norm(plain)) == pytest.approx(15e3)

    def test_ease_is_a_clamped_smoothstep(self):
        assert float(ease(0.0)) == 0.0
        assert float(ease(1.0)) == 1.0
        assert float(ease(0.5)) == pytest.approx(0.5)
        assert float(ease(-3.0)) == 0.0 and float(ease(7.0)) == 1.0
        # zero slope at both ends is the whole point: linear keyframing
        # shows a visible acceleration discontinuity at every waypoint
        assert float(ease(1e-4)) < 1e-7
        assert 1.0 - float(ease(1.0 - 1e-4)) < 1e-7

    def test_starfield_is_deterministic_cached_and_immutable(self):
        first = starfield(64, 48)
        assert first is starfield(64, 48), "identical requests must reuse the cache"
        assert not first.flags.writeable, "a shared array must not be mutable"
        assert np.array_equal(first, starfield(64, 48, seed=7))
        assert not np.array_equal(first, starfield(64, 48, seed=8))


class TestChaseRig:
    _POSITION = np.array([_R + 150e3, 0.0, 0.0])

    def test_the_eye_sits_behind_the_vehicle_along_its_velocity(self):
        velocity = np.array([0.0, 7800.0, 0.0])
        camera = ChaseRig().camera(self._POSITION, velocity, 640, 360)
        offset = self._POSITION - camera.position
        assert float(offset @ velocity) > 0.0, "the eye must trail the vehicle"
        # and the aim point leads it, so the vehicle is not pinned dead centre
        assert float((camera.target - self._POSITION) @ velocity) > 0.0

    def test_a_radial_boost_velocity_does_not_bury_the_camera(self):
        """The regression for a real defect. During boost the velocity is
        steeply radial, so stepping *back* along it also steps *down*: the
        eye ended up 96 km underground at lift-off and the ray tracer
        returned an empty frame — a black screen with a floating trajectory
        and no Earth at all."""
        surface = np.array([_R, 0.0, 0.0])
        straight_up = np.array([300.0, 0.0, 0.0])
        rig = ChaseRig()
        camera = rig.camera(surface, straight_up, 640, 360)
        eye_altitude = float(np.linalg.norm(camera.position)) - _R
        assert eye_altitude >= rig.floor - 1.0

    def test_the_standoff_scales_with_altitude(self):
        """One rig has to frame a 150 km parking arc and a 1300 km apogee.
        A fixed stand-off makes one of the two a line in the corner."""
        rig = ChaseRig(tighten=0.0)
        velocity = np.array([0.0, 7000.0, 0.0])
        low = rig.camera(np.array([_R + 150e3, 0.0, 0.0]), velocity, 640, 360)
        high = rig.camera(np.array([_R + 1300e3, 0.0, 0.0]), velocity, 640, 360)
        low_range = float(np.linalg.norm(low.position - np.array([_R + 150e3, 0.0, 0.0])))
        high_range = float(np.linalg.norm(high.position - np.array([_R + 1300e3, 0.0, 0.0])))
        assert high_range > 2.0 * low_range

    def test_tightening_closes_the_camera_in_over_the_run(self):
        rig = ChaseRig(tighten=0.25)
        velocity = np.array([0.0, 7000.0, 0.0])
        start = rig.camera(self._POSITION, velocity, 640, 360, progress=0.0)
        end = rig.camera(self._POSITION, velocity, 640, 360, progress=1.0)
        assert float(np.linalg.norm(end.position - self._POSITION)) < float(
            np.linalg.norm(start.position - self._POSITION)
        )

    def test_a_stationary_sample_still_yields_a_finite_camera(self):
        camera = ChaseRig().camera(self._POSITION, np.zeros(3), 320, 240)
        assert np.all(np.isfinite(camera.position))
        assert np.all(np.isfinite(np.concatenate(camera.basis())))

    def test_a_vehicle_at_the_body_centre_is_refused(self):
        with pytest.raises(ValueError, match="body centre"):
            ChaseRig().camera(np.zeros(3), np.array([1.0, 0.0, 0.0]), 64, 64)


class TestVehicleGlyph:
    def test_the_nose_points_where_the_attitude_says(self):
        """The claim the glyph makes. For a quarter turn about +z the body
        nose must appear along inertial +y, and it must be the *transpose*
        of the DCM that takes it there — C_E^B maps inertial to body."""
        quaternion = np.array([np.cos(np.pi / 4), 0.0, 0.0, np.sin(np.pi / 4)])
        dcm = dcm_from_quaternion(quaternion)
        centre = np.array([_R + 100e3, 0.0, 0.0])
        nose = glyph_world(centre, dcm, scale=1000.0)[1][0]
        assert np.allclose(nose - centre, 1000.0 * np.array([0.0, 1.0, 0.0]), atol=1e-6)
        assert np.allclose(nose - centre, 1000.0 * (dcm.T @ NOSE_AXIS), atol=1e-9)

    def test_identity_attitude_leaves_the_nose_on_the_body_axis(self):
        centre = np.array([0.0, _R, 0.0])
        nose = glyph_world(centre, np.eye(3), scale=500.0)[1][0]
        assert np.allclose(nose - centre, 500.0 * NOSE_AXIS)

    def test_the_glyph_is_rigid_under_rotation(self):
        """Every pairwise distance must survive the rotation, or the vehicle
        is being sheared by something that is not a proper rotation."""
        rng = np.random.default_rng(3)
        q = rng.normal(size=4)
        dcm = dcm_from_quaternion(q / np.linalg.norm(q))
        centre = np.array([_R + 400e3, 1e5, -2e5])
        plain = np.concatenate(glyph_world(centre, np.eye(3), 700.0))
        turned = np.concatenate(glyph_world(centre, dcm, 700.0))
        for a, b in ((0, 5), (3, 40), (12, 30)):
            assert float(np.linalg.norm(turned[a] - turned[b])) == pytest.approx(
                float(np.linalg.norm(plain[a] - plain[b])), rel=1e-12
            )

    def test_the_fin_count_is_respected(self):
        base = len(glyph_polylines(0))
        assert len(glyph_polylines(4)) == base + 4
        assert len(glyph_polylines(3)) == base + 3
        with pytest.raises(ValueError, match="non-negative"):
            glyph_polylines(-1)

    def test_a_malformed_dcm_is_refused(self):
        with pytest.raises(ValueError, match="3x3"):
            glyph_world(np.zeros(3), np.eye(4), 1.0)


class TestSensorOverlay:
    @staticmethod
    def _site(mask_deg=5.0):
        from passes.orbital.radar import RadarSite

        return RadarSite(
            name="test",
            position=GeodeticPosition.from_degrees(20.0, -30.0),
            mask_elevation=np.deg2rad(mask_deg),
        )

    def test_the_ring_lies_at_the_horizon_central_angle(self):
        """Drawn at the *vehicle's* radius, not on the ground: a footprint
        drawn on the surface is some three times too small at 150 km and
        would understate the sensor by exactly the amount the fractional
        orbital argument turns on."""
        from passes.orbital.warning import horizon_central_angle

        site = self._site()
        altitude = 150.0e3
        ring = horizon_ring(site, altitude, _R, samples=73)
        expected = float(horizon_central_angle(altitude, site.mask_elevation, _R))
        centre = geodetic_to_cartesian(site.position, _R)
        centre = centre / np.linalg.norm(centre)
        for point in ring:
            assert float(np.linalg.norm(point)) == pytest.approx(_R + altitude, rel=1e-12)
            unit = point / np.linalg.norm(point)
            assert float(np.arccos(np.clip(unit @ centre, -1, 1))) == pytest.approx(
                expected, abs=1e-12
            )

    def test_the_ring_closes(self):
        ring = horizon_ring(self._site(), 300e3, _R, samples=91)
        assert np.allclose(ring[0], ring[-1])

    def test_a_higher_vehicle_is_seen_from_further_away(self):
        site = self._site()
        low = horizon_ring(site, 150e3, _R, samples=13)
        high = horizon_ring(site, 1300e3, _R, samples=13)
        centre = geodetic_to_cartesian(site.position, _R)
        centre = centre / np.linalg.norm(centre)

        def angle(ring):
            unit = ring[0] / np.linalg.norm(ring[0])
            return float(np.arccos(np.clip(unit @ centre, -1, 1)))

        assert angle(high) > 2.0 * angle(low)

    def test_site_status_separates_detecting_now_from_has_detected(self):
        """Warning is set by the *first* detection, so a site that has lost
        the vehicle behind its horizon still contributed. Colouring it as
        idle again would misrepresent the model that produced the number."""
        from passes.orbital.radar import CoverageResult
        from passes.orbital.warning import DetectionWindow

        window = DetectionWindow(
            detected=True,
            first_detection_time=100.0,
            warning_time=400.0,
            first_detection_altitude=150e3,
            visible_fraction=0.3,
            last_detection_time=250.0,
        )
        result = CoverageResult({"test": window}, 100.0, 400.0, "test", ("test",))
        assert site_status(result, "test", 50.0) == "idle"
        assert site_status(result, "test", 100.0) == "active"
        assert site_status(result, "test", 200.0) == "active"
        assert site_status(result, "test", 400.0) == "seen"
        assert site_status(result, "missing", 200.0) == "idle"
        assert site_status(None, "test", 200.0) == "idle"


class TestAnimator:
    """The façade a notebook calls. Every assertion is about agreement with
    the history, never about how a frame looks."""

    @staticmethod
    def _history(samples=80):
        times = np.linspace(0.0, 600.0, samples)
        angle = np.linspace(0.0, 0.9, samples)
        radius = _R + 200e3 * np.sin(np.linspace(0.0, np.pi, samples)) + 50e3
        positions = np.stack(
            [radius * np.cos(angle), radius * np.sin(angle), np.zeros(samples)], axis=1
        )
        return SimulationHistory(
            label="synthetic",
            times=times,
            positions=positions,
            extras={"heat": np.linspace(0.0, 1.0, samples)},
        )

    @staticmethod
    def _animator(history, **kwargs):
        kwargs.setdefault("width", 160)
        kwargs.setdefault("height", 120)
        kwargs.setdefault("texture", _checker(64, 128))
        return TrajectoryAnimator(history, **kwargs)

    def test_the_frame_grid_spans_the_whole_history(self):
        """The regression for the defect that truncated every animation.
        ``frame * (len(samples) // n_frames)`` truncates: with 900 samples
        over 130 frames the stride was 6 and the last frame landed on sample
        774 of 899 — 86 % of the flight, still at 130 km, with the whole
        descent missing. A time grid cannot do that."""
        history = self._history(samples=900)
        grid = self._animator(history).times(130)
        assert grid.size == 130
        assert float(grid[0]) == float(history.times[0])
        assert float(grid[-1]) == float(history.times[-1])
        assert np.all(np.diff(grid) > 0.0)

    def test_a_single_frame_sequence_is_refused(self):
        with pytest.raises(ValueError, match="at least two frames"):
            self._animator(self._history()).times(1)

    def test_the_vehicle_is_drawn_where_the_true_state_projects(self):
        """The central claim of the whole layer: the picture agrees with the
        physics. Verified by projecting the history's own state through the
        frame's own camera and comparing with what the frame reports."""
        history = self._history()
        animator = self._animator(history)
        for time in (0.0, 137.0, 600.0):
            frame = animator.frame_at(time)
            truth = history.sample(time)["position"]
            assert np.allclose(frame.position, truth, atol=1e-9)
            px, py, visible = project(truth[None, :], frame.camera, radius=_R)
            assert frame.vehicle_pixel == (float(px[0]), float(py[0]), bool(visible[0]))

    def test_a_flight_result_puts_the_glyph_nose_on_the_integrated_attitude(self):
        """The step-6 claim, against the real coupled simulator rather than
        a synthetic history: the nose is placed by the direction cosine
        matrix of the attitude the run integrated."""
        from passes.flight.simulator import FlightConfiguration, FlightSimulator

        sim = FlightSimulator(FlightConfiguration(n_modes=2))
        y0 = sim.initial_state(altitude=120e3, speed=6500.0, flight_path_angle=np.deg2rad(-8.0))
        y0[sim.layout.angular_rate] = np.array([0.3, 0.1, 0.0])
        history = SimulationHistory.from_flight_result(sim.propagate(y0, 40.0, n_output=21))
        assert history.has_attitude

        frame = self._animator(history).frame_at(20.0)
        dcm = frame.state["dcm"]
        assert np.allclose(dcm @ dcm.T, np.eye(3), atol=1e-12)
        assert float(np.linalg.det(dcm)) == pytest.approx(1.0, abs=1e-12)

        nose = glyph_world(frame.position, dcm, 1000.0)[1][0]
        assert np.allclose(nose - frame.position, 1000.0 * (dcm.T @ NOSE_AXIS), atol=1e-9)
        # ...and that nose is somewhere the camera can actually see
        assert frame.pixel_of(nose)[2]

    def test_a_history_without_attitude_gets_a_marker_and_no_glyph(self):
        """A scenario trajectory is a point mass. Drawing an oriented body
        from an invented identity quaternion would show something the run
        never established, so the two cases must differ in the output and
        not only in intent."""
        from passes.flight.simulator import FlightConfiguration, FlightSimulator

        plain = self._animator(self._history()).frame_at(300.0)
        assert "dcm" not in plain.state
        assert len(plain.axes.collections) >= 1, "expected a scatter marker"

        sim = FlightSimulator(FlightConfiguration(n_modes=2))
        y0 = sim.initial_state(altitude=120e3, speed=6500.0, flight_path_angle=np.deg2rad(-8.0))
        oriented = self._animator(
            SimulationHistory.from_flight_result(sim.propagate(y0, 40.0, n_output=21))
        ).frame_at(20.0)
        assert "dcm" in oriented.state
        # the glyph is polylines, so an attitude-carrying run draws strictly
        # more of them than the same scene without one
        assert len(oriented.axes.lines) > len(plain.axes.lines)

    def test_an_unknown_colour_series_names_what_is_available(self):
        with pytest.raises(KeyError, match="heat"):
            self._animator(self._history(), color_by="stagnation_heat_flux")

    def test_a_known_colour_series_is_accepted_and_reported(self):
        frame = self._animator(self._history(), color_by="heat").frame_at(300.0)
        assert 0.0 <= frame.state["heat"] <= 1.0

    def test_coverage_from_a_different_trajectory_is_refused(self):
        """The one way this layer can lie is by drawing detections computed
        from some other flight, so the clocks are checked against each
        other."""
        from passes.orbital.radar import CoverageResult
        from passes.orbital.warning import DetectionWindow

        window = DetectionWindow(True, 99999.0, 1.0, 1e5, 0.1, 99999.0)
        bogus = CoverageResult({"x": window}, 99999.0, 1.0, "x", ("x",))
        with pytest.raises(ValueError, match="different trajectory"):
            self._animator(self._history(), coverage=bogus)

    def test_frames_past_the_end_hold_the_last_state(self):
        history = self._history()
        frame = self._animator(history).frame_at(1e6)
        assert np.allclose(frame.position, history.positions[-1])
        assert frame.time == pytest.approx(float(history.times[-1]))

    def test_a_degenerate_frame_size_is_refused(self):
        with pytest.raises(ValueError, match="at least 2x2"):
            TrajectoryAnimator(self._history(), texture=_checker(), width=1, height=1)


class TestVideoOutput:
    def test_the_container_is_chosen_from_the_extension(self):
        from matplotlib.animation import FFMpegWriter, PillowWriter

        assert isinstance(video_writer("out.gif", 20), PillowWriter)
        if FFMpegWriter.isAvailable():
            assert isinstance(video_writer("out.mp4", 20), FFMpegWriter)

    def test_an_unknown_extension_says_what_is_supported(self):
        with pytest.raises(ValueError, match="unsupported output extension"):
            video_writer("out.avi", 20)

    @pytest.mark.skipif(
        not __import__("matplotlib.animation", fromlist=["x"]).FFMpegWriter.isAvailable(),
        reason="ffmpeg not available",
    )
    def test_odd_dimensions_are_refused_for_h264(self):
        """yuv420p subsamples chroma 2:1, so an odd dimension is not
        encodable. Better a message than an ffmpeg stack trace."""
        history = TestAnimator._history()
        animator = TrajectoryAnimator(
            history, texture=_checker(64, 128), width=161, height=120
        )
        with pytest.raises(ValueError, match="even frame dimensions"):
            animator.render_sequence("unused.mp4", n_frames=2)

    def test_a_short_sequence_writes_a_readable_file(self, tmp_path):
        history = TestAnimator._history(samples=20)
        animator = TrajectoryAnimator(
            history, texture=_checker(64, 128), width=64, height=64
        )
        out = animator.render_sequence(tmp_path / "run.gif", n_frames=3, fps=5)
        assert out.is_file() and out.stat().st_size > 0


class TestHistoryWindow:
    """Cutting a history to a window, without leaving the physics behind."""

    @staticmethod
    def _history(n=50):
        times = np.linspace(0.0, 100.0, n)
        positions = np.stack(
            [_R + 1e5 * times / 100.0, 1e6 * times / 100.0, np.zeros(n)], axis=1
        )
        return SimulationHistory(
            label="ramp", times=times, positions=positions,
            velocities=np.tile(np.array([1e3, 1e4, 0.0]), (n, 1)),
            quaternions=np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (n, 1)),
            extras={"heat": times},
        )

    def test_the_window_starts_and_ends_exactly_where_asked(self):
        """Snapping to the nearest stored sample would put the end of an
        entry animation short of, or past, the ground."""
        window = self._history().between(17.5, 62.5)
        assert float(window.times[0]) == pytest.approx(17.5)
        assert float(window.times[-1]) == pytest.approx(62.5)
        assert np.all(np.diff(window.times) > 0.0)

    def test_the_window_keeps_everything_the_producer_carried(self):
        window = self._history().between(10.0, 90.0)
        assert window.has_attitude
        assert window.velocities is not None
        assert window.extras is not None and "heat" in window.extras
        assert window.extras["heat"][0] == pytest.approx(10.0)
        assert window.label == "ramp"

    def test_interior_samples_are_preserved_not_resampled(self):
        history = self._history()
        window = history.between(10.0, 90.0)
        interior = history.times[(history.times > 10.0) & (history.times < 90.0)]
        assert np.allclose(window.times[1:-1], interior)

    def test_an_empty_or_inverted_window_is_refused(self):
        with pytest.raises(ValueError, match="stop > start"):
            self._history().between(60.0, 40.0)

    def test_a_window_outside_the_run_is_clamped_to_it(self):
        history = self._history()
        window = history.between(-500.0, 5000.0)
        assert float(window.times[0]) == pytest.approx(float(history.times[0]))
        assert float(window.times[-1]) == pytest.approx(float(history.times[-1]))
