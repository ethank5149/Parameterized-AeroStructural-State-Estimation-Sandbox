"""Ellipsoid geodesy, terrain, imagery and pacing.

The first three are checked against things outside this package — the WGS84
defining constants, published summit and shoreline elevations, and the
Blue Marble tiles' own georeferencing. The fourth is checked against the
property it exists to provide: that a launch and a terminal descent get more
of the frames than a coast does, on the actual trajectories this framework
produces.
"""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

import numpy as np
import pytest

from passes.geodesy import WGS84_MEAN_RADIUS, GeodeticPosition
from passes.orbital.scenario import ballistic_trajectory, fobs_trajectory
from passes.viz.ellipsoid import (
    WGS84,
    Ellipsoid,
    ecef_to_geodetic,
    geodetic_to_ecef,
    local_vertical,
    ray_ellipsoid,
)
from passes.viz.globe import Camera, render
from passes.viz.history import SimulationHistory
from passes.viz.imagery import Texture, default_blue_marble
from passes.viz.pacing import (
    PacingProfile,
    PacingWeights,
    attention_density,
    uniform_pacing,
)
from passes.viz.terrain import ReliefMap, default_terrain

REFERENCE = Path(__file__).resolve().parents[1] / "reference"
HAVE_TERRAIN = (REFERENCE / "GMTED2010").is_dir()
HAVE_IMAGERY = (REFERENCE / "blue-marble-next-gen").is_dir()

LAUNCH = GeodeticPosition.from_degrees(51.0, 59.0, 120.0, "Dombarovskiy")
TARGET = GeodeticPosition.from_degrees(38.87, -77.06, 24.0, "aimpoint")


class TestEllipsoid:
    def test_wgs84_defining_constants(self) -> None:
        assert WGS84.semi_major == 6378137.0
        assert WGS84.flattening == pytest.approx(1.0 / 298.257223563)
        assert WGS84.semi_minor == pytest.approx(6356752.314245, abs=1e-6)
        assert WGS84.eccentricity_squared == pytest.approx(0.00669437999014, abs=1e-14)
        assert WGS84.mean_radius == pytest.approx(6371008.771, abs=1e-3)

    def test_polar_flattening_is_21_km(self) -> None:
        assert WGS84.semi_major - WGS84.semi_minor == pytest.approx(21384.7, abs=0.1)

    def test_round_trip_is_machine_precision(self) -> None:
        rng = np.random.default_rng(0)
        latitude = rng.uniform(-0.4999 * np.pi, 0.4999 * np.pi, 5000)
        longitude = rng.uniform(-np.pi, np.pi, 5000)
        altitude = rng.uniform(-500.0, 400e3, 5000)
        back = ecef_to_geodetic(geodetic_to_ecef(latitude, longitude, altitude))
        assert back[0] == pytest.approx(latitude, abs=1e-12)
        assert back[2] == pytest.approx(altitude, abs=1e-6)

    def test_round_trip_at_orbital_altitude(self) -> None:
        """Bowring alone degrades with height; the Newton step is why this holds."""
        latitude, longitude, altitude = 0.7, -1.3, 35_786_000.0
        back = ecef_to_geodetic(geodetic_to_ecef(latitude, longitude, altitude))
        assert float(back[0]) == pytest.approx(latitude, abs=1e-12)
        assert float(back[2]) == pytest.approx(altitude, rel=1e-12)

    def test_poles_are_exact(self) -> None:
        for sign in (1.0, -1.0):
            point = np.array([0.0, 0.0, sign * (WGS84.semi_minor + 1000.0)])
            latitude, _, altitude = ecef_to_geodetic(point)
            assert float(latitude) == pytest.approx(sign * 0.5 * np.pi)
            assert float(altitude) == pytest.approx(1000.0, abs=1e-6)

    def test_equator_and_pole_radii(self) -> None:
        assert float(WGS84.surface_radius(0.0)) == pytest.approx(WGS84.semi_major)
        assert float(WGS84.surface_radius(0.5 * np.pi)) == pytest.approx(
            WGS84.semi_minor, abs=1e-6
        )

    def test_geocentric_latitude_differs_by_a_fifth_of_a_degree(self) -> None:
        difference = 0.25 * np.pi - float(WGS84.geocentric_latitude(0.25 * np.pi))
        assert np.rad2deg(difference) == pytest.approx(0.1924, abs=1e-3)

    def test_treating_geodetic_as_geocentric_displaces_21_km(self) -> None:
        """The error this module exists to remove, measured."""
        true_point = geodetic_to_ecef(0.25 * np.pi, 0.0, 0.0)
        radius = float(np.linalg.norm(true_point))
        spherical = radius * np.array([np.cos(0.25 * np.pi), 0.0, np.sin(0.25 * np.pi)])
        assert float(np.linalg.norm(true_point - spherical)) == pytest.approx(
            21385.0, abs=50.0
        )

    def test_local_vertical_is_not_radial(self) -> None:
        point = geodetic_to_ecef(0.25 * np.pi, 0.0, 0.0)
        radial = point / np.linalg.norm(point)
        normal = local_vertical(0.25 * np.pi, 0.0)
        angle = np.arccos(np.clip(float(normal @ radial), -1.0, 1.0))
        assert np.rad2deg(angle) == pytest.approx(0.1924, abs=1e-3)

    def test_local_vertical_is_radial_at_the_equator_and_poles(self) -> None:
        for latitude in (0.0, 0.5 * np.pi, -0.5 * np.pi):
            point = geodetic_to_ecef(latitude, 0.3, 0.0)
            radial = point / np.linalg.norm(point)
            normal = local_vertical(latitude, 0.3)
            assert float(normal @ radial) == pytest.approx(1.0, abs=1e-12)

    def test_ray_hits_the_semi_axes(self) -> None:
        polar, _ = ray_ellipsoid(np.array([0.0, 0.0, 3e7]), np.array([[[0.0, 0.0, -1.0]]]))
        assert 3e7 - float(polar[0, 0]) == pytest.approx(WGS84.semi_minor, abs=1e-6)
        equatorial, _ = ray_ellipsoid(
            np.array([3e7, 0.0, 0.0]), np.array([[[-1.0, 0.0, 0.0]]])
        )
        assert 3e7 - float(equatorial[0, 0]) == pytest.approx(
            WGS84.semi_major, abs=1e-6
        )

    def test_ray_misses_are_infinite(self) -> None:
        distance, hit = ray_ellipsoid(
            np.array([3e7, 0.0, 0.0]), np.array([[[0.0, 0.0, 1.0]]])
        )
        assert not bool(hit[0, 0])
        assert not np.isfinite(float(distance[0, 0]))

    def test_ray_hit_lies_on_the_surface(self) -> None:
        rng = np.random.default_rng(3)
        origin = np.array([2.0e7, 1.0e7, 5.0e6])
        directions = rng.normal(size=(40, 3))
        directions /= np.linalg.norm(directions, axis=-1, keepdims=True)
        distance, hit = ray_ellipsoid(origin, directions)
        points = origin + distance[hit][:, None] * directions[hit]
        residual = np.sum((points / WGS84.axes) ** 2, axis=-1)
        assert residual == pytest.approx(np.ones_like(residual), abs=1e-9)

    def test_a_sphere_is_the_zero_flattening_case(self) -> None:
        sphere = Ellipsoid(semi_major=1000.0, flattening=0.0)
        assert sphere.semi_minor == 1000.0
        assert float(sphere.surface_radius(0.7)) == pytest.approx(1000.0)
        assert float(sphere.geocentric_latitude(0.7)) == pytest.approx(0.7)

    def test_rejects_bad_parameters(self) -> None:
        with pytest.raises(ValueError, match="semi_major"):
            Ellipsoid(semi_major=-1.0, flattening=0.0)
        with pytest.raises(ValueError, match="flattening"):
            Ellipsoid(semi_major=1.0, flattening=1.5)

    def test_rejects_a_bad_position_shape(self) -> None:
        with pytest.raises(ValueError, match="trailing axis of 3"):
            ecef_to_geodetic(np.zeros((4, 2)))


@pytest.mark.skipif(not HAVE_TERRAIN, reason="GMTED2010 archive not present")
class TestTerrain:
    #: (name, latitude, longitude, expected metres, tolerance)
    KNOWN: ClassVar[list[tuple[str, float, float, float, float]]] = [
        ("Dead Sea shore", 31.5, 35.5, -412.0, 60.0),
        ("Denver", 39.74, -104.99, 1605.0, 120.0),
        ("open Pacific", 0.0, -140.0, 0.0, 1.0),
    ]

    @staticmethod
    def terrain(product: str = "mea"):  # type: ignore[no-untyped-def]
        from passes.viz.terrain import default_terrain

        return default_terrain(product=product)

    def test_archive_is_indexed(self) -> None:
        model = self.terrain()
        assert model.n_tiles >= 96
        south, north = model.coverage
        assert south <= -70.0 and north >= 84.0

    @pytest.mark.parametrize(("name", "lat", "lon", "expected", "tolerance"), KNOWN)
    def test_known_elevations(
        self, name: str, lat: float, lon: float, expected: float, tolerance: float
    ) -> None:
        sample = self.terrain().elevation(lat, lon)
        assert float(sample.elevation) == pytest.approx(expected, abs=tolerance)

    def test_everest_is_high_but_not_the_summit(self) -> None:
        """A 7.5-arc-second cell cannot resolve a summit narrower than itself.

        The measured spread between products is 75 m and the gap to the true
        8848 m summit is 180 m, so the limit is cell size and not the choice
        of statistic. Asserted so that a future switch to a finer source
        shows up as a failure rather than as a silent improvement nobody
        checked.
        """
        heights = {
            product: float(self.terrain(product).elevation(27.9881, 86.9250).elevation)
            for product in ("min", "mea", "max")
        }
        assert heights["min"] < heights["mea"] < heights["max"]
        assert 8600.0 < heights["mea"] < 8750.0
        assert heights["max"] < 8848.0

    def test_radians_and_degrees_agree(self) -> None:
        model = self.terrain()
        degrees = model.elevation(39.74, -104.99).elevation
        radians = model.elevation(
            np.deg2rad(39.74), np.deg2rad(-104.99), degrees=False
        ).elevation
        assert float(radians) == pytest.approx(float(degrees))

    def test_vectorised_query_keeps_its_shape(self) -> None:
        latitudes = np.array([[31.5, 39.74], [0.0, 51.09]])
        longitudes = np.array([[35.5, -104.99], [-140.0, 59.84]])
        sample = self.terrain().elevation(latitudes, longitudes)
        assert sample.elevation.shape == (2, 2)
        assert sample.void.shape == (2, 2)

    def test_rejects_an_unknown_product(self) -> None:
        from passes.viz.terrain import Terrain

        with pytest.raises(ValueError, match="product must be one of"):
            Terrain(REFERENCE / "GMTED2010", product="mean")


@pytest.mark.skipif(not HAVE_IMAGERY, reason="Blue Marble archive not present")
class TestImagery:
    @staticmethod
    def archive():  # type: ignore[no-untyped-def]
        from passes.viz.imagery import default_blue_marble

        return default_blue_marble()

    def test_all_twelve_months_are_present(self) -> None:
        assert self.archive().months == tuple(range(1, 13))

    def test_every_month_has_eight_tiles(self) -> None:
        archive = self.archive()
        for month in archive.months:
            assert sorted(archive.tiles(month)) == [
                "A1", "A2", "B1", "B2", "C1", "C2", "D1", "D2"
            ]

    def test_month_resolution(self) -> None:
        archive = self.archive()
        assert archive.month_of("2015-01-13T06:00") == 1
        assert archive.month_of("2015-08-01") == 8
        assert archive.month_of("august") == 8
        assert archive.month_of(7) == 7

    def test_month_number_is_validated(self) -> None:
        with pytest.raises(ValueError, match="month must be 1-12"):
            self.archive().month_of(13)

    def test_mosaic_geometry_and_landmarks(self) -> None:
        """Row 0 is +90 and column 0 is -180 — the convention render samples."""
        image = self.archive().mosaic(height=1024, month=1)
        assert image.shape == (1024, 2048, 3)
        assert image.dtype == np.uint8

        def at(latitude: float, longitude: float) -> np.ndarray:
            rows, cols = image.shape[0], image.shape[1]
            row = int((90.0 - latitude) / 180.0 * (rows - 1))
            column = int((longitude + 180.0) / 360.0 * cols) % cols
            return np.asarray(image[row, column], dtype=np.float64)

        ocean = at(0.0, -140.0)
        sahara = at(25.0, 10.0)
        greenland = at(72.0, -40.0)
        # Ocean is dark and blue-dominant; desert is bright and red-dominant;
        # ice is bright and nearly neutral. Any north-south or east-west flip
        # of the mosaic breaks at least one of these.
        assert ocean[2] > ocean[0] and ocean.mean() < 60.0
        assert sahara[0] > sahara[2] and sahara.mean() > 90.0
        assert greenland.mean() > 200.0 and abs(greenland[0] - greenland[2]) < 30.0

    def test_mosaic_height_is_validated(self) -> None:
        with pytest.raises(ValueError, match="even and at least"):
            self.archive().mosaic(height=1025)

    def test_window_rejects_an_antimeridian_crossing(self) -> None:
        with pytest.raises(ValueError, match="antimeridian"):
            self.archive().window((10.0, 20.0), (170.0, -170.0))

    def test_window_rejects_an_empty_latitude_box(self) -> None:
        with pytest.raises(ValueError, match="latitude box"):
            self.archive().window((20.0, 20.0), (10.0, 20.0))

    def test_window_returns_its_actual_bounds(self) -> None:
        """Snapped outward to whole source pixels, and reported as such."""
        crop = self.archive().window((51.0, 51.2), (59.7, 59.95), max_width=512)
        assert crop.data.ndim == 3 and crop.data.shape[2] == 3
        # Outward to whole source pixels, to within the float representation
        # of a degree — the snap is computed in degrees and 59.95 is not one.
        epsilon = 1e-9
        assert crop.south <= 51.0 + epsilon and crop.north >= 51.2 - epsilon
        assert crop.west <= 59.7 + epsilon and crop.east >= 59.95 - epsilon
        # 15 arc-seconds is 1/240 of a degree. This box is 60 source pixels
        # wide against a 512-pixel budget, so the decimation stride is one
        # and the snap cannot exceed a single cell.
        assert crop.north - 51.2 < 1.0 / 240.0
        assert crop.east - 59.95 < 1.0 / 240.0
        assert not crop.wraps

    def test_window_spanning_a_tile_edge_is_assembled_from_both(self) -> None:
        """The tile grid breaks at the equator and every 90 degrees.

        An earlier version took the first intersecting tile and returned a
        crop quietly narrower than the box asked for, which is wrong for
        anything launched from or aimed at a low latitude.
        """
        crop = self.archive().window((-0.4, 0.4), (36.0, 36.8), max_width=512)
        assert crop.south <= -0.4 and crop.north >= 0.4
        rows, _ = crop.shape
        northern = np.asarray(crop.data[: rows // 2], dtype=np.float64)
        southern = np.asarray(crop.data[rows // 2 :], dtype=np.float64)
        # Both halves carry image, so neither tile was dropped. A missing
        # tile would leave its half at exactly zero.
        assert northern.mean() > 1.0
        assert southern.mean() > 1.0

    def test_window_takes_the_month_it_is_asked_for(self) -> None:
        """The crop must follow the same month the mosaic does.

        It did not: the first version indexed the archive by ``months[0]``
        and returned January whatever was asked for, so a close-up could
        show snow the mosaic behind it had melted.
        """
        archive = self.archive()
        if len(archive.months) < 2:  # pragma: no cover - partial archive
            pytest.skip("needs at least two complete months")
        box = ((60.0, 60.3), (30.0, 30.3))
        first = archive.window(*box, month=archive.months[0], max_width=256)
        other = archive.window(*box, month=archive.months[-1], max_width=256)
        assert first.data.shape == other.data.shape
        assert not np.array_equal(first.data, other.data)


class TestPacing:
    @staticmethod
    def fobs() -> SimulationHistory:
        return SimulationHistory.from_trajectory(
            fobs_trajectory(
                LAUNCH, TARGET, parking_altitude=170e3, parking_apogee=250e3,
                samples=600,
            ),
            WGS84_MEAN_RADIUS,
        )

    @staticmethod
    def ballistic() -> SimulationHistory:
        return SimulationHistory.from_trajectory(
            ballistic_trajectory(LAUNCH, TARGET, samples=400), WGS84_MEAN_RADIUS
        )

    def test_grid_endpoints_are_exact(self) -> None:
        history = self.fobs()
        grid = attention_density(history, WGS84_MEAN_RADIUS).grid(240)
        assert grid[0] == history.times[0]
        assert grid[-1] == history.times[-1]

    def test_grid_is_strictly_increasing(self) -> None:
        grid = attention_density(self.fobs(), WGS84_MEAN_RADIUS).grid(500)
        assert np.all(np.diff(grid) > 0.0)

    def test_uniform_pacing_is_a_linear_grid(self) -> None:
        history = self.ballistic()
        grid = uniform_pacing(history).grid(101)
        assert grid == pytest.approx(
            np.linspace(history.times[0], history.times[-1], 101), abs=1e-6
        )

    def test_the_launch_gets_more_frames_than_its_share_of_the_clock(self) -> None:
        """The complaint this module answers: the ascent was five frames."""
        history = self.fobs()
        paced = attention_density(history, WGS84_MEAN_RADIUS)
        boost = next(p for p in history.phases if p.name == "boost")
        share = paced.share(boost.start_time, boost.end_time)
        clock = boost.duration / history.duration
        assert share > 3.0 * clock

    def test_the_coast_is_compressed_but_not_deleted(self) -> None:
        history = self.fobs()
        paced = attention_density(history, WGS84_MEAN_RADIUS)
        coast = next(p for p in history.phases if "parking" in p.name)
        share = paced.share(coast.start_time, coast.end_time)
        clock = coast.duration / history.duration
        assert share < 0.75 * clock
        assert share > 0.15

    def test_the_terminal_minute_gets_more_frames(self) -> None:
        for history in (self.fobs(), self.ballistic()):
            paced = attention_density(history, WGS84_MEAN_RADIUS)
            plain = uniform_pacing(history)
            final = (history.times[-1] - 60.0, history.times[-1])
            assert paced.share(*final) > 2.0 * plain.share(*final)

    def test_the_first_minute_gets_more_frames(self) -> None:
        for history in (self.fobs(), self.ballistic()):
            paced = attention_density(history, WGS84_MEAN_RADIUS)
            plain = uniform_pacing(history)
            opening = (history.times[0], history.times[0] + 60.0)
            assert paced.share(*opening) > 2.0 * plain.share(*opening)

    def test_shares_sum_to_one_over_the_phases(self) -> None:
        history = self.fobs()
        paced = attention_density(history, WGS84_MEAN_RADIUS)
        total = sum(paced.share(p.start_time, p.end_time) for p in history.phases)
        assert total == pytest.approx(1.0, abs=1e-9)

    def test_manoeuvre_is_declared_unavailable_not_approximated(self) -> None:
        """A scenario trajectory carries no velocity, so specific force is not
        double-differenced out of positions — that measured 0.85 m/s^2 of
        noise on a Keplerian coast where the true value is zero."""
        history = self.fobs()
        assert history.velocities is None
        paced = attention_density(history, WGS84_MEAN_RADIUS)
        assert np.all(paced.terms["manoeuvre"] == 0.0)

    def test_playback_rate_is_slower_where_the_density_is_higher(self) -> None:
        history = self.fobs()
        paced = attention_density(history, WGS84_MEAN_RADIUS)
        boost = next(p for p in history.phases if p.name == "boost")
        coast = next(p for p in history.phases if "parking" in p.name)
        during_boost = paced.rate(0.5 * (boost.start_time + boost.end_time), 600, 30)
        during_coast = paced.rate(0.5 * (coast.start_time + coast.end_time), 600, 30)
        assert during_boost < during_coast

    def test_density_must_be_positive(self) -> None:
        with pytest.raises(ValueError, match="strictly positive"):
            PacingProfile(np.array([0.0, 1.0, 2.0]), np.array([1.0, 0.0, 1.0]))

    def test_times_must_increase(self) -> None:
        with pytest.raises(ValueError, match="strictly increasing"):
            PacingProfile(np.array([0.0, 2.0, 1.0]), np.ones(3))

    def test_needs_two_frames(self) -> None:
        with pytest.raises(ValueError, match="at least two frames"):
            uniform_pacing(self.ballistic()).grid(1)

    def test_weights_are_validated(self) -> None:
        with pytest.raises(ValueError, match="floor"):
            PacingWeights(floor=0.0)
        with pytest.raises(ValueError, match="climb weight"):
            PacingWeights(climb=-1.0)
        with pytest.raises(ValueError, match="ceiling"):
            PacingWeights(ceiling=0.5)
        with pytest.raises(ValueError, match="compression"):
            PacingWeights(compression=0.0)


@pytest.mark.skipif(not HAVE_TERRAIN, reason="GMTED2010 archive not present")
class TestRelief:
    """The slopes the globe is shaded and displaced with."""

    @staticmethod
    def relief() -> ReliefMap:
        return default_terrain().relief()

    def test_slopes_are_physical_everywhere_including_the_poles(self) -> None:
        """The defect the widening east stencil exists to fix.

        Differencing over one cell at the top row divides a real elevation
        step by a 7.5 m baseline, and that returned a **slope of 186** — an
        89.7 degree cliff running round the pole. No terrain on Earth
        exceeds about 45 degrees averaged over a 10 km cell.
        """
        relief = self.relief()
        magnitude = np.hypot(relief.slope_east, relief.slope_north)
        assert float(magnitude.max()) < 1.0
        # And the steepest ground is somewhere plausible rather than at a
        # pole: the Antarctic and Andean scarps, not row zero.
        rows = relief.shape[0]
        steepest = int(np.argmax(magnitude) // relief.shape[1])
        latitude = 90.0 - (steepest + 0.5) * 180.0 / rows
        assert -85.0 < latitude < 85.0

    def test_slope_is_rise_over_run_in_metres(self) -> None:
        """A synthetic ramp of known gradient, so the metric is checked.

        A degree of longitude is 111 km at the equator and 19 km at 80
        north; dividing by degrees instead of metres would make every
        high-latitude hill a cliff.
        """
        rows, cols = 180, 360
        # 1000 m rise across the whole equatorial circumference.
        grid = np.tile(np.linspace(0.0, 1000.0, cols, dtype=np.float32), (rows, 1))
        relief = ReliefMap.from_grid(grid)
        equator = rows // 2
        expected = 1000.0 / (2.0 * np.pi * 6378137.0)
        middle = relief.slope_east[equator, cols // 2]
        assert float(middle) == pytest.approx(expected, rel=0.02)

    def test_exaggeration_scales_the_slopes_and_says_so(self) -> None:
        grid = np.tile(np.linspace(0.0, 1000.0, 360, dtype=np.float32), (180, 1))
        plain = ReliefMap.from_grid(grid)
        doubled = ReliefMap.from_grid(grid, exaggeration=2.0)
        assert doubled.exaggeration == 2.0
        assert np.allclose(doubled.slope_east, 2.0 * plain.slope_east)
        # Elevation itself is untouched: the exaggeration is a shading
        # choice and must not leak into a height anyone reads.
        assert np.array_equal(doubled.elevation, plain.elevation)

    def test_longitude_wraps_and_latitude_does_not(self) -> None:
        """A one-sided difference at the antimeridian would draw a
        meridian-long false scarp down the Pacific."""
        grid = np.zeros((180, 360), dtype=np.float32)
        grid[:, 0] = 1000.0
        relief = ReliefMap.from_grid(grid)
        # The step is seen from *both* sides of the seam.
        assert float(relief.slope_east[90, 1]) < 0.0
        assert float(relief.slope_east[90, -1]) > 0.0

    def test_mismatched_grids_are_refused(self) -> None:
        with pytest.raises(ValueError, match="must match the elevation grid"):
            ReliefMap(np.zeros((4, 8)), np.zeros((4, 8)), np.zeros((2, 2)))


@pytest.mark.skipif(
    not (HAVE_TERRAIN and HAVE_IMAGERY), reason="reference archives not present"
)
class TestRendererWiring:
    """What a rendered frame actually gets from the four data modules."""

    @staticmethod
    def imagery() -> Texture:
        return default_blue_marble().texture(height=1024, month=1)

    @staticmethod
    def overhead(latitude: float, longitude: float, altitude: float) -> Camera:
        point = geodetic_to_ecef(
            np.deg2rad(latitude), np.deg2rad(longitude), 0.0, WGS84
        )
        up = local_vertical(np.deg2rad(latitude), np.deg2rad(longitude))
        return Camera(
            position=point + altitude * up,
            target=point,
            up=np.array([0.0, 0.0, 1.0]),
            fov=np.deg2rad(30.0),
            width=180,
            height=180,
        )

    @staticmethod
    def light(latitude: float, longitude: float, elevation_deg: float) -> np.ndarray:
        up = local_vertical(np.deg2rad(latitude), np.deg2rad(longitude))
        east = np.array([-np.sin(np.deg2rad(longitude)), np.cos(np.deg2rad(longitude)), 0.0])
        angle = np.deg2rad(elevation_deg)
        return np.asarray(np.sin(angle) * up + np.cos(angle) * east)

    def test_relief_shading_needs_a_light_off_the_vertical(self) -> None:
        """Measured, because it decides the animator's key-light default.

        Shading works on the cosine against a tilted normal, so a light
        along the local vertical meets terrain slope at second order and
        does essentially nothing. This is why ``lighting="track"`` no
        longer points the key straight up.
        """
        latitude, longitude = 28.0, 86.9  # Himalaya
        camera = self.overhead(latitude, longitude, 400.0e3)
        relief = default_terrain().relief()
        texture = self.imagery()
        contrast = {}
        for elevation_deg in (90.0, 55.0):
            sun = self.light(latitude, longitude, elevation_deg)
            smooth, _ = render(
                camera, texture, WGS84, sun=sun, atmosphere=0.0, specular=0.0
            )
            shaded, _ = render(
                camera, texture, WGS84, sun=sun, relief=relief,
                atmosphere=0.0, specular=0.0,
            )
            contrast[elevation_deg] = float(np.abs(shaded - smooth).max())
        assert contrast[55.0] > 5.0 * contrast[90.0]
        assert contrast[55.0] > 0.01

    def test_relief_shading_does_nothing_over_the_ocean(self) -> None:
        """Flat water has no slope, so the term must be identically zero
        there — a shading model that tinted the sea would be decoration."""
        latitude, longitude = 0.0, -140.0
        camera = self.overhead(latitude, longitude, 400.0e3)
        sun = self.light(latitude, longitude, 55.0)
        texture = self.imagery()
        smooth, _ = render(camera, texture, WGS84, sun=sun, atmosphere=0.0, specular=0.0)
        shaded, _ = render(
            camera, texture, WGS84, sun=sun, relief=default_terrain().relief(),
            atmosphere=0.0, specular=0.0,
        )
        assert np.allclose(shaded, smooth, atol=1e-12)

    def test_displacement_moves_the_ground_by_the_terrain_height(self) -> None:
        """The depth buffer must shorten by the elevation under the ray.

        Looking straight down, the distance to the ground is the camera's
        height less the terrain's, so this is a direct comparison against
        :meth:`Terrain.elevation` rather than against a picture.
        """
        latitude, longitude, height = 28.0, 86.9, 60.0e3
        camera = self.overhead(latitude, longitude, height)
        relief = default_terrain().relief()
        sun = self.light(latitude, longitude, 55.0)
        _, smooth = render(
            camera, self.imagery(), WGS84, sun=sun, relief=relief,
            atmosphere=0.0, specular=0.0,
        )
        _, moved = render(
            camera, self.imagery(), WGS84, sun=sun, relief=relief, displace=True,
            atmosphere=0.0, specular=0.0,
        )
        centre = (camera.height // 2, camera.width // 2)
        lifted = float(smooth[centre] - moved[centre])
        # Against the same coarse grid the march reads, not against the
        # 7.5-arc-second archive: a 9.8 km cell does not carry a summit.
        rows, cols = relief.shape
        row = int((90.0 - latitude) / (180.0 / rows))
        column = int((longitude + 180.0) / (360.0 / cols))
        expected = float(relief.elevation[row, column])
        assert lifted == pytest.approx(expected, rel=0.05, abs=100.0)
        assert lifted > 4000.0, "the Himalaya must lift the ground kilometres"

    def test_displacement_needs_a_relief_map(self) -> None:
        with pytest.raises(ValueError, match="needs a relief map"):
            render(
                self.overhead(0.0, 0.0, 500.0e3), self.imagery(), WGS84, displace=True
            )

    def test_a_native_resolution_crop_composites_over_the_mosaic(self) -> None:
        """The close-up path: the frame must change, and change *sharply*.

        At a 20 km stand-off the global mosaic shows about one texel per
        forty pixels, so the crop is not a refinement of the picture — it
        is the difference between a colour field and a coastline.
        """
        latitude, longitude = 51.09, 59.84  # the bundled launch site
        camera = self.overhead(latitude, longitude, 20.0e3)
        mosaic = self.imagery()
        crop = default_blue_marble().window(
            (latitude - 0.5, latitude + 0.5), (longitude - 0.5, longitude + 0.5),
            month=1, max_width=1024,
        )
        sun = self.light(latitude, longitude, 55.0)
        coarse, _ = render(camera, mosaic, WGS84, sun=sun, atmosphere=0.0, specular=0.0)
        fine, _ = render(
            camera, [mosaic, crop], WGS84, sun=sun, atmosphere=0.0, specular=0.0
        )
        assert crop.ground_resolution < 0.1 * mosaic.ground_resolution
        # Local contrast: the standard deviation of neighbouring-pixel
        # differences, which is what "detail" means and what a smooth
        # gradient does not have.
        def texture_energy(image: np.ndarray) -> float:
            return float(np.std(np.diff(image[..., 0], axis=1)))

        assert texture_energy(fine) > 3.0 * texture_energy(coarse)

    def test_outside_its_box_a_crop_changes_nothing(self) -> None:
        """The composite is bounded by the crop's own footprint, so a
        close-up over the pad cannot alter the far side of the planet."""
        camera = self.overhead(-20.0, -100.0, 2.0e6)
        mosaic = self.imagery()
        crop = default_blue_marble().window((51.0, 51.5), (59.5, 60.0), month=1,
                                            max_width=256)
        sun = self.light(-20.0, -100.0, 55.0)
        plain, _ = render(camera, mosaic, WGS84, sun=sun, atmosphere=0.0, specular=0.0)
        stacked, _ = render(
            camera, [mosaic, crop], WGS84, sun=sun, atmosphere=0.0, specular=0.0
        )
        assert np.array_equal(plain, stacked)
