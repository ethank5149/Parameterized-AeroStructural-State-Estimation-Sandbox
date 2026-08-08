"""GMTED2010 elevation, so the ground is where the ground is.

Two things in this framework quietly assumed the Earth's surface is at zero
altitude: the impact point of a ballistic arc, and the launch site it left
from. Neither is true anywhere interesting. The Dombarovsky pad coordinates
come back at **348 m** from this archive, and a trajectory terminated at the
ellipsoid rather than at the terrain arrives late, low and slightly
downrange of where it actually would.

``reference/GMTED2010`` is the USGS/NGA global elevation model: 96 tiles at
**7.5 arc-seconds** — about 230 m at the equator — covering 70 S to 90 N,
plus twelve 30-arc-second tiles for Antarctica. Each 7.5-second tile is
14400 x 9600 int16 over a 30 x 20 degree box, uncompressed, which is 276 MB
apiece and 26 GB for the mean-elevation product alone.

So nothing is loaded globally. :meth:`Terrain.elevation` **groups its query
points by tile and issues one windowed read per tile**, covering only the
bounding box of the points that fall in it. A ground track crosses two or
three tiles, so a trajectory's whole elevation profile is two or three reads
of a few megabytes each — not one read per sample, and not 26 GB.

Which product, and why it matters
---------------------------------

Each tile ships six statistics: ``mea`` (mean), ``med``, ``min``, ``max``,
``std`` and ``dsc`` (systematic subsample). This uses **mean** by default.
For an impact point that is the right choice — it is the average ground
level over the cell, which is what a footprint sits on. For a *terrain
clearance* question ``max`` is the right one, because a vehicle clears the
highest ground in a cell rather than the average, so the product is a
parameter rather than a constant.

At 7.5 arc-seconds the statistics are close together, because the cell is
already near the source resolution: measured at Everest, ``mea`` gives
8665 m, ``max`` 8702 m and ``min`` 8627 m against a true summit of 8848 m.
The spread between products is 75 m and the gap to the summit is 180 m, so
**neither the choice of product nor the model is what limits a peak
elevation** — cell size is, and no statistic recovers a summit narrower than
its cell.

Voids
-----

The nodata value is -32768 and it means ocean or unmapped, not "sea level".
It is mapped to zero, and :attr:`ElevationSample.void_fraction` reports how
much of a query landed there — because an elevation profile that is 90 %
filled-in voids and one that is 90 % measured look identical otherwise.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = [
    "PRODUCTS",
    "ElevationSample",
    "Terrain",
    "default_terrain",
]

_FloatArray = NDArray[np.float64]

#: The six GMTED2010 statistics, by their filename token.
PRODUCTS = ("mea", "med", "min", "max", "std", "dsc")

#: Sentinel for ocean and unmapped cells.
NODATA = -32768


@dataclass(frozen=True)
class _Tile:
    """One GMTED2010 tile and the box it covers."""

    path: Path
    south: float
    west: float
    height_degrees: float
    width_degrees: float

    @property
    def north(self) -> float:
        return self.south + self.height_degrees

    @property
    def east(self) -> float:
        return self.west + self.width_degrees

    def contains(self, latitude: _FloatArray, longitude: _FloatArray) -> NDArray[np.bool_]:
        return np.asarray(
            (latitude >= self.south)
            & (latitude < self.north)
            & (longitude >= self.west)
            & (longitude < self.east)
        )


@dataclass(frozen=True)
class ElevationSample:
    """Elevations and how much of the query was actually measured."""

    elevation: _FloatArray
    """Metres above the ellipsoid's reference surface."""
    void: NDArray[np.bool_]
    """True where the source had no data — ocean or unmapped."""

    @property
    def void_fraction(self) -> float:
        return float(np.mean(self.void)) if self.void.size else 0.0


def _parse_directory(name: str) -> tuple[float, float, float] | None:
    """``(south, west, degrees_per_tile_lat)`` from a GMTED directory name."""
    match = re.fullmatch(
        r"GMTED2010([NS])(\d{2})([EW])(\d{3})(?:_(\d{3}))?", name
    )
    if match is None:
        return None
    ns, lat, ew, lon, _resolution = match.groups()
    south = float(lat) * (1.0 if ns == "N" else -1.0)
    west = float(lon) * (1.0 if ew == "E" else -1.0)
    # Every band spans 20 degrees of latitude and 30 of longitude, including
    # the Antarctic one, which differs only in being a 30-arc-second product.
    return south, west, 20.0


@dataclass
class Terrain:
    """The GMTED2010 archive, queried by latitude and longitude.

    Attributes
    ----------
    root:
        Directory of ``GMTED2010<band>`` subdirectories.
    product:
        Which of :data:`PRODUCTS` to read. See the module note on why this
        is a choice and not a constant.
    """

    root: Path
    product: str = "mea"
    _tiles: list[_Tile] = field(default_factory=list, repr=False)

    def __post_init__(self) -> None:
        self.root = Path(self.root)
        if self.product not in PRODUCTS:
            msg = f"product must be one of {PRODUCTS}, got {self.product!r}"
            raise ValueError(msg)
        if not self.root.is_dir():
            msg = (
                f"no GMTED2010 archive at {self.root}. Expected USGS GMTED2010 "
                f"tile directories named like GMTED2010N30E060_075."
            )
            raise FileNotFoundError(msg)
        for directory in sorted(self.root.iterdir()):
            if not directory.is_dir():
                continue
            parsed = _parse_directory(directory.name)
            if parsed is None:
                continue
            south, west, span = parsed
            matches = sorted(directory.glob(f"*_gmted_{self.product}*.tif"))
            if not matches:
                continue
            self._tiles.append(
                _Tile(matches[0], south=south, west=west, height_degrees=span,
                      width_degrees=30.0)
            )
        if not self._tiles:
            msg = (
                f"no {self.product} tiles under {self.root}; the directory has "
                f"{len(list(self.root.iterdir()))} entries but none matched "
                f"*_gmted_{self.product}*.tif"
            )
            raise FileNotFoundError(msg)

    @property
    def n_tiles(self) -> int:
        return len(self._tiles)

    @property
    def coverage(self) -> tuple[float, float]:
        """``(south, north)`` latitude limits of the archive, degrees."""
        return (
            min(t.south for t in self._tiles),
            max(t.north for t in self._tiles),
        )

    def elevation(
        self, latitude: ArrayLike, longitude: ArrayLike, degrees: bool = True
    ) -> ElevationSample:
        """Ground elevation at the given points (m).

        One windowed read per tile touched, covering the bounding box of the
        points that fall in it, then bilinear interpolation. Points outside
        the archive's latitude coverage come back as zero and flagged void.
        """
        try:
            import rasterio
            from rasterio.windows import Window
        except ImportError as error:  # pragma: no cover - dependency declared
            msg = "reading GMTED2010 needs rasterio (pip install rasterio)"
            raise ImportError(msg) from error

        lat = np.asarray(latitude, dtype=np.float64)
        lon = np.asarray(longitude, dtype=np.float64)
        lat, lon = np.broadcast_arrays(lat, lon)
        # The output shape is the *broadcast* shape, taken before flattening,
        # so a scalar query returns a scalar rather than a one-element array.
        shape = lat.shape
        lat = np.atleast_1d(lat).ravel().copy()
        lon = np.atleast_1d(lon).ravel().copy()
        if not degrees:
            lat, lon = np.rad2deg(lat), np.rad2deg(lon)
        lon = (lon + 180.0) % 360.0 - 180.0

        out = np.zeros(lat.size)
        void = np.ones(lat.size, dtype=bool)

        for tile in self._tiles:
            inside = tile.contains(lat, lon) & void
            if not np.any(inside):
                continue
            with rasterio.open(tile.path) as handle:
                transform = handle.transform
                # Fractional pixel coordinates, from the file's own affine
                # rather than an assumed origin: GMTED tiles are offset from
                # the whole degree by half an arc-second and assuming a clean
                # corner misregisters every sample.
                inverse = ~transform
                columns, rows = inverse * (lon[inside], lat[inside])
                col0 = int(np.floor(np.min(columns)).item())
                col1 = int(np.ceil(np.max(columns)).item()) + 1
                row0 = int(np.floor(np.min(rows)).item())
                row1 = int(np.ceil(np.max(rows)).item()) + 1
                col0, row0 = max(col0, 0), max(row0, 0)
                col1 = min(col1, handle.width)
                row1 = min(row1, handle.height)
                if col1 <= col0 or row1 <= row0:  # pragma: no cover - clipped away
                    continue
                block = handle.read(
                    1, window=Window(col0, row0, col1 - col0, row1 - row0)
                ).astype(np.float64)

            missing = block <= NODATA + 1
            block = np.where(missing, 0.0, block)
            local_c = np.clip(columns - col0, 0.0, block.shape[1] - 1.000001)
            local_r = np.clip(rows - row0, 0.0, block.shape[0] - 1.000001)
            c0 = local_c.astype(np.int64)
            r0 = local_r.astype(np.int64)
            c1 = np.minimum(c0 + 1, block.shape[1] - 1)
            r1 = np.minimum(r0 + 1, block.shape[0] - 1)
            fc, fr = local_c - c0, local_r - r0
            top = block[r0, c0] * (1 - fc) + block[r0, c1] * fc
            bottom = block[r1, c0] * (1 - fc) + block[r1, c1] * fc
            out[inside] = top * (1 - fr) + bottom * fr
            # A sample is void only if every corner it drew from was void;
            # a coastline cell interpolating one land corner is real data.
            corners = (
                missing[r0, c0] & missing[r0, c1] & missing[r1, c0] & missing[r1, c1]
            )
            void[inside] = corners

        return ElevationSample(
            elevation=np.asarray(out.reshape(shape)),
            void=np.asarray(void.reshape(shape)),
        )

    def coarse(
        self, height: int = 2048, cache: Path | None = None, rebuild: bool = False
    ) -> _FloatArray:
        """A decimated global elevation grid, ``(height, 2*height)`` in metres.

        Row 0 is +90 latitude, column 0 is -180 longitude — the same
        convention as the imagery, so the two can be sampled with one set of
        indices. Used for relief shading on the globe, where per-pixel
        windowed reads of a 26 GB archive are not an option.
        """
        if height < 32 or height % 2 != 0:
            msg = f"coarse height must be even and at least 32, got {height}"
            raise ValueError(msg)
        destination = (
            Path(cache) if cache is not None
            else self.root / "_coarse" / f"gmted-{self.product}-{height}.npy"
        )
        if destination.is_file() and not rebuild:
            return np.asarray(np.load(destination))

        import rasterio

        width = 2 * height
        grid = np.zeros((height, width), dtype=np.float32)
        for tile in self._tiles:
            with rasterio.open(tile.path) as handle:
                rows = max(round(tile.height_degrees / 180.0 * height), 1)
                cols = max(round(tile.width_degrees / 360.0 * width), 1)
                block = handle.read(
                    1,
                    out_shape=(rows, cols),
                    resampling=rasterio.enums.Resampling.average,
                ).astype(np.float32)
            block = np.where(block <= NODATA + 1, 0.0, block)
            top = round((90.0 - tile.north) / 180.0 * height)
            left = round((tile.west + 180.0) / 360.0 * width)
            rows = min(rows, height - top)
            cols = min(cols, width - left)
            if rows > 0 and cols > 0:
                grid[top : top + rows, left : left + cols] = block[:rows, :cols]

        destination.parent.mkdir(parents=True, exist_ok=True)
        np.save(destination, grid)
        return np.asarray(grid)


def default_terrain(root: str | Path | None = None, product: str = "mea") -> Terrain:
    """Locate the archive by walking up from this module."""
    if root is not None:
        return Terrain(Path(root), product=product)
    for parent in Path(__file__).resolve().parents:
        candidate = parent / "reference" / "GMTED2010"
        if candidate.is_dir():
            return Terrain(candidate, product=product)
    msg = (
        "no reference/GMTED2010 directory found above "
        f"{Path(__file__).resolve()}. Pass an explicit root."
    )
    raise FileNotFoundError(msg)
