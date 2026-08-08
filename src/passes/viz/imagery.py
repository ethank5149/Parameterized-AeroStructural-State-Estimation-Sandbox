"""Blue Marble Next Generation as the globe texture, at the month being flown.

``reference/blue-marble-next-gen`` holds all twelve months of NASA's BMNG
topography-and-bathymetry product: eight GeoTIFF tiles each, 21600 x 21600,
three uint8 bands, EPSG:4326. Assembled that is **86400 x 43200 at 15
arc-seconds** — roughly 460 m at the equator, against the 4096 x 2048 JPEG
the renderer used before, which is 9.7 km. Two and a half thousand times the
pixel count.

Which is exactly the problem. The full mosaic is 11 GB as uint8 and 89 GB as
the float64 the renderer wants, so it cannot simply be loaded. Two paths are
provided instead, and they are the two things a trajectory animation
actually asks for:

:meth:`BlueMarble.mosaic`
    A decimated global equirectangular texture at a requested height,
    assembled once and cached to disk as ``uint8``. This is what a full-disc
    or mid-range view needs, where the globe spans at most a thousand pixels
    and anything finer than about 8192 x 4096 is thrown away by the
    rasteriser regardless.
:meth:`BlueMarble.window`
    A native-resolution crop of a latitude/longitude box, read straight from
    the GeoTIFFs. This is what a launch-pad or impact close-up needs, where
    the camera is 20 km up and the visible ground is a fraction of a degree
    across. At that range the global mosaic would be showing one texel per
    fifty pixels.

**Stored as uint8, converted on upload.** The source is uint8, the renderer
divides by 255 anyway, and keeping the cache in the source dtype makes an
8192 x 4096 texture 100 MB instead of the 800 MB it would be as float64 —
which is the difference between a texture that fits on the GPU beside the
frame buffers and one that does not.

**The month is chosen, not assumed.** BMNG's whole point is that the surface
changes: northern hemisphere snow line, Sahel vegetation, sea ice. A
January launch rendered on the August texture is a different planet at the
latitudes an ICBM trajectory actually crosses. :meth:`BlueMarble.for_date`
resolves a date to the right month.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

__all__ = ["MONTH_NAMES", "BlueMarble", "TileKey", "default_blue_marble"]

_ByteImage = NDArray[np.uint8]

MONTH_NAMES = (
    "january", "february", "march", "april", "may", "june",
    "july", "august", "september", "october", "november", "december",
)

#: ``(column, row)`` of each BMNG tile and the degree box it covers.
#: Columns A-D run west to east from 180 W; rows 1-2 run north then south.
_COLUMN_WEST = {"A": -180.0, "B": -90.0, "C": 0.0, "D": 90.0}
_ROW_NORTH = {"1": 90.0, "2": 0.0}

TileKey = str
"""One of ``A1``, ``A2``, ..., ``D2``."""


def _parse(name: str) -> tuple[int, TileKey] | None:
    """Month number and tile key from a BMNG filename, or ``None``."""
    month = re.search(r"\.(\d{4})(\d{2})\.", name)
    tile = re.search(r"([ABCD][12])_geo", name)
    if month is None or tile is None:
        return None
    return int(month.group(2)), tile.group(1)


@dataclass
class BlueMarble:
    """The BMNG archive, indexed by month and tile.

    Attributes
    ----------
    root:
        Directory holding the GeoTIFFs, in any arrangement — the index is
        built from *filenames*, so the month subdirectories, the loose files
        at the top level and the ``(1)`` duplicates a download leaves behind
        all resolve to the same place.
    cache:
        Where assembled mosaics are written. A mosaic costs a minute or two
        to build because the source tiles are deflate-compressed and
        striped, with no overviews; it is not something to redo per frame.
    """

    root: Path
    cache: Path | None = None
    _index: dict[int, dict[TileKey, Path]] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        self.root = Path(self.root)
        if not self.root.is_dir():
            msg = (
                f"no Blue Marble archive at {self.root}. Expected NASA's Blue "
                f"Marble Next Generation GeoTIFFs, eight tiles per month named "
                f"like world.topo.bathy.200401.3x21600x21600.C1_geo.tif."
            )
            raise FileNotFoundError(msg)
        if self.cache is None:
            self.cache = self.root / "_mosaics"
        for path in sorted(self.root.rglob("*.tif")):
            parsed = _parse(path.name)
            if parsed is None:
                continue
            month, tile = parsed
            # First match wins, so a "(1)" duplicate never displaces the
            # original it was copied from.
            self._index.setdefault(month, {}).setdefault(tile, path)

    @property
    def months(self) -> tuple[int, ...]:
        """Month numbers with a complete eight-tile set."""
        return tuple(sorted(m for m, t in self._index.items() if len(t) == 8))

    def tiles(self, month: int) -> dict[TileKey, Path]:
        """The eight tile paths for a month number (1-12)."""
        available = self._index.get(int(month), {})
        if len(available) != 8:
            msg = (
                f"month {month} has {len(available)} of 8 Blue Marble tiles in "
                f"{self.root}; complete months are {self.months}"
            )
            raise FileNotFoundError(msg)
        return dict(available)

    @staticmethod
    def month_of(when: date | datetime | str | int) -> int:
        """Resolve a date, month name or number to a month number."""
        if isinstance(when, int):
            if not 1 <= when <= 12:
                msg = f"month must be 1-12, got {when}"
                raise ValueError(msg)
            return when
        if isinstance(when, (date, datetime)):
            return int(when.month)
        text = str(when).strip()
        # Lower-cased only for the month-name lookup: an ISO timestamp
        # carries a capital "T" that numpy will not parse in lower case.
        if text.lower() in MONTH_NAMES:
            return MONTH_NAMES.index(text.lower()) + 1
        stamp = np.datetime64(text).astype("datetime64[M]").astype(int)
        return int(stamp % 12) + 1

    def for_date(self, when: date | datetime | str | int, height: int = 4096) -> _ByteImage:
        """Global mosaic for the month containing ``when``."""
        return self.mosaic(height=height, month=self.month_of(when))

    # -- global mosaic -----------------------------------------------------

    def mosaic_path(self, height: int, month: int) -> Path:
        assert self.cache is not None
        return self.cache / f"bmng-{month:02d}-{height}.npy"

    def mosaic(
        self, height: int = 4096, month: int = 1, rebuild: bool = False
    ) -> _ByteImage:
        """Decimated global equirectangular texture, ``(height, 2*height, 3)`` uint8.

        Row 0 is +90 latitude and column 0 is -180 longitude, which is the
        convention :func:`passes.viz.globe.render` samples with.

        Built by asking rasterio for a decimated read of each tile directly
        into its slot in the output, so the full-resolution image is never
        materialised. The source has no overviews, so this still decompresses
        every strip — about a minute a month — which is why the result is
        cached.
        """
        if height < 64 or height % 2 != 0:
            msg = f"mosaic height must be even and at least 64, got {height}"
            raise ValueError(msg)
        destination = self.mosaic_path(height, month)
        if destination.is_file() and not rebuild:
            return np.asarray(np.load(destination))

        try:
            import rasterio
        except ImportError as error:  # pragma: no cover - dependency declared
            msg = "reading Blue Marble GeoTIFFs needs rasterio (pip install rasterio)"
            raise ImportError(msg) from error

        width = 2 * height
        tile_h, tile_w = height // 2, width // 4
        image = np.zeros((height, width, 3), dtype=np.uint8)
        columns = {"A": 0, "B": 1, "C": 2, "D": 3}

        for key, path in sorted(self.tiles(month).items()):
            column, row = key[0], key[1]
            top = 0 if row == "1" else tile_h
            left = columns[column] * tile_w
            with rasterio.open(path) as handle:
                block = handle.read(
                    indexes=[1, 2, 3],
                    out_shape=(3, tile_h, tile_w),
                    resampling=rasterio.enums.Resampling.average,
                )
            image[top : top + tile_h, left : left + tile_w] = np.transpose(
                block, (1, 2, 0)
            )

        destination.parent.mkdir(parents=True, exist_ok=True)
        np.save(destination, image)
        return image

    # -- native-resolution window ------------------------------------------

    def window(
        self,
        latitude: tuple[float, float],
        longitude: tuple[float, float],
        max_width: int = 4096,
    ) -> tuple[_ByteImage, tuple[float, float, float, float]]:
        """Native-resolution crop of a lat/lon box, in degrees.

        Returns the image and its actual ``(south, north, west, east)``
        bounds, which are snapped outward to whole source pixels — a caller
        that assumed it got exactly the box it asked for would misregister
        the texture by up to half a pixel, and at 15 arc-seconds that is
        230 m on the ground.

        Boxes spanning the antimeridian are refused rather than silently
        wrapped: the crop would be two disjoint reads, and returning one
        array with a seam in the middle of it is worse than an error.
        """
        try:
            import rasterio
            from rasterio.windows import Window
        except ImportError as error:  # pragma: no cover - dependency declared
            msg = "reading Blue Marble GeoTIFFs needs rasterio (pip install rasterio)"
            raise ImportError(msg) from error

        south, north = sorted(float(v) for v in latitude)
        west, east = float(longitude[0]), float(longitude[1])
        if east <= west:
            msg = (
                f"longitude box must run west to east without crossing the "
                f"antimeridian, got ({west}, {east})"
            )
            raise ValueError(msg)
        if not (-90.0 <= south < north <= 90.0):
            msg = f"latitude box must lie in [-90, 90] and be non-empty, got {latitude}"
            raise ValueError(msg)

        month = self.months[0]
        pieces: list[tuple[Any, ...]] = []
        for key, path in sorted(self.tiles(month).items()):
            tile_west = _COLUMN_WEST[key[0]]
            tile_north = _ROW_NORTH[key[1]]
            tile_east, tile_south = tile_west + 90.0, tile_north - 90.0
            if east <= tile_west or west >= tile_east:
                continue
            if north <= tile_south or south >= tile_north:
                continue
            pieces.append((key, path, tile_west, tile_north))
        if not pieces:  # pragma: no cover - the boxes above tile the globe
            msg = f"no Blue Marble tile covers {latitude}, {longitude}"
            raise ValueError(msg)

        # Single-tile fast path is the common one: a launch-pad or impact
        # close-up is a fraction of a degree and never straddles a 90-degree
        # tile edge except by coincidence.
        key, path, tile_west, tile_north = pieces[0]
        with rasterio.open(path) as handle:
            degrees_per_pixel = 90.0 / handle.width
            col0 = int(np.floor((west - tile_west) / degrees_per_pixel))
            col1 = int(np.ceil((east - tile_west) / degrees_per_pixel))
            row0 = int(np.floor((tile_north - north) / degrees_per_pixel))
            row1 = int(np.ceil((tile_north - south) / degrees_per_pixel))
            col0, col1 = max(col0, 0), min(col1, handle.width)
            row0, row1 = max(row0, 0), min(row1, handle.height)
            span_w, span_h = col1 - col0, row1 - row0
            if span_w <= 0 or span_h <= 0:  # pragma: no cover - guarded above
                msg = f"empty crop for {latitude}, {longitude}"
                raise ValueError(msg)
            scale = min(1.0, max_width / span_w)
            out_w = max(round(span_w * scale), 1)
            out_h = max(round(span_h * scale), 1)
            block = handle.read(
                indexes=[1, 2, 3],
                window=Window(col0, row0, span_w, span_h),
                out_shape=(3, out_h, out_w),
                resampling=rasterio.enums.Resampling.average,
            )
        bounds = (
            tile_north - row1 * degrees_per_pixel,
            tile_north - row0 * degrees_per_pixel,
            tile_west + col0 * degrees_per_pixel,
            tile_west + col1 * degrees_per_pixel,
        )
        return np.transpose(block, (1, 2, 0)), bounds


def default_blue_marble(root: str | Path | None = None) -> BlueMarble:
    """Locate the archive by walking up from this module.

    Same reasoning as the old texture loader: a notebook runs from its own
    folder and a test from the repository root, so a path relative to the
    process working directory works in exactly one of them.
    """
    if root is not None:
        return BlueMarble(Path(root))
    for parent in Path(__file__).resolve().parents:
        candidate = parent / "reference" / "blue-marble-next-gen"
        if candidate.is_dir():
            return BlueMarble(candidate)
    msg = (
        "no reference/blue-marble-next-gen directory found above "
        f"{Path(__file__).resolve()}. Pass an explicit root."
    )
    raise FileNotFoundError(msg)
