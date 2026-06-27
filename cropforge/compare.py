"""
cropforge/compare.py
=====================
compare(*farms) -- overlay multiple Farm runs in a single dashboard session.

PRD v0.4.0 Section 8.1:
    compare(farm_irrigated, farm_rainfed)
    Opens the dashboard showing both farms on the same time-series chart,
    colour-coded by farm name.

Implementation:
    compare() is architecturally a multi-farm visualise() call (PRD §8.1 note).
    It merges the Parquet logs of all supplied farms into one combined session
    directory, then boots the server.

    Each farm's field names are prefixed with the farm name so they stay
    distinguishable in the field-selector dropdown and time-series legend:
        "Irrigated :: PlotA"   (from farm named "Irrigated")
        "Rainfed  :: PlotA"    (from farm named "Rainfed")

    The existing multi-field infrastructure (FieldBufferRegistry, multi-field
    callbacks) handles any number of such prefixed fields without modification.

Author : Saswat Sundar Rath, ICAR-IARI Jharkhand
Licence: MIT
"""

from __future__ import annotations

import logging
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cropforge.farm import Farm

logger = logging.getLogger(__name__)


def compare(*farms: "Farm", port: int = 7860) -> None:
    """Launch the CropForge dashboard overlaying multiple farm runs.

    Each farm's fields are prefixed with the farm name in the legend so
    traces are immediately distinguishable:
        "Irrigated :: PlotA" vs "Rainfed :: PlotA"

    Parameters
    ----------
    *farms:
        Two or more ``Farm`` instances.  Each must have been run
        (``farm.run()`` called) and have a valid ``_last_log_path``.
    port:
        TCP port for the dashboard server (default 7860).

    Raises
    ------
    ValueError
        If fewer than 2 farms are supplied, or if any farm has no log.

    Examples
    --------
    >>> farm_a.run(days=90)
    >>> farm_b.run(days=90)
    >>> from cropforge import compare
    >>> compare(farm_a, farm_b)
    """
    import cropforge as _cf
    from cropforge.runtime import CropForgeVisualizeError

    # ---- Validation -------------------------------------------------------
    if len(farms) < 2:
        raise ValueError(
            "compare() requires at least 2 Farm objects. "
            f"Got {len(farms)}. Use farm.visualize() for a single farm."
        )

    for farm in farms:
        if not farm._last_log_path:
            raise CropForgeVisualizeError(
                f"Farm {farm.name!r} has no simulation log. "
                "Call farm.run() before compare()."
            )
        log_dir = Path(farm._last_log_path)
        if not log_dir.exists() or not list(log_dir.rglob("*.parquet")):
            raise CropForgeVisualizeError(
                f"Farm {farm.name!r} log at {farm._last_log_path!r} "
                "is empty or missing. Re-run farm.run()."
            )

    # ---- Build a merged session directory --------------------------------
    # We create a temporary directory that the dashboard server will read.
    # For each Parquet table (plants, soil, environment), we copy all
    # partition files from all farms, rewriting the field_name partition
    # directory to include the farm name as a prefix:
    #   plants/field_name=Irrigated :: PlotA/day=1/part-0.parquet
    #
    # This approach is zero-config: the existing _load_parquet() + callbacks
    # handle any field name string, so prefixed names "just work".

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    merged_dir = Path(tempfile.mkdtemp(prefix="cropforge_compare_"))
    logger.info("compare(): merged session dir = %s", merged_dir)

    for farm in farms:
        farm_prefix = farm.name.replace(" ", "_").replace("/", "_")
        src_root = Path(farm._last_log_path)

        for table in ("plants", "soil", "environment"):
            src_table = src_root / table
            if not src_table.exists():
                continue

            dst_table = merged_dir / table

            # Walk partition dirs: field_name=X / day=D / *.parquet
            for field_part_dir in src_table.iterdir():
                if not field_part_dir.is_dir():
                    continue
                # Extract original field name from Hive partition name
                if field_part_dir.name.startswith("field_name="):
                    orig_field = field_part_dir.name[len("field_name="):]
                else:
                    orig_field = field_part_dir.name

                # New partition dir: "FarmName :: OriginalField"
                new_field_label = f"{farm.name} -- {orig_field}"
                # Hive-encode the new field name for the directory
                new_field_dir_name = f"field_name={new_field_label}"
                dst_field_dir = dst_table / new_field_dir_name

                # Copy all day partitions under this field
                for day_part_dir in field_part_dir.iterdir():
                    if not day_part_dir.is_dir():
                        continue
                    dst_day_dir = dst_field_dir / day_part_dir.name
                    dst_day_dir.mkdir(parents=True, exist_ok=True)
                    for parquet_file in day_part_dir.glob("*.parquet"):
                        shutil.copy2(parquet_file, dst_day_dir / parquet_file.name)

    logger.info(
        "compare(): merged %d farms into %s. Booting dashboard.",
        len(farms), merged_dir,
    )

    # ---- Boot the dashboard on the merged session ------------------------
    from cropforge.viz.server import boot
    boot(log_path=str(merged_dir), cropforge_version=_cf.__version__)
