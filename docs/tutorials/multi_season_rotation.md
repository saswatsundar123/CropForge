# Multi-Season Crop Rotation (v0.4.0)

CropForge v0.4.0 introduces the ability to run consecutive seasons on the same soil, preserving soil moisture, nitrogen state, and physical structure between runs. This is critical for simulating crop rotations and long-term soil health.

## The Multi-Season Workflow

To run a multi-season simulation, you will:

1. Run the first season on a `Farm` instance.
2. Save the final soil state using `farm.save_state()`.
3. Load the state into the next season using `farm.load_state()`.
4. Call `farm.prepare_next_season()` to correctly bump the season counter and prepare the engine.
5. Run the second season.

## Example: Two-Season Rotation

Below is a complete script demonstrating a two-season workflow on the same field.

```python
import tempfile
from pathlib import Path
from cropforge import Farm, Field

output_dir = tempfile.mkdtemp(prefix="cropforge_2season_")

# ===========================================================================
# Season 1
# ===========================================================================
print("Running Season 1 (days 1–10)...")
farm = Farm(name="RotationDemo")
field = Field(name="WheatPlot", rows=4, cols=4)
farm.add_field(field)

# Run the first season
farm.run(days=10)

# Save the state at the end of the season
state_file = Path(output_dir) / "s1_end.cfstate"
farm.save_state(str(state_file))
print(f"Season 1 complete. Saved state to {state_file}")

# ===========================================================================
# Season 2
# ===========================================================================
print("Preparing Season 2...")

# Load the saved state from Season 1
farm.load_state(str(state_file))

# Advance the season counter and prepare the simulation engine
farm.prepare_next_season()

print("Running Season 2 (days 11–20)...")
# Run the second season (starts at day 11 due to state carry-over)
farm.run(days=10)

# ===========================================================================
# Visualisation
# ===========================================================================
# Launch the dashboard to see both seasons
farm.visualize()
```

## Dashboard Behaviour

When you visualize a multi-season log:

1. The Time-Series chart will automatically draw an amber dashed vertical line at the boundary between seasons (e.g., at day 11).
2. All data across both seasons is preserved and can be scrubbed in the 3D viewport.
3. The **Export CSV** button in the dashboard will export the full multi-season dataset with a dedicated `season` column.
