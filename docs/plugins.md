# Plugin Ecosystem (v0.4.0)

CropForge is designed to be extensible. Researchers can package custom crop models as plugins and distribute them on PyPI.

## Using a Plugin

If another researcher has published a plugin, you can install it via `pip`:

```bash
pip install cropforge-wheat
```

Then, in your simulation script, retrieve the plugin and apply it to a field:

```python
from cropforge import Farm, Field, get_plugin

farm = Farm(name="PluginDemo")
field = Field(name="Plot1", rows=10, cols=10)

# Load the plugin by its registered name
wheat_plugin = get_plugin("wheat")

# The setup method applies the plugin's hooks to the farm and field
wheat_plugin.setup(farm, field)

farm.run(days=120)
```

## Creating a Plugin

To create a plugin, you subclass `CropPlugin` and register it using the `@register_crop` decorator. Your plugin can define custom setup logic, physics hooks, and daily step functions.

### Example: A Custom Maize Plugin

```python
from cropforge import Farm, Field, CropPlugin, register_crop

@register_crop("maize")
class MaizePlugin(CropPlugin):
    """A custom CropForge plugin for simulating Maize."""

    def __init__(self):
        self.crop_name = "Maize"
        self.default_spacing_cm = 30.0

    def setup(self, farm: Farm, field: Field) -> None:
        """Called once when the plugin is applied to a field."""
        farm.add_field(field)
        
        # Register a custom daily growth function
        @farm.step(interval="daily")
        def _maize_growth(state, env):
            self.on_daily_step(state, env)
            
    def on_daily_step(self, field_state, env_state) -> None:
        """Custom daily growth logic."""
        for plant in field_state.plants:
            if plant.alive:
                # Custom maize biomass accumulation
                plant.biomass_g += env_state.radiation_mj_m2 * 1.5
                plant.height_cm += 1.2
```

## Publishing to PyPI

We encourage naming your package `cropforge-{crop}` (e.g., `cropforge-soybean`). 

When users install your package, they just need to import it once to trigger the `@register_crop` decorator, making it available via `get_plugin("soybean")`.

You can structure your Python package like this:

```text
cropforge-maize/
├── pyproject.toml
└── cropforge_maize/
    ├── __init__.py
    └── plugin.py       # Contains your @register_crop("maize") class
```

In your `__init__.py`, import the plugin so it registers automatically:

```python
from .plugin import MaizePlugin
__all__ = ["MaizePlugin"]
```
