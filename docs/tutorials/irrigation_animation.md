# Irrigation Animation

Irrigation events still add water to the top soil layer, and in v1.0.0 they
also log a frontend path so the enhanced dashboard can animate a sprinkler pass.

```python
from cropforge import Event

farm.add_event(Event.irrigation(
    field="Plot",
    interval_days=15,
    amount_mm=30.0,
    start_day=10,
    end_day=90,
))
```

When the event fires:

- top-layer soil moisture increases;
- the event name appears in the environment log;
- a `machine_type="sprinkler"` path is written to the machinery table;
- `/api/buffer/day/{day}` returns the sprinkler metadata;
- `farm.visualize(quality="enhanced")` shows localized blue water particles
  following the sprinkler path.

Standard quality mode keeps the path metadata but skips particle emission for
lower GPU cost.
