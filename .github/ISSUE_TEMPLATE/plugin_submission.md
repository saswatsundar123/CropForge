---
name: Crop plugin submission
about: Propose or submit a first-party crop plugin
title: "[Plugin]: "
labels: crop-package
assignees: ""
---

## Crop And Variety

- Crop species:
- Variety or cultivar:
- Target region or production system:

## Model Scope

Describe the phenology, biomass, yield, root, stress, or management behaviour
implemented by the plugin.

## Scientific Basis

List equations, calibration sources, papers, datasets, or field-trial references.
Include units for all parameters.

## Validation Evidence

Describe the tests, example scripts, or benchmark outputs included with the PR.

## Assets

If the plugin includes visual assets, list GLTF/GLB files, triangle budgets,
licensing, and whether first-party auto-registration is included.

## Compatibility Checklist

- [ ] Plugin subclasses `CropPlugin`.
- [ ] Plugin is registered with `@register_crop(...)`.
- [ ] `default_crop()` returns a valid `Crop`.
- [ ] Tests cover phenology, biomass or yield, and edge cases.
- [ ] Existing test suite passes with `pytest`.
- [ ] Documentation or an example script is included.
