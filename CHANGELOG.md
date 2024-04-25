# Changelog

## Unreleased

Release date: YYYY-MM-DD

Code freeze date: YYYY-MM-DD

### Description

### Dependency Changes

- `boario` >=0.5.9

### Added

- Included `boario` in the supplychain module [#81](https://github.com/CLIMADA-project/climada_petals/pull/81/)

### Changed

### Fixed

- Fix `climada.hazard.tc_rainfield` for TC tracks crossing the antimeridian [#105](https://github.com/CLIMADA-project/climada_petals/pull/105)

### Deprecated

### Removed

## 4.1.0

Release date: 2024-02-19

### Dependency Changes

Updated:

- `climada` >=4.0 &rarr; ==4.1

Added:

- `overpy` >=0.7
- `osm-flex` >=1.1.1
- `pymrio` >=0.5

### Changed

- Restructured `Supplychain` module, which now uses `pymrio` to download and handle multi-regional input output tables [#66](https://github.com/CLIMADA-project/climada_petals/pull/66)
- Restructured `openstreetmap` module to draw functionalities from external package osm-flex [#103](https://github.com/CLIMADA-project/climada_petals/pull/103)
- As part of `climada_petals.hazard.tc_rainfield`, implement a new, physics-based TC rain model ("TCR") in addition to the existing implementation of the purely statistical R-CLIPER model ([#85](https://github.com/CLIMADA-project/climada_petals/pull/85))
- Conda environment now avoids `default` channel packages, as these are incompatible to conda-forge [#110](https://github.com/CLIMADA-project/climada_petals/pull/110)

## 4.0.2

Release date: 2023-09-27

### Dependency Changes

- `pandas` >=1.5,<2.0 &rarr; >=1.5 (compatibility with pandas 2.x)

### Changed

- improved integration tests for notebooks and external data apis

### Fixed

- implicit casting from `DataArray` to `int` in reading mehtods made explicit [#95](https://github.com/CLIMADA-project/climada_petals/pull/95/files)

## 4.0.1

Release date: 2023-09-06

### Fixed

- `TCForecast` now skips "untrackable" TCs when reading multi-message `.bufr` files [#91](https://github.com/CLIMADA-project/climada_petals/pull/91)

## 4.0.0

Release date: 2023-09-01

### Dependency Changes

Upgraded:

- shapely `1.8` -> `2.0` ([#80](https://github.com/CLIMADA-project/climada_petals/pull/80))

### Changed

- refactored `climada_petals.river_flood.RiverFlood.from_nc`, removing calls to `set_raster` ([#80](https://github.com/CLIMADA-project/climada_petals/pull/80))
- Replace `tag` attribute with string `description` in classes derived from `Exposure` [#89](https://github.com/CLIMADA-project/climada_petals/pull/89)

### Removed

- `tag` attribute from hazard classes [#88](https://github.com/CLIMADA-project/climada_petals/pull/88)

## v3.3.2

Release date: 2023-08-25

### Description

Patch release

## v3.3.1

Release date: 2023-08-24

### Description

Rearranged file-system structure: `data` subdirectory of `climada_petals`.

## v3.3.0

Release date: 2023-05-08

### Description

Release aligned with climada (core) 3.3.

### Added

- Changelog and PR description template based on the Climada Core repository [#72](https://github.com/CLIMADA-project/climada_petals/pull/72)

### Changed

- Rework docs and switch to Book theme [#63](ttps://github.com/CLIMADA-project/climada_petals/pull/63)

### Fixed

- fix issue [#69](https://github.com/CLIMADA-project/climada_petals/issues/70) Warn.zeropadding for islands [](https://github.com/CLIMADA-project/climada_petals/pull/70)

