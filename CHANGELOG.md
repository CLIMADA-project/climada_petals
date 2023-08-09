# Changelog

## Unreleased

Release date: YYYY-MM-DD

Code freeze date: YYYY-MM-DD

### Dependency Changes

Upgraded:

- shapely `1.8` -> `2.0` ([#80](https://github.com/CLIMADA-project/climada_petals/pull/80))

### Added

### Changed

- refactored `climada_petals.river_flood.RiverFlood.from_nc`, removing calls to `set_raster` ([#80](https://github.com/CLIMADA-project/climada_petals/pull/80))
- several adaptations have been necessary because of the `Tag` class being removed from the CLIMADA core package:
  [#88](https://github.com/CLIMADA-project/climada_petals/pull/88)
  [#89](https://github.com/CLIMADA-project/climada_petals/pull/89)

### Fixed

### Deprecated

### Removed

## v3.3.1

Release date: 2023-05-08

### Description

Release aligned with climada (core) 3.3.

### Dependency Changes

### Added

- Changelog and PR description template based on the Climada Core repository [#72](https://github.com/CLIMADA-project/climada_petals/pull/72)

### Changed

- Rework docs and switch to Book theme [#63](ttps://github.com/CLIMADA-project/climada_petals/pull/63)

### Fixed

- fix issue [#69](https://github.com/CLIMADA-project/climada_petals/issues/70) Warn.zeropadding for islands [](https://github.com/CLIMADA-project/climada_petals/pull/70)

### Deprecated

### Removed
