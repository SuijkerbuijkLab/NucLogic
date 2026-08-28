# NucLogic tests

Run everything:

```
pixi run python tests/run_all.py
```

Or a single test directly:

```
pixi run python tests/test_load_image.py
```

Each test is a standalone script that prints `[PASS]` / `[FAIL]` lines and exits
non-zero on failure. There is no pytest dependency.

## Sample data

Four tests need real microscopy files. They resolve the paths through
`tests/local_data.py` (gitignored) or the equivalent environment variables, and
report `SKIPPED` when a file is unavailable — so the suite runs anywhere, it just
covers less.

| Name | What it should point at |
| --- | --- |
| `NUCLOGIC_TEST_IMS` | a multi-timepoint `.ims` straight from the microscope |
| `NUCLOGIC_TEST_IMS_CROPPED` | a `_cropped.ims` written by the pipeline |
| `NUCLOGIC_TEST_ND2` | any Nikon `.nd2` |

Create `tests/local_data.py` with those names assigned to paths, or set them as
environment variables (which take precedence).

## What each test covers

| Test | Covers |
| --- | --- |
| `test_load_image.py` | TIFF (OME/ImageJ/plain/compressed) and OME-Zarr, unit conversion, extension matching, error paths |
| `test_nd2_czi.py` | a real CZI written with pylibCZIrw; the ND2 loader driven by a stub reader |
| `test_real_nd2.py` | a real Nikon file: axis order verified against an independent transpose of `nd2.imread` |
| `test_real_ims.py` | a real Imaris file: lazy open, voxel size, pixel fidelity vs the reader itself |
| `test_ims_multichunk.py` | multi-chunk IMS reads. **Fails loudly if `_DimensionPreservingReader` is removed** |
| `test_ims_time.py` | the nanosecond time-interval fix against real files |
| `test_ims_roundtrip.py` | `save_as_ims` → `load_image`: voxel size, interval and acquisition start survive |
| `test_ims_mocked.py` | IMS loader behaviour with a stub reader (3D/4D/5D padding, laziness) |
| `test_migration.py` | every migrated module imports; new load path matches the old one on real files |
| `test_write_metadata.py` | every file the pipeline writes reads back correctly through `load_image` |
| `test_max_project.py` | projections equal a plain numpy max; written metadata round-trips |
| `test_updater.py` | dependency-aware rebuild, LFS pointer protection |
| `test_update_banner.py` | update banner state machine and the restart wiring |

`audit_write_metadata.py` is a report, not a test: it lists every
`tifffile.imwrite` in the pipeline and which metadata each one records.

## Notes

Tests run in separate processes so a crash cannot take the rest down and file
handles are always released. `_harness.py` puts the pixi environment's DLL
directories on `PATH`, so running with a bare `python.exe` works too — without
it, importing `h5py` fails on Windows.
