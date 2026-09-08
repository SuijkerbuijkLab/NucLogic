"""Combining the voxel size read from a file with the user's override.

The override is per axis. Each box in the advanced settings is either filled
in, in which case that number is used, or left on "auto", in which case the
value read from the file is used. Nothing else consults the metadata_missing
flag: a number the user typed is a deliberate statement about their microscope
and always wins, whether or not the file also had an opinion.

This used to be all-or-nothing -- the whole override applied only when the file
carried no calibration at all, and a blank box meant 1.0 rather than "leave it
alone". That silently discarded good measured values: a merged MetaMorph sample
has a real Z step but no XY calibration, so filling in only XY replaced the
correct Z with 1.0 and scaled every volume wrong.
"""


def parse_override(z_text, y_text, x_text):
    """Turn three text boxes into a (z, y, x) override, None where blank.

    Raises ValueError with a message meant for the user when a box holds
    something that is not a positive number.
    """
    values = []
    for axis, text in (("Z", z_text), ("Y", y_text), ("X", x_text)):
        text = (text or "").strip()
        if not text:
            values.append(None)
            continue
        try:
            value = float(text)
        except ValueError:
            raise ValueError(
                f"Invalid voxel size override for {axis}: {text!r} is not a "
                f"number. Leave a box empty to read that axis from the file."
            ) from None
        if value <= 0:
            raise ValueError(
                f"Invalid voxel size override for {axis}: must be greater "
                f"than zero."
            )
        values.append(value)
    return tuple(values)


def resolve_voxel_size(measured, override, metadata_missing=False):
    """(z, y, x) in micrometres, taking each axis from the override if set.

    `measured` is what load_image read from the file; `override` may be None,
    or a (z, y, x) tuple whose entries are None where the user left the box on
    auto. `metadata_missing` only drives a warning: an axis that the file never
    calibrated and the user did not fill in is silently 1 um, which quietly
    invalidates every size-based measurement, so say so.
    """
    measured = _to_tuple(measured)
    override = _to_tuple(override, allow_missing=True)

    resolved = tuple(
        measured[axis] if override[axis] is None else override[axis]
        for axis in range(3)
    )

    if resolved != measured:
        overridden = ", ".join(
            f"{name}={override[axis]}"
            for axis, name in enumerate("ZYX")
            if override[axis] is not None
        )
        print(f"Using voxel size override: {overridden} (um); resolved {resolved}")

    if metadata_missing:
        guessed = [
            name
            for axis, name in enumerate("ZYX")
            if override[axis] is None and resolved[axis] == 1.0
        ]
        if guessed:
            print(
                f"Warning: the file carries no {'/'.join(guessed)} calibration "
                f"and no override was given, so 1.0 um per pixel is assumed. "
                f"Sizes, volumes and distances will be in pixels, not "
                f"micrometres. Set the voxel size override in the advanced "
                f"settings to fix this."
            )

    return resolved


def _to_tuple(voxel_like, allow_missing=False):
    default = (None, None, None) if allow_missing else (1.0, 1.0, 1.0)
    try:
        return tuple(
            None if (allow_missing and voxel_like[axis] is None)
            else float(voxel_like[axis])
            for axis in range(3)
        )
    except (TypeError, ValueError, IndexError, KeyError):
        return default
