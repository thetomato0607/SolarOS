"""Site and array constants for Pecan Street home 661 in Austin, TX."""

import pvlib

AUSTIN_LAT = 30.2672   # deg N
AUSTIN_LON = -97.7431  # deg W
AUSTIN_ALT = 150.0     # metres (average Austin elevation)
LOCAL_TZ = "America/Chicago"

# Array geometry, from the Dataport metadata row for home 661:
# pv_panel_direction = "South", amount_of_south_facing_pv = 6.3 and
# total_amount_of_pv = 6.3, so the array is 100% south-facing, 6.3 kW DC.
# Tilt is not recorded, so it is assumed equal to the latitude (30 deg).
TILT_DEG = 30.0        # degrees from horizontal (assumption)
AZIMUTH_DEG = 180.0    # degrees, 180 = south (from metadata)
CAPACITY_KW = 6.3      # kW DC nameplate (from metadata; an early fit used 5.6)


def make_location(name=None) -> pvlib.location.Location:
    """Return the pvlib Location for the site; ``name`` only affects its repr."""
    return pvlib.location.Location(
        latitude=AUSTIN_LAT,
        longitude=AUSTIN_LON,
        tz=LOCAL_TZ,
        altitude=AUSTIN_ALT,
        name=name,
    )
