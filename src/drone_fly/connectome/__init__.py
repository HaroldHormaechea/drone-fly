"""Connectome stage: fetch MaleCNS connectivity from neuPrint and cache it locally.

Responsible for the only inbound external data. Reads are treated as read-only
reference data; the neuPrint auth token is supplied via the environment, never
committed. Bulk connectome data is cached under data/connectome/ and kept out of
version control for licensing reasons.

For UC-01 the load path is offline only (a cached matrix on disk); live neuPrint
fetching is a separate, later concern. See :mod:`drone_fly.connectome.loader`.
"""

from drone_fly.connectome.fetch import (
    ConnectomeDownloadError,
    ensure_full_connectome,
)
from drone_fly.connectome.loader import (
    CONNECTOME_DIR_ENV,
    DEFAULT_CONNECTOME_DIR,
    DEFAULT_SAVE_STEM,
    FIXTURE_EXPECTED_SCALE,
    MALECNS_V1_EXPECTED_SCALE,
    ConnectomeData,
    ExpectedScale,
    load_connectome,
    save_connectome,
)
from drone_fly.connectome.prune import (
    DEFAULT_PRUNE_K,
    DEFAULT_PRUNE_RULE,
    PRUNE_RULE_PATH_SLACK,
    prune_to_subcircuit,
    slice_connectome,
)

__all__ = [
    "CONNECTOME_DIR_ENV",
    "DEFAULT_CONNECTOME_DIR",
    "DEFAULT_SAVE_STEM",
    "FIXTURE_EXPECTED_SCALE",
    "MALECNS_V1_EXPECTED_SCALE",
    "ConnectomeData",
    "ConnectomeDownloadError",
    "ExpectedScale",
    "ensure_full_connectome",
    "load_connectome",
    "save_connectome",
    "DEFAULT_PRUNE_K",
    "DEFAULT_PRUNE_RULE",
    "PRUNE_RULE_PATH_SLACK",
    "prune_to_subcircuit",
    "slice_connectome",
]
