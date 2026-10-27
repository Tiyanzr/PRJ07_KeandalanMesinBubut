"""Single source of truth for PRJ07 paths and constants."""
from pathlib import Path

ROOT    = Path(__file__).resolve().parent                       # ...\Program & Data\PRJ07
RAW     = ROOT.parent / "Raw Data" / "Monitoring Mesin Bubut"   # shared source data
BRONZE  = RAW / "bronze"                                        # partitioned sensor parquet
REF     = RAW / "ref"                                           # machine_util, emergencies, maintenance
INTERIM = ROOT / "data" / "interim"
FIG     = ROOT / "outputs" / "figures"
TAB     = ROOT / "outputs" / "tables"

PATCHES = {"patch1": {"start": "2026-06-15", "end": "2026-10-19"}}   # add a new entry per export, never edit old ones
PATCH   = "patch1"
SILVER  = ROOT / "data" / "silver" / PATCH

MACHINES     = {3: "Machine 3", 4: "Machine 4 (krakatau)"}
TEST_DAYS    = ["2026-06-15", "2026-06-29"]   # operator TEST/TESTING
ANOMALY_DAYS = ["2026-07-10"]                 # rapid repeated button presses
COMMISSIONING_STAFF = ["HANAFI", "REV"]      # PT Parametrik staff testing during installation (confirmed by researcher)
TZ_DATA      = "UTC"                          # sensors, status log, maintenance history
TZ_LOCAL     = "Asia/Jakarta"                 # emergencies are logged in local time (WIB)
DUCKDB_TEMP  = "E:/duckdb_tmp"

for _p in (SILVER, INTERIM, FIG, TAB):
    _p.mkdir(parents=True, exist_ok=True)
