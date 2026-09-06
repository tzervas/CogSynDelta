"""The interconnect test package -- shared fixtures live in `conftest.py` (lane IC-8,
spec `docs/design/INTERCONNECT-MODULE-SPEC.md` section 7 Table 10: "shared test
fixtures live in `tests/interconnect/conftest.py` and `tests/interconnect/__init__.py`,
both owned by IC-8"). This file exists to make the directory an ordinary package;
`conftest.py` carries every fixture other lanes' test files, and this file's own
`test_integration.py`/`test_smoke.py`, import.
"""
