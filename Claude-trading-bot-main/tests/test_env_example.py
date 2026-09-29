"""A6: .env.example documents every environment variable the code reads, with safe defaults."""

import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _env_vars_in_code():
    names = set()
    for f in ("config.py", "binance_client.py"):
        src = open(os.path.join(ROOT, f), encoding="utf-8").read()
        names |= set(re.findall(r'(?:os\.getenv|_env_num)\(\s*"([A-Z0-9_]+)"', src))
    return names


def _example():
    path = os.path.join(ROOT, ".env.example")
    values = {}
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            values[k.strip()] = v.strip()
    return values


def test_every_env_var_is_documented():
    missing = _env_vars_in_code() - set(_example())
    assert not missing, f"missing in .env.example: {sorted(missing)}"


def test_example_is_safe_demo_mode_without_keys():
    ex = _example()
    assert ex["PAPER_TRADING"] == "true" and ex["USE_TESTNET"] == "true"
    for key in ("BINANCE_API_KEY", "BINANCE_API_SECRET", "ANTHROPIC_API_KEY"):
        assert ex[key] == "", f"{key} must be empty in the example"
    assert ex["ALLOW_UNVALIDATED_STRATEGIES"] == "false"
    assert ex["RESET_ON_STARTUP"] == "false"
