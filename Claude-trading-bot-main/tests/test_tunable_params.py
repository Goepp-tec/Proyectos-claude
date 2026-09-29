"""B1: each strategy declares its tunable params (min / max / step); only those change."""

import copy

import pytest

import config
from strategies import ALL_STRATEGIES
from strategies.base_strategy import ParamSpec


@pytest.mark.parametrize("S", ALL_STRATEGIES, ids=lambda S: S.__name__)
def test_every_strategy_declares_valid_tunables(S):
    s = S()
    assert s.TUNABLE_PARAMS, f"{s.name} declares no tunable params"
    for name, spec in s.TUNABLE_PARAMS.items():
        assert isinstance(spec, ParamSpec)
        assert name in s.params, f"{s.name}.{name} is not a real param"
        assert spec.min < spec.max and spec.step > 0
        assert spec.min <= s.params[name] <= spec.max, f"{s.name}.{name} default out of range"


def test_values_are_clamped_to_hard_limits():
    s = ALL_STRATEGIES[0]()
    name, spec = next(iter(s.TUNABLE_PARAMS.items()))
    assert s.set_tunable_param(name, spec.max * 10) == spec.max
    assert s.params[name] == spec.max
    assert s.set_tunable_param(name, spec.min - 1000) == spec.min
    assert s.params[name] == spec.min


def test_undeclared_params_cannot_be_tuned():
    s = ALL_STRATEGIES[0]()
    with pytest.raises(KeyError):
        s.set_tunable_param("candle_interval", "1h")


def test_frozen_strategy_rejects_any_change():
    s = ALL_STRATEGIES[0]()
    s.freeze()
    name = next(iter(s.TUNABLE_PARAMS))
    before = dict(s.params)
    with pytest.raises(RuntimeError):
        s.set_tunable_param(name, s.params[name])
    with pytest.raises(RuntimeError):
        s.update_params({name: 1})
    assert s.params == before


def test_restore_only_applies_declared_params_within_limits():
    s = ALL_STRATEGIES[0]()
    name, spec = next(iter(s.TUNABLE_PARAMS.items()))
    s.restore_tunables({name: spec.max + 5, "candle_interval": "1m", "bogus": 1})
    assert s.params[name] == spec.max
    assert s.params["candle_interval"] == ALL_STRATEGIES[0]().params["candle_interval"]
    assert "bogus" not in s.params


@pytest.mark.parametrize("S", ALL_STRATEGIES, ids=lambda S: S.__name__)
def test_custom_params_do_not_mutate_shared_config_defaults(S):
    snapshot = copy.deepcopy(config.STRATEGY_PARAMS)
    s = S()
    name, spec = next(iter(s.TUNABLE_PARAMS.items()))
    S(params={name: spec.max})
    s.clone({name: spec.min})
    assert config.STRATEGY_PARAMS == snapshot
    assert S().params == s.params   # a fresh instance still gets the defaults
