"""B3: frozen baseline copies keep default params even when the learner has learned values."""

import json

import database as db
from strategies import ALL_STRATEGIES


def test_baselines_are_frozen_defaults_while_learners_restore_learned_values(temp_db):
    import main
    learners = [S() for S in ALL_STRATEGIES]
    target = learners[0]
    param, spec = next(iter(target.TUNABLE_PARAMS.items()))
    learned = spec.clamp(target.params[param] + spec.step)
    db.set_meta(f"learned_params:{target.name}", json.dumps({param: learned}))

    main.restore_learned_params(learners)
    baselines = main.build_baselines({s.name for s in learners})

    assert target.params[param] == learned
    for b in baselines:
        assert b.frozen and b.is_active
        assert b.params == type(b)().params            # untouched defaults
    assert baselines[0] is not target                   # separate instances


def test_only_active_strategies_get_an_active_baseline():
    import main
    names = {ALL_STRATEGIES[0]().name}
    baselines = main.build_baselines(names)
    assert {b.name for b in baselines if b.is_active} == names
