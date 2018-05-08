#!/usr/bin/env python
# coding=utf-8

try:
    import mock

except ImportError:
    import unittest.mock as mock

from marvin_automl_example_engine.training import MetricsEvaluator


class TestMetricsEvaluator:
    def test_execute(self, mocked_params):
        ac = MetricsEvaluator()
        ac.execute(params=mocked_params)
        assert not ac._params