"""Tests for bounded SAREnv network retries around OSMnx queries."""

import geopandas as gpd
import pytest
import requests
import shapely

from sarenv.core.generation import OVERPASS_MAX_WORKERS
from sarenv.core.geometries import GeoPolygon
from sarenv.io import osm_query


def _query_area():
    return GeoPolygon(shapely.box(10.28, 55.14, 10.29, 55.15), crs="EPSG:4326")


def test_query_features_does_not_retry_successful_query(monkeypatch):
    calls = 0
    empty_result = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")

    def succeed(*args, **kwargs):
        nonlocal calls
        calls += 1
        return empty_result

    sleeps = []
    monkeypatch.setattr(osm_query.ox, "features_from_polygon", succeed)
    monkeypatch.setattr(osm_query.time, "sleep", sleeps.append)

    result = osm_query.query_features(_query_area(), {"building": True})

    assert result == {"building": None}
    assert calls == 1
    assert sleeps == []


@pytest.mark.parametrize(
    "network_error",
    [
        requests.exceptions.ConnectTimeout("temporary connect timeout"),
        requests.exceptions.ReadTimeout("temporary read timeout"),
        requests.exceptions.ConnectionError("temporary connection error"),
    ],
)
def test_query_features_retries_transient_network_errors(
    monkeypatch,
    network_error,
):
    calls = 0
    empty_result = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")

    def fail_once_then_succeed(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise network_error
        return empty_result

    sleeps = []
    monkeypatch.setattr(
        osm_query.ox,
        "features_from_polygon",
        fail_once_then_succeed,
    )
    monkeypatch.setattr(osm_query.random, "uniform", lambda low, high: 0.0)
    monkeypatch.setattr(osm_query.time, "sleep", sleeps.append)

    result = osm_query.query_features(_query_area(), {"building": True})

    assert result == {"building": None}
    assert calls == 2
    assert sleeps == [30.0]


def test_query_features_stops_after_three_network_attempts(monkeypatch):
    calls = 0

    def always_fail(*args, **kwargs):
        nonlocal calls
        calls += 1
        raise requests.exceptions.ConnectionError("temporary connection error")

    sleeps = []
    monkeypatch.setattr(
        osm_query.ox,
        "features_from_polygon",
        always_fail,
    )
    monkeypatch.setattr(osm_query.random, "uniform", lambda low, high: high)
    monkeypatch.setattr(osm_query.time, "sleep", sleeps.append)

    result = osm_query.query_features(_query_area(), {"building": True})

    assert result is None
    assert calls == 3
    assert sleeps == [35.0, 70.0]


def test_query_features_does_not_retry_non_network_errors(monkeypatch):
    calls = 0

    def fail_with_value_error(*args, **kwargs):
        nonlocal calls
        calls += 1
        raise ValueError("invalid geometry")

    sleeps = []
    monkeypatch.setattr(
        osm_query.ox,
        "features_from_polygon",
        fail_with_value_error,
    )
    monkeypatch.setattr(osm_query.time, "sleep", sleeps.append)

    result = osm_query.query_features(_query_area(), {"building": True})

    assert result is None
    assert calls == 1
    assert sleeps == []


def test_overpass_worker_limit_matches_controlled_result():
    assert OVERPASS_MAX_WORKERS == 2
