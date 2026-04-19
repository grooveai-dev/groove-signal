"""Unit tests for layer coverage validation and relay pipeline assembly."""

import pytest

from src.relay.relay import RelayNode, get_node_for_layer, validate_layer_coverage


def test_validate_coverage_full():
    nodes = [
        {"node_id": "a", "layer_start": 0, "layer_end": 15},
        {"node_id": "b", "layer_start": 16, "layer_end": 31},
    ]
    assert validate_layer_coverage(nodes, 32) is True


def test_validate_coverage_gap():
    nodes = [
        {"node_id": "a", "layer_start": 0, "layer_end": 14},
        {"node_id": "b", "layer_start": 16, "layer_end": 31},
    ]
    assert validate_layer_coverage(nodes, 32) is False


def test_validate_coverage_missing_start():
    nodes = [
        {"node_id": "a", "layer_start": 5, "layer_end": 31},
    ]
    assert validate_layer_coverage(nodes, 32) is False


def test_validate_coverage_missing_end():
    nodes = [
        {"node_id": "a", "layer_start": 0, "layer_end": 20},
    ]
    assert validate_layer_coverage(nodes, 32) is False


def test_validate_coverage_three_nodes():
    nodes = [
        {"node_id": "a", "layer_start": 0, "layer_end": 10},
        {"node_id": "b", "layer_start": 11, "layer_end": 21},
        {"node_id": "c", "layer_start": 22, "layer_end": 31},
    ]
    assert validate_layer_coverage(nodes, 32) is True


def test_validate_coverage_single_node():
    nodes = [
        {"node_id": "a", "layer_start": 0, "layer_end": 31},
    ]
    assert validate_layer_coverage(nodes, 32) is True


def test_validate_coverage_empty():
    assert validate_layer_coverage([], 32) is False


def test_get_node_for_layer():
    nodes = [
        {"node_id": "a", "layer_start": 0, "layer_end": 15},
        {"node_id": "b", "layer_start": 16, "layer_end": 31},
    ]
    assert get_node_for_layer(nodes, 0)["node_id"] == "a"
    assert get_node_for_layer(nodes, 15)["node_id"] == "a"
    assert get_node_for_layer(nodes, 16)["node_id"] == "b"
    assert get_node_for_layer(nodes, 31)["node_id"] == "b"
    assert get_node_for_layer(nodes, 32) is None


def test_relay_assemble_no_nodes():
    relay = RelayNode()
    with pytest.raises(RuntimeError, match="No active compute nodes"):
        relay.assemble_pipeline("test-model", "sess-1")
