"""Tests for the metrics registry, helpers, and HTTP endpoints."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from mnemo import metrics as metrics_mod
from mnemo.server import app


@pytest.fixture(autouse=True)
def _reset_registry():
    """Isolate each test by resetting the global metrics registry."""
    metrics_mod.REGISTRY.reset()
    yield
    metrics_mod.REGISTRY.reset()


# ---------------------------------------------------------------------------
# Registry unit tests
# ---------------------------------------------------------------------------

class TestRegistry:
    def test_counter_increments(self):
        metrics_mod.REGISTRY.inc("test_counter")
        metrics_mod.REGISTRY.inc("test_counter")
        assert metrics_mod.REGISTRY.counter_value("test_counter") == 2.0

    def test_counter_with_labels(self):
        metrics_mod.REGISTRY.inc("req", endpoint="/v1/chat", status="200")
        metrics_mod.REGISTRY.inc("req", endpoint="/v1/chat", status="200")
        metrics_mod.REGISTRY.inc("req", endpoint="/health", status="200")
        assert metrics_mod.REGISTRY.counter_value("req", endpoint="/v1/chat", status="200") == 2.0
        assert metrics_mod.REGISTRY.counter_value("req", endpoint="/health", status="200") == 1.0

    def test_histogram_summary(self):
        for v in [10.0, 20.0, 30.0, 40.0, 50.0]:
            metrics_mod.REGISTRY.observe("latency", v)
        snap = metrics_mod.REGISTRY.snapshot()
        summary = snap["histograms"]["latency"]
        assert summary["count"] == 5
        assert summary["min"] == 10.0
        assert summary["max"] == 50.0
        assert summary["p50"] == 30.0

    def test_reset_clears_state(self):
        metrics_mod.REGISTRY.inc("x")
        metrics_mod.REGISTRY.observe("y", 1.0)
        metrics_mod.REGISTRY.reset()
        snap = metrics_mod.REGISTRY.snapshot()
        assert snap["counters"] == {}
        assert snap["histograms"] == {}


# ---------------------------------------------------------------------------
# Named helper tests
# ---------------------------------------------------------------------------

class TestNamedHelpers:
    def test_record_request(self):
        metrics_mod.record_request("/v1/chat", 200, 42.5)
        snap = metrics_mod.REGISTRY.snapshot()
        assert snap["counters"]['mnemo_requests_total{endpoint="/v1/chat",status="200"}'] == 1.0
        assert snap["histograms"]['mnemo_request_duration_ms{endpoint="/v1/chat"}']['count'] == 1

    def test_record_tokens_saved_positive(self):
        metrics_mod.record_tokens_saved(512)
        assert metrics_mod.REGISTRY.counter_value("mnemo_tokens_saved_total") == 512.0

    def test_record_tokens_saved_zero_ignored(self):
        metrics_mod.record_tokens_saved(0)
        assert metrics_mod.REGISTRY.counter_value("mnemo_tokens_saved_total") == 0.0

    def test_record_memory_retrieval_hits(self):
        metrics_mod.record_memory_retrieval(hits=5)
        assert metrics_mod.REGISTRY.counter_value("mnemo_memory_hits_total") == 5.0
        assert metrics_mod.REGISTRY.counter_value("mnemo_memory_misses_total") == 0.0

    def test_record_memory_retrieval_misses(self):
        metrics_mod.record_memory_retrieval(hits=0, misses=1)
        assert metrics_mod.REGISTRY.counter_value("mnemo_memory_misses_total") == 1.0

    def test_record_memory_write(self):
        metrics_mod.record_memory_write("fact", 3)
        metrics_mod.record_memory_write("triple", 2)
        snap = metrics_mod.REGISTRY.snapshot()
        assert snap["counters"]['mnemo_memories_written_total{kind="fact"}'] == 3.0
        assert snap["counters"]['mnemo_memories_written_total{kind="triple"}'] == 2.0

    def test_record_compaction(self):
        metrics_mod.record_compaction(original_rows=40, new_rows=12)
        assert metrics_mod.REGISTRY.counter_value("mnemo_compactions_total") == 1.0
        assert metrics_mod.REGISTRY.counter_value("mnemo_compaction_rows_removed_total") == 28.0

    def test_record_compaction_no_shrinkage(self):
        metrics_mod.record_compaction(original_rows=5, new_rows=5)
        assert metrics_mod.REGISTRY.counter_value("mnemo_compaction_rows_removed_total") == 0.0


# ---------------------------------------------------------------------------
# Token estimation helpers
# ---------------------------------------------------------------------------

class TestTokenEstimation:
    def test_approx_tokens_from_text(self):
        assert metrics_mod.approx_tokens_from_text("abcd") == 1
        assert metrics_mod.approx_tokens_from_text("") == 0
        assert metrics_mod.approx_tokens_from_text("a" * 400) == 100

    def test_approx_tokens_multiple_parts(self):
        assert metrics_mod.approx_tokens_from_text("abcd", "efgh") == 2

    def test_approx_tokens_chat_messages(self):
        msgs = [{"role": "user", "content": "a" * 400}, {"role": "assistant", "content": "b" * 400}]
        assert metrics_mod.approx_tokens_chat_messages(msgs) == 200


# ---------------------------------------------------------------------------
# Prometheus text exporter
# ---------------------------------------------------------------------------

class TestPrometheusExporter:
    def test_empty_registry_returns_empty_string(self):
        output = metrics_mod.metrics_prometheus()
        assert output == ""

    def test_counter_in_output(self):
        metrics_mod.record_request("/health", 200, 5.0)
        output = metrics_mod.metrics_prometheus()
        assert "# TYPE mnemo_requests_total counter" in output
        assert "mnemo_requests_total" in output
        assert "200" in output

    def test_histogram_quantiles_in_output(self):
        for v in [10.0, 50.0, 100.0]:
            metrics_mod.record_request("/v1/chat", 200, v)
        output = metrics_mod.metrics_prometheus()
        assert 'quantile="0.5"' in output
        assert 'quantile="0.95"' in output
        assert 'quantile="0.99"' in output
        assert "mnemo_request_duration_ms_count" in output
        assert "mnemo_request_duration_ms_sum" in output

    def test_help_lines_present(self):
        metrics_mod.record_tokens_saved(100)
        output = metrics_mod.metrics_prometheus()
        assert "# HELP mnemo_tokens_saved_total" in output


# ---------------------------------------------------------------------------
# JSON exporter
# ---------------------------------------------------------------------------

class TestJsonExporter:
    def test_json_structure(self):
        metrics_mod.record_request("/v1/chat", 200, 30.0)
        metrics_mod.record_tokens_saved(100)
        data = metrics_mod.metrics_json()
        assert "counters" in data
        assert "histograms" in data

    def test_json_counter_value(self):
        metrics_mod.record_tokens_saved(250)
        data = metrics_mod.metrics_json()
        assert data["counters"]["mnemo_tokens_saved_total"] == 250.0

    def test_json_histogram_has_percentiles(self):
        for v in [10.0, 20.0, 30.0]:
            metrics_mod.record_request("/v1/chat", 200, v)
        data = metrics_mod.metrics_json()
        hist = data["histograms"]['mnemo_request_duration_ms{endpoint="/v1/chat"}']
        assert "p50" in hist
        assert "p95" in hist
        assert "p99" in hist
        assert hist["count"] == 3


# ---------------------------------------------------------------------------
# HTTP endpoint tests
# ---------------------------------------------------------------------------

class TestMetricsEndpoints:
    def test_prometheus_endpoint_accessible_without_auth(self):
        with TestClient(app) as client:
            r = client.get("/metrics")
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/plain")

    def test_prometheus_endpoint_content_after_request(self):
        with TestClient(app) as client:
            client.get("/health")
            r = client.get("/metrics")
        assert "mnemo_requests_total" in r.text
        assert "/health" in r.text

    def test_json_metrics_endpoint(self):
        with TestClient(app) as client:
            client.get("/health")
            r = client.get("/v1/metrics")
        assert r.status_code == 200
        data = r.json()
        assert "counters" in data
        assert "histograms" in data

    def test_prometheus_endpoint_shows_histogram(self):
        with TestClient(app) as client:
            client.get("/health")
            client.get("/health")
            r = client.get("/metrics")
        assert "mnemo_request_duration_ms" in r.text
        assert 'quantile="0.5"' in r.text

    def test_multiple_requests_accumulate(self):
        with TestClient(app) as client:
            for _ in range(5):
                client.get("/health")
            r = client.get("/v1/metrics")
        data = r.json()
        health_key = 'mnemo_requests_total{endpoint="/health",status="200"}'
        assert data["counters"].get(health_key, 0) >= 5
