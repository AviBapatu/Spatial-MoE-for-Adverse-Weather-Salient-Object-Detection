"""MoE routing diagnostics: collection, aggregation, and reporting.

The public surface preserves the historical ``src.diagnostics`` module exports
so existing callers (training loop, ``run_moe_diagnostics.py``, tests) keep
working unchanged:

- ``RoutingTracker`` / ``WeatherAnalyzer`` — rank-local collectors.
- ``Visualizer`` / ``MoEDiagnosticsEngine`` / ``ExpertSimilarityAnalyzer`` —
  reporting and analysis.
- ``routing_summary`` / ``merge_states`` / ``collapse_warnings`` /
  ``weather_divergence`` — pure aggregation helpers.
"""

from src.diagnostics.aggregation import (
    collapse_warnings,
    merge_states,
    routing_summary,
    weather_divergence,
)
from src.diagnostics.collectors import RoutingTracker, WeatherAnalyzer
from src.diagnostics.reporting import (
    ExpertSimilarityAnalyzer,
    MoEDiagnosticsEngine,
    Visualizer,
    build_scale_pad_mask,
)

__all__ = [
    "RoutingTracker",
    "WeatherAnalyzer",
    "Visualizer",
    "MoEDiagnosticsEngine",
    "ExpertSimilarityAnalyzer",
    "build_scale_pad_mask",
    "routing_summary",
    "merge_states",
    "collapse_warnings",
    "weather_divergence",
]
