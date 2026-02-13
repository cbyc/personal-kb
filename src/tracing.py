"""OpenTelemetry tracing setup for Arize Phoenix."""

from src.config import get_settings


def setup_tracing() -> None:
    """Configure OpenTelemetry to export traces to Phoenix.

    Reads tracing_enabled and phoenix_endpoint from settings.
    Does nothing if tracing is disabled.
    """
    settings = get_settings()
    if not settings.tracing_enabled:
        return

    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.trace import set_tracer_provider
    from pydantic_ai import Agent

    tracer_provider = TracerProvider()
    exporter = OTLPSpanExporter(endpoint=settings.phoenix_endpoint)
    tracer_provider.add_span_processor(SimpleSpanProcessor(exporter))
    set_tracer_provider(tracer_provider)

    Agent.instrument_all()
