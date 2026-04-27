"""OpenTelemetry setup — console-only tracing for the home automation agent."""

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter

# ---------- Provider setup (runs once on import) ----------

resource = Resource.create({"service.name": "home-agent"})

provider = TracerProvider(resource=resource)
provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
trace.set_tracer_provider(provider)

# ---------- Shared tracer ----------

tracer = trace.get_tracer("home-agent")


# ---------- Helpers ----------

def log_llm_io(span: trace.Span, prompt: str, completion: str) -> None:
    """Record LLM prompt and completion as span events for hallucination auditing."""
    span.add_event("llm.prompt", attributes={"llm.prompt": prompt[:4096]})
    span.add_event("llm.completion", attributes={"llm.completion": completion[:4096]})
