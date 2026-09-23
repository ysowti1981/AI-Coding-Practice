from __future__ import annotations

import logging

from models import Span
from transformer import SpanTransformer
from validator import SpanValidator

logger = logging.getLogger(__name__)


class SpanIngestor:
    """Validates and transforms raw OTel span dicts into typed Span objects."""

    def __init__(
        self,
        validator: SpanValidator | None = None,
        transformer: SpanTransformer | None = None,
    ) -> None:
        self._validator = validator or SpanValidator()
        self._transformer = transformer or SpanTransformer()

    def ingest(self, raw_spans: list[dict]) -> tuple[list[Span], list[dict]]:
        """
        Process a batch of raw span dicts.

        Returns:
            (accepted, rejected) where rejected items are
            {"raw": dict, "errors": list[str]}.
        """
        accepted: list[Span] = []
        rejected: list[dict] = []

        for raw in raw_spans:
            errors = self._validator.validate(raw)
            if errors:
                rejected.append({"raw": raw, "errors": errors})
                logger.warning("Rejected span: %s", errors)
                continue
            span = self._transformer.transform(raw)
            accepted.append(span)

        return accepted, rejected
