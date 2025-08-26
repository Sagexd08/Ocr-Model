"""
Structured logging helpers.
Provides a LoggerAdapter that injects job_id and page into log records.
"""
import logging
from typing import Optional, Mapping, Any

class ContextLoggerAdapter(logging.LoggerAdapter):
    """Logger adapter that ensures job_id and page keys exist in every record."""

    def __init__(self, logger: logging.Logger, extra: Optional[Mapping[str, Any]] = None):
        super().__init__(logger, dict(extra or {}))
        # Normalize keys
        if "job_id" not in self.extra:
            self.extra["job_id"] = None
        if "page" not in self.extra:
            self.extra["page"] = None

    def set_job(self, job_id: Optional[str]):
        self.extra["job_id"] = job_id

    def set_page(self, page: Optional[int]):
        self.extra["page"] = page

    def process(self, msg, kwargs):
        extra = kwargs.get("extra", {})
        # Merge adapter extra with call extra; call extra wins
        merged = {**self.extra, **extra}
        kwargs["extra"] = merged
        # Prefix message with minimal context for plain formatters
        prefix = []
        if merged.get("job_id"):
            prefix.append(f"job_id={merged['job_id']}")
        if merged.get("page") is not None:
            prefix.append(f"page={merged['page']}")
        if prefix:
            msg = f"{' '.join(prefix)} {msg}"
        return msg, kwargs


def get_context_logger(name: str, job_id: Optional[str] = None, page: Optional[int] = None) -> ContextLoggerAdapter:
    base = logging.getLogger(name)
    adapter = ContextLoggerAdapter(base, {"job_id": job_id, "page": page})
    return adapter

