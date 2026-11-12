"""Exception types for AtomicNG."""


class AtomicNGError(Exception):
    """The basic atomicng error."""


class SpecError(AtomicNGError):
    """Incorrect or incompatible table specification."""


class OverlayError(AtomicNGError):
    """A corrupt, unsigned, or incompatible .atomicng overlay."""


class GuardViolation(AtomicNGError):
    """The security policy is violated.."""

    def __init__(self, message: str, violations: list[str] | None = None) -> None:
        super().__init__(message)
        self.violations = violations or []


class BudgetExceeded(GuardViolation):
    """The overlay row or write budget was exhausted."""


class BackendError(AtomicNGError):
    """Backend unavailable or model unsupported.Hint - in message."""


class PublishRejected(AtomicNGError):
    """Safety-gate rejects publication of overlay (drift/hot-rows/verify failed)."""
