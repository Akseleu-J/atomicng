"""Иерархия ошибок atomicng (P2: каждая ошибка объясняет, что делать)."""


class AtomicNGError(Exception):
    """Базовая ошибка atomicng."""


class SpecError(AtomicNGError):
    """Некорректная или несовместимая спецификация таблицы."""


class OverlayError(AtomicNGError):
    """Повреждённый, неподписанный/несовместимый .atomicng-оверлей."""


class GuardViolation(AtomicNGError):
    """Нарушена политика записи (guards). Атрибут violations — список деталей."""

    def __init__(self, message: str, violations: list[str] | None = None) -> None:
        super().__init__(message)
        self.violations = violations or []


class BudgetExceeded(GuardViolation):
    """Исчерпан бюджет оверлея (строки/записи)."""


class BackendError(AtomicNGError):
    """Бэкенд недоступен или модель не поддерживается. Подсказка — в сообщении."""


class PublishRejected(AtomicNGError):
    """Safety-gate отклонил публикацию оверлея (drift/hot-rows/verify не прошли)."""
