from abc import ABC, abstractmethod


class CurriculumScheduler(ABC):
    """Abstract base for curriculum schedulers.

    Subclasses implement step() to decide when to advance to the next phase.
    """

    def __init__(self, phases: list[dict]):
        self.phases = phases
        self.current_phase = 0

    @abstractmethod
    def step(self, metrics: dict) -> dict | None:
        """Called after each episode. Returns next phase params or None."""
        ...

    @property
    def is_at_last_phase(self) -> bool:
        return self.current_phase >= len(self.phases) - 1
