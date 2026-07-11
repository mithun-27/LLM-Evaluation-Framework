from abc import ABC, abstractmethod
from typing import Any, Dict


class BaseMetric(ABC):
    """
    Abstract base class for all evaluation metrics.
    """

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def calculate(
        self,
        reference: str,
        prediction: str,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Calculate the evaluation metric.
        """
        pass

    def __str__(self) -> str:
        return self.name