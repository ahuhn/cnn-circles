from typing import Any, Dict, Protocol


class KerasModel(Protocol):
    def compile(self, optimizer: Any, loss: Any, metrics: Any) -> Any:
        ...

    def fit(
        self,
        x: Any,
        y: Any,
        epochs: int,
        validation_data: Any,
    ) -> Any:
        ...

    def summary(self) -> Any:
        ...

    def evaluate(
        self,
        x: Any,
        y: Any,
        verbose: int,
    ) -> Any:
        ...


class TFTensor(Protocol):
    pass


class TFShape(Protocol):
    pass


class TFHistory(Protocol):
    history: Dict[str, Any]


class TFConstantMask(Protocol):
    pass
