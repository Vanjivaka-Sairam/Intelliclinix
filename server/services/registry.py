from typing import Callable, Dict

# Type alias for inference runner functions
InferenceRunner = Callable[[str, dict], None]


_RUNNERS: Dict[str, InferenceRunner] = {}


def register_runner(name: str, fn: InferenceRunner) -> None:
    """
    Register an inference runner function under a short name, e.g. "cellpose".

    This is the central dispatch table that allows the API layer to remain
    agnostic of the underlying model implementation.
    """
    if not isinstance(name, str) or not name:
        raise ValueError("Runner name must be a non-empty string")

    if name in _RUNNERS:
        raise ValueError(f"Inference runner '{name}' is already registered")

    _RUNNERS[name] = fn


def get_runner(name: str) -> InferenceRunner:
    """
    Fetch a previously registered inference runner.

    Raises ValueError with a clear message if the name is unknown.
    """
    try:
        return _RUNNERS[name]
    except KeyError:
        raise ValueError(f"No inference runner registered under name '{name}'")


def list_runners() -> Dict[str, InferenceRunner]:
    """
    Return a shallow copy of the registered runners mapping.
    Primarily useful for debugging / introspection.
    """
    return dict(_RUNNERS)


