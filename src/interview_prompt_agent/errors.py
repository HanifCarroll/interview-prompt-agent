"""Domain-specific exceptions."""


class AgentError(RuntimeError):
    """Base error for expected runtime failures."""


class DependencyMissingError(AgentError):
    """A selected backend needs an optional package or binary that is missing."""


class BackendUnavailableError(AgentError):
    """A selected backend is installed but cannot run in the current environment."""
