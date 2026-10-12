"""Classifier backend errors shared without registry import cycles."""


class UnavailableClassifierError(ValueError):
    """A named paper classifier cannot run because its backend is absent."""
