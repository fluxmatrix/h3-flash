"""Inference backend adapters."""

from .official_diffusers import OfficialDiffusersBackend, OfficialGenerationResult

__all__ = ["OfficialDiffusersBackend", "OfficialGenerationResult"]
