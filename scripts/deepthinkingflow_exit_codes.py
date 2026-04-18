#!/usr/bin/env python3
"""Shared exit codes for DeepThinkingFlow CLI-facing scripts."""

from __future__ import annotations

OK = 0
USAGE_ERROR = 2
MISSING_DEPENDENCY = 3
INVALID_ARTIFACT = 4
INCOMPATIBLE_RUNTIME = 5
PRECONDITION_FAILED = 6
VERIFICATION_FAILED = 7


EXIT_CODE_LABELS = {
    OK: "ok",
    USAGE_ERROR: "usage_error",
    MISSING_DEPENDENCY: "missing_dependency",
    INVALID_ARTIFACT: "invalid_artifact",
    INCOMPATIBLE_RUNTIME: "incompatible_runtime",
    PRECONDITION_FAILED: "precondition_failed",
    VERIFICATION_FAILED: "verification_failed",
}
