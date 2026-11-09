"""
PPOEM engine: detection of ML model loading in Python projects and
project-wide static resolution of model names.

    from ppoem.extractor import extract_project
    result = extract_project(Path("path/to/cloned/repo"))
    result.literal_string, result.variable_based, result.excluded, result.trace
"""

from .extractor import EXTRACTOR_VERSION, Extraction, ProjectExtractor, extract_project  # noqa: F401
