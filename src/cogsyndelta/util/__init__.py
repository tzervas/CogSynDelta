"""Small cross-cutting utilities shared by CSD entry points.

Kept separate from `cogsyndelta.regions` so a tool (e.g. the gpu-pack job runner) can
depend on a thin, stable surface here without pulling in the training-region code.
"""
