# AI-Assisted CFD Solver Diagnostics & Post-Processing

A physics-aware Python framework for CFD post-processing, solver diagnostics,
and convergence validation using residuals, forces, and probes.

## Why this project?
CFD users often rely on residual plots alone to judge convergence.
In practice, residuals can be misleading and hide pseudo-convergence.

This project demonstrates a solver-engineer approach to CFD validation using:
- Residual decay analysis
- Force convergence checks
- Physics-aware solver verdict logic
