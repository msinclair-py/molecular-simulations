# ruff: noqa: SIM115
"""Custom OpenMM reporters for molecular simulations."""

from pathlib import Path

import numpy as np


class RCReporter:
    """Custom reaction-coordinate reporter for OpenMM. Computes reaction
    coordinate progress for a given frame, and reports the target, rc0,
    current state, rc, and both distances that comprise the reaction
    coordinate, d_ik, d_ij.
    """

    def __init__(
        self, file: Path, report_interval: int, atom_indices: list[int], rc0: float
    ):
        """Initialize the reaction coordinate reporter.

        Args:
            file: Path to the output file for reaction coordinate data.
            report_interval: Number of simulation steps between reports.
            atom_indices: List of three atom indices [i, j, k] defining the
                reaction coordinate as dist(i,k) - dist(j,k).
            rc0: Target reaction coordinate value for the current window.
        """
        self.file = open(file, 'w')
        self.file.write('rc0,rc,dist_ik, dist_jk\n')

        self.report_interval = report_interval
        self.atom_indices = atom_indices
        self.rc0 = rc0

    def __del__(self):
        """Close the output file when the reporter is destroyed."""
        self.file.close()

    def describeNextReport(self, simulation):
        """Describe when the next report will be generated.

        Args:
            simulation: The OpenMM Simulation object being reported on.

        Returns:
            A tuple containing (steps_until_next_report, need_positions,
            need_velocities, need_forces, need_energy, wrapped_positions).
        """
        steps = self.report_interval - simulation.currentStep % self.report_interval
        return (steps, True, False, False, False, None)

    def report(self, simulation, state):
        """Generate a report for the current simulation state.

        Computes the reaction coordinate as the difference between two
        distances and writes the target rc0, current rc, and individual
        distances to the output file.

        Args:
            simulation: The OpenMM Simulation object being reported on.
            state: The current State of the simulation containing positions.
        """
        pos = state.getPositions(asNumpy=True)

        i, j, k = self.atom_indices
        dist_ik = np.linalg.norm(pos[i] - pos[k])
        dist_jk = np.linalg.norm(pos[j] - pos[k])

        rc = dist_ik - dist_jk

        self.file.write(f'{self.rc0},{rc},{dist_ik},{dist_jk}\n')
        self.file.flush()
