"""
Unit tests for simulate/reporters.py module

This module tests the custom OpenMM reporters used for tracking
reaction coordinates during EVB simulations. All tests drive the reporter
with real OpenMM Simulation/State objects (Reference platform) -- no mocks.
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest

# Mark tests that don't require OpenMM as unit tests
pytestmark = pytest.mark.unit


def _check_openmm() -> bool:
    """Return True if OpenMM and its Reference platform are available."""
    try:
        from openmm import Platform

        Platform.getPlatformByName('Reference')
        return True
    except Exception:
        return False


requires_openmm = pytest.mark.skipif(not _check_openmm(), reason='OpenMM not available')


def _make_state(positions_nm):
    """Build a real OpenMM State whose positions are ``positions_nm``.

    A force-free System with one particle per supplied coordinate is built on
    the Reference platform; the positions (interpreted as nanometers) are set
    and a real State carrying them is returned. This is exactly what an OpenMM
    reporter receives at report time.
    """
    from openmm import Context, Platform, System, VerletIntegrator
    from openmm.unit import nanometer, picosecond

    system = System()
    for _ in positions_nm:
        system.addParticle(1.0)

    context = Context(
        system,
        VerletIntegrator(0.001 * picosecond),
        Platform.getPlatformByName('Reference'),
    )
    context.setPositions(np.array(positions_nm, dtype=float) * nanometer)
    return context.getState(getPositions=True)


def _make_simulation(current_step: int, n_atoms: int = 3):
    """Build a real OpenMM Simulation with ``currentStep`` preset.

    OpenMM's ``Simulation.currentStep`` is a plain settable attribute, so this
    gives ``describeNextReport`` a genuine Simulation object to interrogate.
    """
    from openmm import Platform, System, VerletIntegrator
    from openmm.app import Element, Simulation, Topology
    from openmm.unit import picosecond

    top = Topology()
    chain = top.addChain()
    res = top.addResidue('UNK', chain)
    carbon = Element.getBySymbol('C')

    system = System()
    for _ in range(n_atoms):
        top.addAtom('C', carbon, res)
        system.addParticle(12.0)

    sim = Simulation(
        top,
        system,
        VerletIntegrator(0.001 * picosecond),
        Platform.getPlatformByName('Reference'),
    )
    sim.currentStep = current_step
    return sim


class TestRCReporterInit:
    """Test suite for RCReporter class initialization."""

    def test_rc_reporter_init_creates_file(self) -> None:
        """Test RCReporter initialization creates output file with header.

        The reporter should create a CSV file with columns:
        rc0, rc, dist_ik, dist_jk
        """
        from molecular_simulations.simulate.reporters import RCReporter

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            rc_file = path / 'rc_test.log'

            reporter = RCReporter(
                file=rc_file,
                report_interval=10,
                atom_indices=[0, 1, 2],
                rc0=0.1,
            )

            # File should exist and have header
            assert rc_file.exists()

            # Close to flush
            reporter.file.close()

            content = rc_file.read_text()
            assert 'rc0,rc,dist_ik, dist_jk' in content

    def test_rc_reporter_stores_parameters(self) -> None:
        """Test RCReporter stores initialization parameters correctly."""
        from molecular_simulations.simulate.reporters import RCReporter

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            rc_file = path / 'rc_test.log'

            reporter = RCReporter(
                file=rc_file,
                report_interval=25,
                atom_indices=[10, 20, 30],
                rc0=0.15,
            )

            assert reporter.report_interval == 25
            assert reporter.atom_indices == [10, 20, 30]
            assert reporter.rc0 == 0.15

            reporter.file.close()

    def test_rc_reporter_file_is_writable(self) -> None:
        """Test RCReporter opens file in write mode."""
        from molecular_simulations.simulate.reporters import RCReporter

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            rc_file = path / 'rc_test.log'

            reporter = RCReporter(
                file=rc_file,
                report_interval=10,
                atom_indices=[0, 1, 2],
                rc0=0.1,
            )

            # Should be able to write to file
            assert reporter.file.writable()

            reporter.file.close()


@requires_openmm
class TestRCReporterDescribeNextReport:
    """Test suite for describeNextReport method, driven by a real Simulation."""

    def test_describe_next_report_returns_correct_tuple(self) -> None:
        """Test describeNextReport returns proper tuple for OpenMM.

        OpenMM expects a tuple of (steps, positions, velocities, forces,
        energies, wrapPositions) to determine what data to collect.
        """
        from molecular_simulations.simulate.reporters import RCReporter

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            rc_file = path / 'rc_test.log'

            reporter = RCReporter(
                file=rc_file,
                report_interval=100,
                atom_indices=[0, 1, 2],
                rc0=0.1,
            )

            simulation = _make_simulation(current_step=50)

            result = reporter.describeNextReport(simulation)

            # Should return 6-tuple
            assert len(result) == 6

            # First element is steps until next report
            steps_until_report = result[0]
            assert steps_until_report == 50  # 100 - 50

            # Second element should be True (needs positions)
            assert result[1] is True

            # Others should be False (no velocities, forces, energies)
            assert result[2] is False
            assert result[3] is False
            assert result[4] is False

            reporter.file.close()

    def test_describe_next_report_at_interval_boundary(self) -> None:
        """Test describeNextReport when currentStep is at interval boundary."""
        from molecular_simulations.simulate.reporters import RCReporter

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            rc_file = path / 'rc_test.log'

            reporter = RCReporter(
                file=rc_file,
                report_interval=100,
                atom_indices=[0, 1, 2],
                rc0=0.1,
            )

            simulation = _make_simulation(current_step=100)

            result = reporter.describeNextReport(simulation)

            # At boundary, should report at next interval
            steps_until_report = result[0]
            assert steps_until_report == 100

            reporter.file.close()

    @pytest.mark.parametrize(
        'current_step,interval,expected',
        [
            (0, 10, 10),
            (5, 10, 5),
            (9, 10, 1),
            (10, 10, 10),
            (11, 10, 9),
            (99, 100, 1),
            (150, 100, 50),
        ],
    )
    def test_describe_next_report_various_steps(
        self, current_step: int, interval: int, expected: int
    ) -> None:
        """Test describeNextReport calculates correct steps for various inputs."""
        from molecular_simulations.simulate.reporters import RCReporter

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            rc_file = path / 'rc_test.log'

            reporter = RCReporter(
                file=rc_file,
                report_interval=interval,
                atom_indices=[0, 1, 2],
                rc0=0.1,
            )

            simulation = _make_simulation(current_step=current_step)

            result = reporter.describeNextReport(simulation)
            assert result[0] == expected

            reporter.file.close()


@requires_openmm
class TestRCReporterReport:
    """Test suite for report method, driven by real OpenMM States."""

    def test_report_writes_correct_format(self) -> None:
        """Test report writes data in correct CSV format.

        Format: rc0, rc, dist_ik, dist_jk
        where rc = dist_ik - dist_jk
        """
        from molecular_simulations.simulate.reporters import RCReporter

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            rc_file = path / 'rc_test.log'

            reporter = RCReporter(
                file=rc_file,
                report_interval=10,
                atom_indices=[0, 1, 2],
                rc0=0.1,
            )

            simulation = _make_simulation(current_step=0)

            # Positions: atom 0 at origin, atom 1 at (1,0,0), atom 2 at (0.5,0,0)
            state = _make_state(
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.5, 0.0, 0.0],
                ]
            )

            reporter.report(simulation, state)
            reporter.file.close()

            content = rc_file.read_text()
            lines = content.strip().split('\n')

            # Should have header + 1 data line
            assert len(lines) == 2

            # Parse data line
            data_line = lines[1]
            values = data_line.split(',')

            assert len(values) == 4

            rc0 = float(values[0])
            rc = float(values[1])
            dist_ik = float(values[2])
            dist_jk = float(values[3])

            assert rc0 == 0.1

            # dist_ik = distance(atom_i, atom_k) = distance(0, 2) = 0.5
            # dist_jk = distance(atom_j, atom_k) = distance(1, 2) = 0.5
            assert dist_ik == pytest.approx(0.5, rel=1e-5)
            assert dist_jk == pytest.approx(0.5, rel=1e-5)

            # rc = dist_ik - dist_jk = 0.5 - 0.5 = 0.0
            assert rc == pytest.approx(0.0, rel=1e-5)

    def test_report_multiple_calls(self) -> None:
        """Test report correctly appends multiple data lines."""
        from molecular_simulations.simulate.reporters import RCReporter

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            rc_file = path / 'rc_test.log'

            reporter = RCReporter(
                file=rc_file,
                report_interval=10,
                atom_indices=[0, 1, 2],
                rc0=0.1,
            )

            simulation = _make_simulation(current_step=0)

            # Report 3 times with different positions
            for i in range(3):
                state = _make_state(
                    [
                        [0.0, 0.0, 0.0],
                        [float(i + 1), 0.0, 0.0],
                        [0.5, 0.0, 0.0],
                    ]
                )
                reporter.report(simulation, state)

            reporter.file.close()

            content = rc_file.read_text()
            lines = content.strip().split('\n')

            # Should have header + 3 data lines
            assert len(lines) == 4

    def test_report_flushes_buffer(self) -> None:
        """Test report flushes file buffer after writing.

        The data is observable on disk before the file is closed, proving the
        reporter flushed.
        """
        from molecular_simulations.simulate.reporters import RCReporter

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            rc_file = path / 'rc_test.log'

            reporter = RCReporter(
                file=rc_file,
                report_interval=10,
                atom_indices=[0, 1, 2],
                rc0=0.1,
            )

            simulation = _make_simulation(current_step=0)
            state = _make_state(
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.5, 0.0, 0.0],
                ]
            )

            reporter.report(simulation, state)

            # Read the file WITHOUT closing the reporter: a data line is only
            # visible here if report() flushed the buffer.
            content = Path(rc_file).read_text()
            lines = content.strip().split('\n')
            assert len(lines) == 2  # header + flushed data line

            reporter.file.close()


@requires_openmm
class TestRCReporterDistanceCalculations:
    """Test suite for distance difference calculations from real States."""

    def test_distance_calculation_simple(self) -> None:
        """Test distance calculation with simple geometry.

        Tests the reaction coordinate rc = dist(i,k) - dist(j,k)
        for atoms arranged in a line.
        """
        from molecular_simulations.simulate.reporters import RCReporter

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            rc_file = path / 'rc_test.log'

            # Atoms: i=0, j=1, k=2
            reporter = RCReporter(
                file=rc_file,
                report_interval=10,
                atom_indices=[0, 1, 2],
                rc0=0.0,
            )

            simulation = _make_simulation(current_step=0)

            # Atom k is between i and j
            # i at (0,0,0), k at (0.3,0,0), j at (1,0,0)
            # dist_ik = 0.3, dist_jk = 0.7
            # rc = 0.3 - 0.7 = -0.4
            state = _make_state(
                [
                    [0.0, 0.0, 0.0],  # i
                    [1.0, 0.0, 0.0],  # j
                    [0.3, 0.0, 0.0],  # k
                ]
            )

            reporter.report(simulation, state)
            reporter.file.close()

            content = rc_file.read_text()
            lines = content.strip().split('\n')
            values = lines[1].split(',')

            rc = float(values[1])
            dist_ik = float(values[2])
            dist_jk = float(values[3])

            assert dist_ik == pytest.approx(0.3, rel=1e-5)
            assert dist_jk == pytest.approx(0.7, rel=1e-5)
            assert rc == pytest.approx(-0.4, rel=1e-5)

    def test_distance_calculation_3d(self) -> None:
        """Test distance calculation in 3D space."""
        from molecular_simulations.simulate.reporters import RCReporter

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            rc_file = path / 'rc_test.log'

            reporter = RCReporter(
                file=rc_file,
                report_interval=10,
                atom_indices=[0, 1, 2],
                rc0=0.0,
            )

            simulation = _make_simulation(current_step=0)

            # 3D positions
            # i at origin, j at (1,1,1), k at (0.5,0.5,0.5)
            state = _make_state(
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 1.0, 1.0],
                    [0.5, 0.5, 0.5],
                ]
            )

            reporter.report(simulation, state)
            reporter.file.close()

            content = rc_file.read_text()
            lines = content.strip().split('\n')
            values = lines[1].split(',')

            dist_ik = float(values[2])
            dist_jk = float(values[3])
            rc = float(values[1])

            # dist_ik = sqrt(0.5^2 + 0.5^2 + 0.5^2) = sqrt(0.75) ~ 0.866
            expected_dist = np.sqrt(0.75)
            assert dist_ik == pytest.approx(expected_dist, rel=1e-5)
            assert dist_jk == pytest.approx(expected_dist, rel=1e-5)

            # rc = 0 since k is equidistant from i and j
            assert rc == pytest.approx(0.0, rel=1e-5)

    def test_distance_positive_rc(self) -> None:
        """Test positive reaction coordinate when k is closer to j."""
        from molecular_simulations.simulate.reporters import RCReporter

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            rc_file = path / 'rc_test.log'

            reporter = RCReporter(
                file=rc_file,
                report_interval=10,
                atom_indices=[0, 1, 2],
                rc0=0.0,
            )

            simulation = _make_simulation(current_step=0)

            # k is closer to j than to i
            # i at (0,0,0), j at (1,0,0), k at (0.8,0,0)
            # dist_ik = 0.8, dist_jk = 0.2
            # rc = 0.8 - 0.2 = 0.6
            state = _make_state(
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.8, 0.0, 0.0],
                ]
            )

            reporter.report(simulation, state)
            reporter.file.close()

            content = rc_file.read_text()
            lines = content.strip().split('\n')
            values = lines[1].split(',')

            rc = float(values[1])
            assert rc == pytest.approx(0.6, rel=1e-5)

    def test_distance_negative_rc(self) -> None:
        """Test negative reaction coordinate when k is closer to i."""
        from molecular_simulations.simulate.reporters import RCReporter

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            rc_file = path / 'rc_test.log'

            reporter = RCReporter(
                file=rc_file,
                report_interval=10,
                atom_indices=[0, 1, 2],
                rc0=0.0,
            )

            simulation = _make_simulation(current_step=0)

            # k is closer to i than to j
            # i at (0,0,0), j at (1,0,0), k at (0.2,0,0)
            # dist_ik = 0.2, dist_jk = 0.8
            # rc = 0.2 - 0.8 = -0.6
            state = _make_state(
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.2, 0.0, 0.0],
                ]
            )

            reporter.report(simulation, state)
            reporter.file.close()

            content = rc_file.read_text()
            lines = content.strip().split('\n')
            values = lines[1].split(',')

            rc = float(values[1])
            assert rc == pytest.approx(-0.6, rel=1e-5)


class TestRCReporterCleanup:
    """Test suite for RCReporter cleanup behavior."""

    def test_destructor_closes_file(self) -> None:
        """Test that __del__ closes the file handle."""
        from molecular_simulations.simulate.reporters import RCReporter

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            rc_file = path / 'rc_test.log'

            reporter = RCReporter(
                file=rc_file,
                report_interval=10,
                atom_indices=[0, 1, 2],
                rc0=0.1,
            )

            # Get file handle reference
            file_handle = reporter.file
            assert not file_handle.closed

            # Explicitly call destructor
            reporter.__del__()

            assert file_handle.closed


@requires_openmm
class TestRCReporterEdgeCases:
    """Test suite for edge cases and error conditions."""

    def test_zero_distance(self) -> None:
        """Test handling when atoms are at the same position."""
        from molecular_simulations.simulate.reporters import RCReporter

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            rc_file = path / 'rc_test.log'

            reporter = RCReporter(
                file=rc_file,
                report_interval=10,
                atom_indices=[0, 1, 2],
                rc0=0.0,
            )

            simulation = _make_simulation(current_step=0)

            # All atoms at same position
            state = _make_state(
                [
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                ]
            )

            # Should not raise
            reporter.report(simulation, state)
            reporter.file.close()

            content = rc_file.read_text()
            lines = content.strip().split('\n')
            values = lines[1].split(',')

            # All distances should be 0
            assert float(values[1]) == 0.0  # rc
            assert float(values[2]) == 0.0  # dist_ik
            assert float(values[3]) == 0.0  # dist_jk

    def test_large_coordinates(self) -> None:
        """Test handling of large coordinate values."""
        from molecular_simulations.simulate.reporters import RCReporter

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            rc_file = path / 'rc_test.log'

            reporter = RCReporter(
                file=rc_file,
                report_interval=10,
                atom_indices=[0, 1, 2],
                rc0=0.0,
            )

            simulation = _make_simulation(current_step=0)

            # Large coordinate values (e.g., far from origin)
            state = _make_state(
                [
                    [1000.0, 1000.0, 1000.0],
                    [1001.0, 1000.0, 1000.0],
                    [1000.5, 1000.0, 1000.0],
                ]
            )

            reporter.report(simulation, state)
            reporter.file.close()

            content = rc_file.read_text()
            lines = content.strip().split('\n')
            values = lines[1].split(',')

            # Distances should be correct despite large absolute positions
            assert float(values[2]) == pytest.approx(0.5, rel=1e-5)  # dist_ik
            assert float(values[3]) == pytest.approx(0.5, rel=1e-5)  # dist_jk
            assert float(values[1]) == pytest.approx(0.0, rel=1e-5)  # rc

    def test_very_small_report_interval(self) -> None:
        """Test with minimal report interval of 1."""
        from molecular_simulations.simulate.reporters import RCReporter

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            rc_file = path / 'rc_test.log'

            reporter = RCReporter(
                file=rc_file,
                report_interval=1,
                atom_indices=[0, 1, 2],
                rc0=0.0,
            )

            simulation = _make_simulation(current_step=5)

            result = reporter.describeNextReport(simulation)

            # With interval=1, should report every step
            assert result[0] == 1

            reporter.file.close()


@requires_openmm
@pytest.mark.parametrize(
    'atom_i,atom_j,atom_k,positions,expected_rc',
    [
        # Linear arrangement, k at midpoint
        (0, 1, 2, [[0, 0, 0], [1, 0, 0], [0.5, 0, 0]], 0.0),
        # Linear, k closer to i
        (0, 1, 2, [[0, 0, 0], [1, 0, 0], [0.2, 0, 0]], -0.6),
        # Linear, k closer to j
        (0, 1, 2, [[0, 0, 0], [1, 0, 0], [0.8, 0, 0]], 0.6),
        # Different atom ordering
        (2, 1, 0, [[0.5, 0, 0], [1, 0, 0], [0, 0, 0]], 0.0),
        # 2D arrangement
        (0, 1, 2, [[0, 0, 0], [1, 1, 0], [0.5, 0.5, 0]], 0.0),
    ],
)
class TestRCReporterParametrized:
    """Parametrized tests for various atom arrangements."""

    def test_rc_calculation(
        self,
        atom_i: int,
        atom_j: int,
        atom_k: int,
        positions: list[list[float]],
        expected_rc: float,
    ) -> None:
        """Test reaction coordinate calculation for various geometries."""
        from molecular_simulations.simulate.reporters import RCReporter

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            rc_file = path / 'rc_test.log'

            reporter = RCReporter(
                file=rc_file,
                report_interval=10,
                atom_indices=[atom_i, atom_j, atom_k],
                rc0=0.0,
            )

            simulation = _make_simulation(current_step=0)
            state = _make_state(positions)

            reporter.report(simulation, state)
            reporter.file.close()

            content = rc_file.read_text()
            lines = content.strip().split('\n')
            values = lines[1].split(',')

            rc = float(values[1])
            assert rc == pytest.approx(expected_rc, rel=1e-5)


@requires_openmm
class TestRCReporterIntegration:
    """Integration-style tests for RCReporter."""

    def test_full_simulation_workflow(self) -> None:
        """Test RCReporter in a simulated full workflow.

        This test simulates multiple report cycles as would occur
        during an actual EVB simulation, each backed by a real State.
        """
        from molecular_simulations.simulate.reporters import RCReporter

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            rc_file = path / 'rc_workflow.log'

            reporter = RCReporter(
                file=rc_file,
                report_interval=100,
                atom_indices=[0, 1, 2],
                rc0=-0.2,  # Target RC for this window
            )

            simulation = _make_simulation(current_step=0)

            # Simulate trajectory: k moves from i towards j
            n_frames = 10
            for frame in range(n_frames):
                # k position interpolates from near i to near j
                k_pos = 0.1 + frame * 0.08  # 0.1 to 0.82
                state = _make_state(
                    [
                        [0.0, 0.0, 0.0],  # i (donor)
                        [1.0, 0.0, 0.0],  # j (acceptor)
                        [k_pos, 0.0, 0.0],  # k (transferring atom)
                    ]
                )

                reporter.report(simulation, state)

            reporter.file.close()

            # Verify output
            content = rc_file.read_text()
            lines = content.strip().split('\n')

            assert len(lines) == n_frames + 1  # header + data

            # Check that RC values progress correctly
            rc_values = []
            for line in lines[1:]:
                values = line.split(',')
                rc_values.append(float(values[1]))

            # RC should go from negative to positive as k moves from i to j
            assert rc_values[0] < 0  # k close to i
            assert rc_values[-1] > 0  # k close to j

            # All rc0 values should be the target
            for line in lines[1:]:
                values = line.split(',')
                assert float(values[0]) == -0.2


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
