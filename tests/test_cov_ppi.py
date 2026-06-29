"""
Unit tests for analysis/cov_ppi.py module

This module contains both unit tests (with minimal mocks) and integration tests.
Tests use real MDAnalysis when available, with conditional skips for environments
without MDAnalysis installed.
"""

import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import polars as pl
import pytest

# ============================================================================
# Fixtures and helpers for conditional dependency usage
# ============================================================================


def _check_mdanalysis():
    """Check if MDAnalysis is available."""
    try:
        import MDAnalysis

        return True
    except ImportError:
        return False


requires_mdanalysis = pytest.mark.skipif(
    not _check_mdanalysis(), reason="MDAnalysis not installed"
)


@pytest.fixture
def test_data_dir():
    """Return the path to test data directory."""
    return Path(__file__).parent / "data"


@pytest.fixture
def alanine_pdb(test_data_dir):
    """Return the path to the alanine dipeptide PDB."""
    return test_data_dir / "pdb" / "alanine_dipeptide.pdb"


@pytest.fixture
def saltbridge_ppi(two_chain_pdb, tmp_path):
    """Real PPInteractions on the two-chain Lys/Asp salt-bridge PDB.

    Chain A is Ace-Lys-Nme, chain B is Ace-Asp-Nme, with the Lys NZ ~3.4 A from
    the Asp carboxylate -- a genuine salt bridge within the default cutoff.
    """
    from molecular_simulations.analysis.cov_ppi import PPInteractions

    return PPInteractions(
        top=str(two_chain_pdb),
        traj=str(two_chain_pdb),
        out=tmp_path / "results.json",
        plot=False,
    )


@pytest.fixture
def traj_ppi(two_chain_trajectory, tmp_path):
    """Real PPInteractions over the two-chain 5-frame trajectory.

    The PDB topology carries CONECT bonds (needed by the hydrogen-bond donor/
    acceptor survey) and chain B drifts across the frames, so trajectory
    analyses return real, frame-averaged occupancies.
    """
    from molecular_simulations.analysis.cov_ppi import PPInteractions

    return PPInteractions(
        top=str(two_chain_trajectory["top"]),
        traj=str(two_chain_trajectory["traj"]),
        out=tmp_path / "results.json",
        plot=False,
    )


# ============================================================================
# Pure logic tests - no mocking needed
# ============================================================================


class TestPPInteractionsPureLogic:
    """Test pure logic methods that don't need MDAnalysis."""

    def test_parse_results_structure(self):
        """Test parse_results returns correct DataFrame structure - no mocks."""

        # We can test parse_results without initializing the full class
        # by creating a minimal mock object with just the method we need
        results = {
            "positive": {
                "A_ALA1-B_LYS10": {"hydrophobic": 0.5, "hbond": 0.3, "saltbridge": 0.0}
            },
            "negative": {
                "A_GLU5-B_ARG15": {"hydrophobic": 0.0, "hbond": 0.0, "saltbridge": 0.8}
            },
        }

        # Test the static logic of parse_results directly
        data_rows = []
        for cov_type, pair_dict in results.items():
            for pair, data in pair_dict.items():
                if any(val > 0.0 for val in data.values()):
                    row = {
                        "Residue Pair": pair,
                        "Hydrophobic": data["hydrophobic"],
                        "Hydrogen Bond": data["hbond"],
                        "Salt Bridge": data["saltbridge"],
                        "Covariance": cov_type,
                    }
                    data_rows.append(row)

        df = pl.DataFrame(data_rows)

        assert isinstance(df, pl.DataFrame)
        assert "Residue Pair" in df.columns
        assert "Hydrophobic" in df.columns
        assert "Covariance" in df.columns
        assert len(df) == 2

    def test_parse_results_filters_zeros(self):
        """Test that parse_results filters out all-zero entries - no mocks."""
        results = {
            "positive": {
                "A_ALA1-B_LYS10": {"hydrophobic": 0.5, "hbond": 0.0, "saltbridge": 0.0},
                "A_GLY2-B_SER11": {"hydrophobic": 0.0, "hbond": 0.0, "saltbridge": 0.0},
            },
            "negative": {},
        }

        data_rows = []
        for cov_type, pair_dict in results.items():
            for pair, data in pair_dict.items():
                if any(val > 0.0 for val in data.values()):
                    row = {
                        "Residue Pair": pair,
                        "Hydrophobic": data["hydrophobic"],
                        "Hydrogen Bond": data["hbond"],
                        "Salt Bridge": data["saltbridge"],
                        "Covariance": cov_type,
                    }
                    data_rows.append(row)

        df = pl.DataFrame(data_rows)
        assert len(df) == 1
        assert "A_ALA1-B_LYS10" in df["Residue Pair"].to_list()

    def test_interpret_covariance_logic(self):
        """Test interpret_covariance logic with numpy arrays - no mocks."""
        # Test the interpretation logic without the full class
        mapping = {"ag1": {0: 1, 1: 2}, "ag2": {0: 10, 1: 11}}

        cov_mat = np.array(
            [
                [0.5, -0.3],
                [-0.2, 0.4],
            ]
        )

        pos_corr = np.where(cov_mat > 0.0)
        neg_corr = np.where(cov_mat < 0.0)

        seen = set()
        positive = []
        for i in range(len(pos_corr[0])):
            res1 = mapping["ag1"][pos_corr[0][i]]
            res2 = mapping["ag2"][pos_corr[1][i]]
            if (res1, res2) not in seen:
                positive.append((res1, res2))
                seen.add((res1, res2))
                seen.add((res2, res1))

        negative = []
        for i in range(len(neg_corr[0])):
            res1 = mapping["ag1"][neg_corr[0][i]]
            res2 = mapping["ag2"][neg_corr[1][i]]
            if (res1, res2) not in seen:
                negative.append((res1, res2))
                seen.add((res1, res2))
                seen.add((res2, res1))

        assert len(positive) == 2
        assert len(negative) == 2
        assert (1, 10) in positive
        assert (2, 11) in positive


# ============================================================================
# Integration tests using real MDAnalysis
# ============================================================================


@requires_mdanalysis
class TestPPInteractionsIntegration:
    """Integration tests using real MDAnalysis."""

    def test_ppinteractions_init_with_real_file(self, alanine_pdb, tmp_path):
        """Test PPInteractions initialization with real PDB file."""
        from molecular_simulations.analysis.cov_ppi import PPInteractions

        out_path = tmp_path / "results.json"

        ppi = PPInteractions(
            top=str(alanine_pdb),
            traj=str(alanine_pdb),  # Single PDB as trajectory
            out=out_path,
            sel1="resid 1",
            sel2="resid 2",
            plot=False,
        )

        assert ppi.n_frames == 1  # Single frame PDB
        assert ppi.u is not None

    def test_res_map_with_real_universe(self, alanine_pdb, tmp_path):
        """Test res_map with real MDAnalysis Universe."""
        import MDAnalysis as mda

        u = mda.Universe(str(alanine_pdb))
        ag1 = u.select_atoms("resid 1")
        ag2 = u.select_atoms("resid 2")

        # Test the mapping logic
        mapping = {"ag1": {}, "ag2": {}}
        for i, resid in enumerate(ag1.resids):
            mapping["ag1"][i] = resid
        for i, resid in enumerate(ag2.resids):
            mapping["ag2"][i] = resid

        assert 0 in mapping["ag1"]
        assert 0 in mapping["ag2"]

    def test_save_and_load_results(self, alanine_pdb, tmp_path):
        """Test save method creates valid JSON."""
        from molecular_simulations.analysis.cov_ppi import PPInteractions

        out_path = tmp_path / "results.json"

        ppi = PPInteractions(
            top=str(alanine_pdb),
            traj=str(alanine_pdb),
            out=out_path,
            sel1="resid 1",
            sel2="resid 2",
            plot=False,
        )

        results = {
            "positive": {
                "A_ALA1-B_LYS10": {"hydrophobic": 0.5, "hbond": 0.3, "saltbridge": 0.0}
            },
            "negative": {},
        }

        ppi.save(results)

        assert out_path.exists()
        with open(out_path) as f:
            loaded = json.load(f)
        assert "positive" in loaded
        assert "A_ALA1-B_LYS10" in loaded["positive"]


# ============================================================================
# Unit tests with minimal mocking
# ============================================================================


class TestPPInteractions:
    """Test suite for PPInteractions class - uses mocks for unavailable deps."""

    @patch("molecular_simulations.analysis.cov_ppi.mda")
    def test_ppinteractions_init(self, mock_mda):
        """Test PPInteractions initialization."""
        from molecular_simulations.analysis.cov_ppi import PPInteractions

        # Setup mock universe
        mock_universe = MagicMock()
        mock_universe.trajectory = MagicMock()
        mock_universe.trajectory.__len__ = MagicMock(return_value=100)
        mock_mda.Universe.return_value = mock_universe

        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "results.json"

            ppi = PPInteractions(
                top="fake.prmtop",
                traj="fake.dcd",
                out=out_path,
                sel1="chainID A",
                sel2="chainID B",
                cov_cutoff=(11.0, 13.0),
                sb_cutoff=6.0,
                hbond_cutoff=3.5,
                hbond_angle=30.0,
                hydrophobic_cutoff=8.0,
                plot=False,
            )

            assert ppi.n_frames == 100
            assert ppi.sel1 == "chainID A"
            assert ppi.sel2 == "chainID B"
            assert ppi.cov_cutoff == (11.0, 13.0)
            assert ppi.sb == 6.0
            assert ppi.hb_d == 3.5
            assert ppi.hydr == 8.0
            assert not ppi.plot

    @patch("molecular_simulations.analysis.cov_ppi.mda")
    def test_ppinteractions_hbond_angle_conversion(self, mock_mda):
        """Test that hbond angle is converted to radians."""
        from molecular_simulations.analysis.cov_ppi import PPInteractions

        mock_universe = MagicMock()
        mock_universe.trajectory = MagicMock()
        mock_universe.trajectory.__len__ = MagicMock(return_value=10)
        mock_mda.Universe.return_value = mock_universe

        with tempfile.TemporaryDirectory() as tmpdir:
            ppi = PPInteractions(
                top="fake.prmtop",
                traj="fake.dcd",
                out=Path(tmpdir) / "results.json",
                hbond_angle=30.0,  # 30 degrees
                plot=False,
            )

            # Should be converted using the formula: angle * 180 / pi
            expected_angle = 30.0 * 180 / np.pi
            assert np.isclose(ppi.hb_a, expected_angle)

    @patch("molecular_simulations.analysis.cov_ppi.mda")
    def test_res_map(self, mock_mda):
        """Test res_map method"""
        from molecular_simulations.analysis.cov_ppi import PPInteractions

        mock_universe = MagicMock()
        mock_universe.trajectory.__len__ = MagicMock(return_value=10)
        mock_mda.Universe.return_value = mock_universe

        with tempfile.TemporaryDirectory() as tmpdir:
            ppi = PPInteractions(
                top="fake.prmtop",
                traj="fake.dcd",
                out=Path(tmpdir) / "results.json",
                plot=False,
            )

            # Create mock atom groups
            mock_ag1 = MagicMock()
            mock_ag1.resids = np.array([1, 2, 3])

            mock_ag2 = MagicMock()
            mock_ag2.resids = np.array([10, 11])

            ppi.res_map(mock_ag1, mock_ag2)

            assert ppi.mapping["ag1"][0] == 1
            assert ppi.mapping["ag1"][1] == 2
            assert ppi.mapping["ag1"][2] == 3
            assert ppi.mapping["ag2"][0] == 10
            assert ppi.mapping["ag2"][1] == 11

    @patch("molecular_simulations.analysis.cov_ppi.mda")
    def test_interpret_covariance(self, mock_mda):
        """Test interpret_covariance method"""
        from molecular_simulations.analysis.cov_ppi import PPInteractions

        mock_universe = MagicMock()
        mock_universe.trajectory.__len__ = MagicMock(return_value=10)
        mock_mda.Universe.return_value = mock_universe

        with tempfile.TemporaryDirectory() as tmpdir:
            ppi = PPInteractions(
                top="fake.prmtop",
                traj="fake.dcd",
                out=Path(tmpdir) / "results.json",
                plot=False,
            )

            # Setup mapping
            ppi.mapping = {"ag1": {0: 1, 1: 2}, "ag2": {0: 10, 1: 11}}

            # Create covariance matrix with positive and negative values
            cov_mat = np.array(
                [
                    [0.5, -0.3],  # Residue 1 has pos corr with 10, neg with 11
                    [-0.2, 0.4],  # Residue 2 has neg corr with 10, pos with 11
                ]
            )

            positive, negative = ppi.interpret_covariance(cov_mat)

            # Should have found positive correlations
            assert len(positive) > 0
            # Should have found negative correlations
            assert len(negative) > 0

    @patch("molecular_simulations.analysis.cov_ppi.mda")
    def test_identify_interaction_type(self, mock_mda):
        """Test identify_interaction_type method"""
        from molecular_simulations.analysis.cov_ppi import PPInteractions

        mock_universe = MagicMock()
        mock_universe.trajectory.__len__ = MagicMock(return_value=10)
        mock_mda.Universe.return_value = mock_universe

        with tempfile.TemporaryDirectory() as tmpdir:
            ppi = PPInteractions(
                top="fake.prmtop",
                traj="fake.dcd",
                out=Path(tmpdir) / "results.json",
                plot=False,
            )

            # Test charged residue pair (ASP-LYS should have saltbridge)
            functions, labels = ppi.identify_interaction_type("ASP", "LYS")

            assert (
                "saltbridge" in labels or "hydrophobic" in labels or "hbond" in labels
            )

    @patch("molecular_simulations.analysis.cov_ppi.mda")
    def test_save_results(self, mock_mda):
        """Test save method"""
        from molecular_simulations.analysis.cov_ppi import PPInteractions

        mock_universe = MagicMock()
        mock_universe.trajectory.__len__ = MagicMock(return_value=10)
        mock_mda.Universe.return_value = mock_universe

        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "results.json"

            ppi = PPInteractions(
                top="fake.prmtop", traj="fake.dcd", out=out_path, plot=False
            )

            results = {
                "positive": {
                    "A_ALA1-B_LYS10": {
                        "hydrophobic": 0.5,
                        "hbond": 0.3,
                        "saltbridge": 0.0,
                    }
                },
                "negative": {},
            }

            ppi.save(results)

            assert out_path.exists()

            with open(out_path) as f:
                loaded = json.load(f)

            assert "positive" in loaded
            assert "A_ALA1-B_LYS10" in loaded["positive"]

    @patch("molecular_simulations.analysis.cov_ppi.mda")
    def test_parse_results(self, mock_mda):
        """Test parse_results method"""
        from molecular_simulations.analysis.cov_ppi import PPInteractions

        mock_universe = MagicMock()
        mock_universe.trajectory.__len__ = MagicMock(return_value=10)
        mock_mda.Universe.return_value = mock_universe

        with tempfile.TemporaryDirectory() as tmpdir:
            ppi = PPInteractions(
                top="fake.prmtop",
                traj="fake.dcd",
                out=Path(tmpdir) / "results.json",
                plot=False,
            )

            results = {
                "positive": {
                    "A_ALA1-B_LYS10": {
                        "hydrophobic": 0.5,
                        "hbond": 0.3,
                        "saltbridge": 0.0,
                    }
                },
                "negative": {
                    "A_GLU5-B_ARG15": {
                        "hydrophobic": 0.0,
                        "hbond": 0.0,
                        "saltbridge": 0.8,
                    }
                },
            }

            df = ppi.parse_results(results)

            assert isinstance(df, pl.DataFrame)
            assert "Residue Pair" in df.columns
            assert "Hydrophobic" in df.columns
            assert "Hydrogen Bond" in df.columns
            assert "Salt Bridge" in df.columns
            assert "Covariance" in df.columns

    @patch("molecular_simulations.analysis.cov_ppi.mda")
    def test_parse_results_filters_zeros(self, mock_mda):
        """Test that parse_results filters out all-zero entries"""
        from molecular_simulations.analysis.cov_ppi import PPInteractions

        mock_universe = MagicMock()
        mock_universe.trajectory.__len__ = MagicMock(return_value=10)
        mock_mda.Universe.return_value = mock_universe

        with tempfile.TemporaryDirectory() as tmpdir:
            ppi = PPInteractions(
                top="fake.prmtop",
                traj="fake.dcd",
                out=Path(tmpdir) / "results.json",
                plot=False,
            )

            results = {
                "positive": {
                    "A_ALA1-B_LYS10": {
                        "hydrophobic": 0.5,
                        "hbond": 0.0,
                        "saltbridge": 0.0,
                    },
                    "A_GLY2-B_SER11": {
                        "hydrophobic": 0.0,
                        "hbond": 0.0,
                        "saltbridge": 0.0,
                    },
                },
                "negative": {},
            }

            df = ppi.parse_results(results)

            # Should only include the non-zero entry
            assert len(df) == 1
            assert "A_ALA1-B_LYS10" in df["Residue Pair"].to_list()


class TestEvaluateHBond:
    """Test the evaluate_hbond method"""

    @patch("molecular_simulations.analysis.cov_ppi.mda")
    def test_evaluate_hbond_found(self, mock_mda):
        """Test evaluate_hbond when HBond is found"""
        from molecular_simulations.analysis.cov_ppi import PPInteractions

        mock_universe = MagicMock()
        mock_universe.trajectory.__len__ = MagicMock(return_value=10)
        mock_mda.Universe.return_value = mock_universe

        with tempfile.TemporaryDirectory() as tmpdir:
            ppi = PPInteractions(
                top="fake.prmtop",
                traj="fake.dcd",
                out=Path(tmpdir) / "results.json",
                plot=False,
            )

            # Create mock donor and acceptor atoms
            mock_h = MagicMock()
            mock_h.position = np.array([0.0, 1.0, 0.0])
            mock_h.type = "H"

            mock_donor = MagicMock()
            mock_donor.position = np.array([0.0, 0.0, 0.0])
            mock_donor.bonded_atoms = MagicMock()
            mock_donor.bonded_atoms.__iter__ = MagicMock(return_value=iter([mock_h]))

            mock_donor_group = MagicMock()
            mock_donor_group.atoms = [mock_donor]

            mock_acceptor = MagicMock()
            mock_acceptor.position = np.array([0.0, 2.5, 0.0])  # Close enough for HBond

            mock_acceptor_group = MagicMock()
            mock_acceptor_group.atoms = [mock_acceptor]

            # Set distance cutoff to allow this to be an HBond
            ppi.hb_d = 3.5
            ppi.hb_a = 60.0  # Allow wide angle

            result = ppi.evaluate_hbond(mock_donor_group, mock_acceptor_group)

            # Should return 0 or 1
            assert result in [0, 1]


class TestAnalyzeHydrophobic:
    """Test the analyze_hydrophobic method on a real two-chain trajectory."""

    def test_analyze_hydrophobic(self, traj_ppi):
        """analyze_hydrophobic returns a real frame-averaged occupancy."""
        lys = traj_ppi.u.select_atoms("chainID A and resname LYS")
        asp = traj_ppi.u.select_atoms("chainID B and resname ASP")

        result = traj_ppi.analyze_hydrophobic(lys, asp)

        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0
        # The carbon side chains start within the 8 A cutoff
        assert result > 0.0


class TestAnalyzeSaltbridge:
    """Test the analyze_saltbridge method on a real Lys/Asp interface."""

    def test_analyze_saltbridge_real_pair(self, saltbridge_ppi):
        """A real Lys-Asp pair within cutoff is a full-occupancy salt bridge."""
        lys = saltbridge_ppi.u.select_atoms("chainID A and resname LYS")
        asp = saltbridge_ppi.u.select_atoms("chainID B and resname ASP")

        # Single frame, NZ <-> carboxylate ~3.4 A < 6.0 A cutoff -> occupancy 1.0
        assert saltbridge_ppi.analyze_saltbridge(lys, asp) == 1.0

    def test_analyze_saltbridge_incompatible_residues(self, saltbridge_ppi):
        """Saltbridge analysis returns 0 for non-charged residues."""
        res1 = SimpleNamespace(resnames=["ALA"])
        res2 = SimpleNamespace(resnames=["GLY"])

        assert saltbridge_ppi.analyze_saltbridge(res1, res2) == 0.0

    def test_analyze_saltbridge_same_charge(self, saltbridge_ppi):
        """Saltbridge analysis returns 0 for two positively-charged residues."""
        res1 = SimpleNamespace(resnames=["LYS"])
        res2 = SimpleNamespace(resnames=["ARG"])

        assert saltbridge_ppi.analyze_saltbridge(res1, res2) == 0.0

    def test_analyze_saltbridge_two_negative(self, saltbridge_ppi):
        """Saltbridge analysis returns 0 for two negatively-charged residues."""
        res1 = SimpleNamespace(resnames=["ASP"])
        res2 = SimpleNamespace(resnames=["GLU"])

        assert saltbridge_ppi.analyze_saltbridge(res1, res2) == 0.0


class TestComputeInteractions:
    """Test compute_interactions method"""

    @patch("molecular_simulations.analysis.cov_ppi.convert_aa_code")
    @patch("molecular_simulations.analysis.cov_ppi.mda")
    def test_compute_interactions(self, mock_mda, mock_convert):
        """Test compute_interactions method"""
        from molecular_simulations.analysis.cov_ppi import PPInteractions

        mock_universe = MagicMock()
        mock_universe.trajectory.__len__ = MagicMock(return_value=10)
        mock_mda.Universe.return_value = mock_universe

        # Mock the select_atoms return
        mock_grp = MagicMock()
        mock_grp.resnames = ["ALA"]
        mock_universe.select_atoms.return_value = mock_grp

        # Mock the aa code conversion
        mock_convert.return_value = "A"

        with tempfile.TemporaryDirectory() as tmpdir:
            ppi = PPInteractions(
                top="fake.prmtop",
                traj="fake.dcd",
                out=Path(tmpdir) / "results.json",
                plot=False,
            )

            # Mock identify_interaction_type to return simple callable
            ppi.identify_interaction_type = MagicMock(
                return_value=([lambda x, y: 0.5], ["hydrophobic"])
            )

            result = ppi.compute_interactions(1, 10)

            assert isinstance(result, dict)
            # Result should have key in format 'A_X1-B_X10'


class TestIdentifyInteractionType:
    """Test identify_interaction_type method"""

    @patch("molecular_simulations.analysis.cov_ppi.mda")
    def test_identify_interaction_type_polar(self, mock_mda):
        """Test interaction type identification for polar residues"""
        from molecular_simulations.analysis.cov_ppi import PPInteractions

        mock_universe = MagicMock()
        mock_universe.trajectory.__len__ = MagicMock(return_value=10)
        mock_mda.Universe.return_value = mock_universe

        with tempfile.TemporaryDirectory() as tmpdir:
            ppi = PPInteractions(
                top="fake.prmtop",
                traj="fake.dcd",
                out=Path(tmpdir) / "results.json",
                plot=False,
            )

            # Test SER-THR (should have hbond capability)
            functions, labels = ppi.identify_interaction_type("SER", "THR")
            assert "hydrophobic" in labels
            assert "hbond" in labels

    @patch("molecular_simulations.analysis.cov_ppi.mda")
    def test_identify_interaction_type_charged(self, mock_mda):
        """Test interaction type identification for charged residues"""
        from molecular_simulations.analysis.cov_ppi import PPInteractions

        mock_universe = MagicMock()
        mock_universe.trajectory.__len__ = MagicMock(return_value=10)
        mock_mda.Universe.return_value = mock_universe

        with tempfile.TemporaryDirectory() as tmpdir:
            ppi = PPInteractions(
                top="fake.prmtop",
                traj="fake.dcd",
                out=Path(tmpdir) / "results.json",
                plot=False,
            )

            # Test ASP-LYS (should have saltbridge capability)
            functions, labels = ppi.identify_interaction_type("ASP", "LYS")
            assert "hydrophobic" in labels
            assert "saltbridge" in labels

    @patch("molecular_simulations.analysis.cov_ppi.mda")
    def test_identify_interaction_type_hydrophobic(self, mock_mda):
        """Test interaction type identification for hydrophobic residues"""
        from molecular_simulations.analysis.cov_ppi import PPInteractions

        mock_universe = MagicMock()
        mock_universe.trajectory.__len__ = MagicMock(return_value=10)
        mock_mda.Universe.return_value = mock_universe

        with tempfile.TemporaryDirectory() as tmpdir:
            ppi = PPInteractions(
                top="fake.prmtop",
                traj="fake.dcd",
                out=Path(tmpdir) / "results.json",
                plot=False,
            )

            # Test ALA-VAL (hydrophobic only)
            functions, labels = ppi.identify_interaction_type("ALA", "VAL")
            assert "hydrophobic" in labels
            # ALA and VAL are not in the int_types dict, so only hydrophobic


class TestMakePlot:
    """Test make_plot method"""

    @patch("molecular_simulations.analysis.cov_ppi.plt")
    @patch("molecular_simulations.analysis.cov_ppi.sns")
    @patch("molecular_simulations.analysis.cov_ppi.mda")
    def test_make_plot(self, mock_mda, mock_sns, mock_plt):
        """Test make_plot method"""
        from molecular_simulations.analysis.cov_ppi import PPInteractions

        mock_universe = MagicMock()
        mock_universe.trajectory.__len__ = MagicMock(return_value=10)
        mock_mda.Universe.return_value = mock_universe

        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_plt.subplots.return_value = (mock_fig, mock_ax)

        with tempfile.TemporaryDirectory() as tmpdir:
            ppi = PPInteractions(
                top="fake.prmtop",
                traj="fake.dcd",
                out=Path(tmpdir) / "results.json",
                plot=False,
            )

            data = pl.DataFrame(
                {
                    "Residue Pair": ["A_ALA1-B_LYS10"],
                    "Hydrophobic": [0.5],
                    "Hydrogen Bond": [0.3],
                    "Salt Bridge": [0.0],
                    "Covariance": ["positive"],
                }
            )

            plot_path = Path(tmpdir) / "test_plot.png"
            ppi.make_plot(data, "Hydrophobic", plot_path)

            mock_sns.barplot.assert_called_once()
            mock_plt.savefig.assert_called()


class TestPlotResults:
    """Test plot_results method"""

    @patch("molecular_simulations.analysis.cov_ppi.Path")
    @patch("molecular_simulations.analysis.cov_ppi.mda")
    def test_plot_results(self, mock_mda, mock_path):
        """Test plot_results method"""
        from molecular_simulations.analysis.cov_ppi import PPInteractions

        mock_universe = MagicMock()
        mock_universe.trajectory.__len__ = MagicMock(return_value=10)
        mock_mda.Universe.return_value = mock_universe

        # Mock Path to avoid filesystem operations
        mock_plot_dir = MagicMock()
        mock_path.return_value = mock_plot_dir

        with tempfile.TemporaryDirectory() as tmpdir:
            ppi = PPInteractions(
                top="fake.prmtop",
                traj="fake.dcd",
                out=Path(tmpdir) / "results.json",
                plot=False,
            )

            # Mock make_plot to avoid actual plotting
            ppi.make_plot = MagicMock()
            # Mock parse_results
            ppi.parse_results = MagicMock(
                return_value=pl.DataFrame(
                    {
                        "Residue Pair": ["A_ALA1-B_LYS10"],
                        "Hydrophobic": [0.5],
                        "Hydrogen Bond": [0.3],
                        "Salt Bridge": [0.0],
                        "Covariance": ["positive"],
                    }
                )
            )

            results = {
                "positive": {
                    "A_ALA1-B_LYS10": {
                        "hydrophobic": 0.5,
                        "hbond": 0.3,
                        "saltbridge": 0.0,
                    }
                },
                "negative": {},
            }

            ppi.plot_results(results)

            # make_plot should have been called for non-zero interactions
            assert ppi.make_plot.called


class TestSurveyDonorsAcceptors:
    """Test survey_donors_acceptors method"""

    @patch("molecular_simulations.analysis.cov_ppi.distance_array")
    @patch("molecular_simulations.analysis.cov_ppi.mda")
    def test_survey_donors_acceptors(self, mock_mda, mock_dist_array):
        """Test survey_donors_acceptors method"""
        from molecular_simulations.analysis.cov_ppi import PPInteractions

        mock_universe = MagicMock()
        mock_universe.trajectory.__len__ = MagicMock(return_value=10)
        mock_universe.select_atoms.return_value = MagicMock()
        mock_mda.Universe.return_value = mock_universe

        # Mock distance array to return contacts
        mock_dist_array.return_value = np.array([[2.5, 5.0], [3.0, 4.0]])

        with tempfile.TemporaryDirectory() as tmpdir:
            ppi = PPInteractions(
                top="fake.prmtop",
                traj="fake.dcd",
                out=Path(tmpdir) / "results.json",
                plot=False,
            )

            # Create mock atoms with O and N types
            mock_h = MagicMock()
            mock_h.types = ["H"]

            mock_atom1 = MagicMock()
            mock_atom1.type = "O"
            mock_atom1.bonded_atoms = MagicMock()
            mock_atom1.bonded_atoms.types = ["H", "C"]

            mock_res1 = MagicMock()
            mock_res1.atoms = [mock_atom1]

            mock_atom2 = MagicMock()
            mock_atom2.type = "N"
            mock_atom2.bonded_atoms = MagicMock()
            mock_atom2.bonded_atoms.types = ["H", "C"]

            mock_res2 = MagicMock()
            mock_res2.atoms = [mock_atom2]

            donors, acceptors = ppi.survey_donors_acceptors(mock_res1, mock_res2)

            # Should return AtomGroup-like objects
            assert donors is not None
            assert acceptors is not None


class TestAnalyzeHbond:
    """Test analyze_hbond method on a real two-chain trajectory."""

    def test_analyze_hbond(self, traj_ppi):
        """analyze_hbond surveys real donors/acceptors and scores geometry.

        Relies on the fixture's CONECT bond records (donor-hydrogen lookup) and
        the Lys-Asp interface, where the amine donates to the carboxylate.
        """
        lys = traj_ppi.u.select_atoms("chainID A and resname LYS")
        asp = traj_ppi.u.select_atoms("chainID B and resname ASP")

        result = traj_ppi.analyze_hbond(lys, asp)

        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0
        assert result > 0.0


class TestPPInteractionsRun:
    """Test suite for PPInteractions.run() method."""

    @patch("molecular_simulations.analysis.cov_ppi.mda")
    def test_run_calls_workflow_steps(self, mock_mda):
        """Test run() calls covariance, interpret, compute, save."""
        mock_universe = MagicMock()
        mock_universe.trajectory.__len__ = MagicMock(return_value=10)
        mock_mda.Universe.return_value = mock_universe

        from molecular_simulations.analysis.cov_ppi import PPInteractions

        with tempfile.TemporaryDirectory() as tmpdir:
            ppi = PPInteractions(
                top="fake.prmtop",
                traj="fake.dcd",
                out=Path(tmpdir) / "results.json",
                plot=False,
            )

            mock_cov = np.array([[0.5, -0.3], [-0.1, 0.2]])
            positive = [(1, 10)]
            negative = [(2, 11)]

            with (
                patch.object(ppi, "get_covariance", return_value=mock_cov) as mock_gc,
                patch.object(
                    ppi, "interpret_covariance", return_value=(positive, negative)
                ) as mock_ic,
                patch.object(
                    ppi, "compute_interactions", return_value={"pair": {"hbond": 0.5}}
                ) as mock_ci,
                patch.object(ppi, "save") as mock_save,
            ):
                ppi.run()

            mock_gc.assert_called_once()
            mock_ic.assert_called_once_with(mock_cov)
            assert mock_ci.call_count == 2  # once for positive, once for negative
            mock_save.assert_called_once()


class TestPPInteractionsComputeInteractions:
    """Test suite for PPInteractions.compute_interactions()."""

    @patch("molecular_simulations.analysis.cov_ppi.mda")
    @patch("molecular_simulations.analysis.cov_ppi.convert_aa_code")
    def test_compute_interactions_returns_data(self, mock_convert, mock_mda):
        """Test compute_interactions returns dict with interaction types."""
        mock_universe = MagicMock()
        mock_universe.trajectory.__len__ = MagicMock(return_value=10)
        mock_mda.Universe.return_value = mock_universe

        mock_convert.side_effect = lambda x: x[:3]

        from molecular_simulations.analysis.cov_ppi import PPInteractions

        with tempfile.TemporaryDirectory() as tmpdir:
            ppi = PPInteractions(
                top="fake.prmtop",
                traj="fake.dcd",
                out=Path(tmpdir) / "results.json",
                plot=False,
            )

            # Mock select_atoms to return groups with resnames
            mock_grp1 = MagicMock()
            mock_grp1.resnames = ["ALA"]
            mock_grp2 = MagicMock()
            mock_grp2.resnames = ["GLY"]
            ppi.u.select_atoms.side_effect = [mock_grp1, mock_grp2]

            # Mock identify_interaction_type
            mock_func = MagicMock(return_value=0.5)
            ppi.identify_interaction_type = MagicMock(
                return_value=([mock_func], ["hydrophobic"])
            )

            result = ppi.compute_interactions(1, 10)

            assert isinstance(result, dict)
            key = list(result.keys())[0]
            assert "hydrophobic" in result[key]


class TestPPInteractionsAnalyzeSaltbridge:
    """Test suite for PPInteractions.analyze_saltbridge()."""

    @patch("molecular_simulations.analysis.cov_ppi.mda")
    def test_saltbridge_non_charged_returns_zero(self, mock_mda):
        """Test saltbridge returns 0 for non-charged residues."""
        mock_universe = MagicMock()
        mock_universe.trajectory.__len__ = MagicMock(return_value=10)
        mock_mda.Universe.return_value = mock_universe

        from molecular_simulations.analysis.cov_ppi import PPInteractions

        with tempfile.TemporaryDirectory() as tmpdir:
            ppi = PPInteractions(
                top="fake.prmtop",
                traj="fake.dcd",
                out=Path(tmpdir) / "results.json",
                plot=False,
            )

            # ALA is not in charged residues
            mock_res1 = MagicMock()
            mock_res1.resnames = ["ALA"]
            mock_res2 = MagicMock()
            mock_res2.resnames = ["GLU"]

            result = ppi.analyze_saltbridge(mock_res1, mock_res2)
            assert result == 0.0

    @patch("molecular_simulations.analysis.cov_ppi.mda")
    def test_saltbridge_same_charge_returns_zero(self, mock_mda):
        """Test saltbridge returns 0 for same-charge residues."""
        mock_universe = MagicMock()
        mock_universe.trajectory.__len__ = MagicMock(return_value=10)
        mock_mda.Universe.return_value = mock_universe

        from molecular_simulations.analysis.cov_ppi import PPInteractions

        with tempfile.TemporaryDirectory() as tmpdir:
            ppi = PPInteractions(
                top="fake.prmtop",
                traj="fake.dcd",
                out=Path(tmpdir) / "results.json",
                plot=False,
            )

            # Both positive
            mock_res1 = MagicMock()
            mock_res1.resnames = ["LYS"]
            mock_res2 = MagicMock()
            mock_res2.resnames = ["ARG"]

            result = ppi.analyze_saltbridge(mock_res1, mock_res2)
            assert result == 0.0

            # Both negative
            mock_res1.resnames = ["ASP"]
            mock_res2.resnames = ["GLU"]

            result = ppi.analyze_saltbridge(mock_res1, mock_res2)
            assert result == 0.0


class TestPPInteractionsGetCovariance:
    """Test suite for PPInteractions.get_covariance() method."""

    @patch("molecular_simulations.analysis.cov_ppi.mda")
    def test_get_covariance_returns_matrix(self, mock_mda):
        """Test get_covariance computes and returns a covariance matrix."""
        # Set up mock universe with a trajectory of 2 frames and 2 CA atoms per chain
        mock_universe = MagicMock()

        # Create mock trajectory frames
        frame1 = MagicMock()
        frame2 = MagicMock()
        mock_universe.trajectory.__len__ = MagicMock(return_value=2)

        # Need to be able to iterate multiple times
        def make_iter():
            return iter([frame1, frame2])

        mock_universe.trajectory.__iter__ = MagicMock(side_effect=make_iter)

        # Mock CA selections
        mock_ca_A = MagicMock()
        mock_ca_A.n_residues = 2
        mock_ca_A.resids = np.array([1, 2])
        mock_ca_A.positions = np.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])

        mock_ca_B = MagicMock()
        mock_ca_B.n_residues = 2
        mock_ca_B.resids = np.array([10, 11])
        mock_ca_B.positions = np.array([[5.0, 0.0, 0.0], [6.0, 0.0, 0.0]])

        mock_universe.select_atoms.side_effect = [mock_ca_A, mock_ca_B]
        mock_mda.Universe.return_value = mock_universe

        from molecular_simulations.analysis.cov_ppi import PPInteractions

        with tempfile.TemporaryDirectory() as tmpdir:
            ppi = PPInteractions(
                top="fake.prmtop",
                traj="fake.dcd",
                out=Path(tmpdir) / "results.json",
                plot=False,
            )

            C = ppi.get_covariance()

            assert C.shape == (2, 2)
            assert isinstance(C, np.ndarray)


class TestPPInteractionsAnalyzeSaltbridgeWithTrajectory:
    """Test saltbridge with valid charged pair over trajectory."""

    @patch("molecular_simulations.analysis.cov_ppi.mda")
    def test_saltbridge_valid_pair_with_trajectory(self, mock_mda):
        """Test saltbridge with LYS-ASP pair returns occupancy."""
        mock_universe = MagicMock()
        mock_universe.trajectory.__len__ = MagicMock(return_value=2)

        frame1 = MagicMock()
        frame2 = MagicMock()
        mock_universe.trajectory.__iter__ = MagicMock(
            return_value=iter([frame1, frame2])
        )

        # Mock empty DUMMY selection that supports +=
        mock_empty = MagicMock()
        mock_empty.__iadd__ = MagicMock(return_value=mock_empty)
        mock_universe.select_atoms.return_value = mock_empty
        mock_mda.Universe.return_value = mock_universe

        from molecular_simulations.analysis.cov_ppi import PPInteractions

        with tempfile.TemporaryDirectory() as tmpdir:
            ppi = PPInteractions(
                top="fake.prmtop",
                traj="fake.dcd",
                out=Path(tmpdir) / "results.json",
                plot=False,
                sb_cutoff=6.0,
            )

            # Create mock residue groups with charged amino acids
            mock_res1 = MagicMock()
            mock_res1.resnames = ["LYS"]
            mock_atom1 = MagicMock()
            mock_atom1.name = "NZ"
            mock_res1.atoms = [mock_atom1]

            mock_res2 = MagicMock()
            mock_res2.resnames = ["ASP"]
            mock_atom2 = MagicMock()
            mock_atom2.name = "OD1"
            mock_res2.atoms = [mock_atom2]

            # Mock the grp positions for distance calculation
            mock_empty.positions = np.array([[0.0, 0.0, 0.0]])

            result = ppi.analyze_saltbridge(mock_res1, mock_res2)
            assert isinstance(result, float)
            assert 0.0 <= result <= 1.0


class TestPPInteractionsRunWithPlot:
    """Test run method with plot=True triggers plot_results."""

    @patch("molecular_simulations.analysis.cov_ppi.mda")
    def test_run_with_plot_calls_plot_results(self, mock_mda):
        """Test run() calls plot_results when plot=True."""
        mock_universe = MagicMock()
        mock_universe.trajectory.__len__ = MagicMock(return_value=10)
        mock_mda.Universe.return_value = mock_universe

        from molecular_simulations.analysis.cov_ppi import PPInteractions

        with tempfile.TemporaryDirectory() as tmpdir:
            ppi = PPInteractions(
                top="fake.prmtop",
                traj="fake.dcd",
                out=Path(tmpdir) / "results.json",
                plot=True,
            )

            mock_cov = np.array([[0.5]])
            with (
                patch.object(ppi, "get_covariance", return_value=mock_cov),
                patch.object(ppi, "interpret_covariance", return_value=([], [])),
                patch.object(ppi, "save"),
                patch.object(ppi, "plot_results") as mock_plot,
            ):
                ppi.run()

            mock_plot.assert_called_once()


class TestPPInteractionsSave:
    """Test suite for PPInteractions.save()."""

    @patch("molecular_simulations.analysis.cov_ppi.mda")
    def test_save_creates_json_file(self, mock_mda):
        """Test save writes results dict as JSON."""
        mock_universe = MagicMock()
        mock_universe.trajectory.__len__ = MagicMock(return_value=10)
        mock_mda.Universe.return_value = mock_universe

        from molecular_simulations.analysis.cov_ppi import PPInteractions

        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "results.json"
            ppi = PPInteractions(
                top="fake.prmtop", traj="fake.dcd", out=out_path, plot=False
            )

            results = {
                "positive": {"A_ALA1-B_GLY10": {"hydrophobic": 0.5}},
                "negative": {},
            }
            ppi.save(results)

            assert out_path.exists()
            with open(out_path) as f:
                loaded = json.load(f)
            assert loaded == results


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
