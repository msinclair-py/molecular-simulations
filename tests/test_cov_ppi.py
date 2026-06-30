"""
Unit tests for analysis/cov_ppi.py module

This module contains both unit tests (with minimal mocks) and integration tests.
Tests use real MDAnalysis when available, with conditional skips for environments
without MDAnalysis installed.
"""

import json
from pathlib import Path
from types import SimpleNamespace

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
    """Real PPInteractions tests on the two-chain fixture."""

    def test_ppinteractions_init(self, two_chain_pdb, tmp_path):
        """Test PPInteractions initialization stores parameters."""
        from molecular_simulations.analysis.cov_ppi import PPInteractions

        ppi = PPInteractions(
            top=str(two_chain_pdb),
            traj=str(two_chain_pdb),
            out=tmp_path / "results.json",
            sel1="chainID A",
            sel2="chainID B",
            cov_cutoff=(11.0, 13.0),
            sb_cutoff=6.0,
            hbond_cutoff=3.5,
            hbond_angle=30.0,
            hydrophobic_cutoff=8.0,
            plot=False,
        )

        assert ppi.n_frames == 1  # single-frame PDB used as its own trajectory
        assert ppi.sel1 == "chainID A"
        assert ppi.sel2 == "chainID B"
        assert ppi.cov_cutoff == (11.0, 13.0)
        assert ppi.sb == 6.0
        assert ppi.hb_d == 3.5
        assert ppi.hydr == 8.0
        assert not ppi.plot

    def test_ppinteractions_hbond_angle_conversion(self, two_chain_pdb, tmp_path):
        """Test that hbond angle is converted on construction."""
        from molecular_simulations.analysis.cov_ppi import PPInteractions

        ppi = PPInteractions(
            top=str(two_chain_pdb),
            traj=str(two_chain_pdb),
            out=tmp_path / "results.json",
            hbond_angle=30.0,
            plot=False,
        )

        expected_angle = 30.0 * 180 / np.pi
        assert np.isclose(ppi.hb_a, expected_angle)

    def test_res_map(self, saltbridge_ppi):
        """Test res_map builds a per-atom residue index mapping."""
        ag1 = saltbridge_ppi.u.select_atoms("chainID A")
        ag2 = saltbridge_ppi.u.select_atoms("chainID B")

        saltbridge_ppi.res_map(ag1, ag2)

        # Chain A starts at resid 1 (ACE), chain B at resid 4 (ACE)
        assert saltbridge_ppi.mapping["ag1"][0] == 1
        assert saltbridge_ppi.mapping["ag2"][0] == 4

    def test_interpret_covariance(self, saltbridge_ppi):
        """Test interpret_covariance splits positive/negative correlations."""
        saltbridge_ppi.mapping = {"ag1": {0: 1, 1: 2}, "ag2": {0: 10, 1: 11}}

        cov_mat = np.array(
            [
                [0.5, -0.3],  # Residue 1: pos corr with 10, neg with 11
                [-0.2, 0.4],  # Residue 2: neg corr with 10, pos with 11
            ]
        )

        positive, negative = saltbridge_ppi.interpret_covariance(cov_mat)

        assert len(positive) > 0
        assert len(negative) > 0

    def test_identify_interaction_type(self, saltbridge_ppi):
        """Test identify_interaction_type for a charged Asp-Lys pair."""
        functions, labels = saltbridge_ppi.identify_interaction_type("ASP", "LYS")

        assert "saltbridge" in labels or "hydrophobic" in labels or "hbond" in labels

    def test_save_results(self, saltbridge_ppi):
        """Test save writes a JSON results file."""
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

        saltbridge_ppi.save(results)

        assert saltbridge_ppi.out.exists()
        with open(saltbridge_ppi.out) as f:
            loaded = json.load(f)
        assert "A_ALA1-B_LYS10" in loaded["positive"]

    def test_parse_results(self, saltbridge_ppi):
        """Test parse_results returns a structured DataFrame."""
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

        df = saltbridge_ppi.parse_results(results)

        assert isinstance(df, pl.DataFrame)
        assert "Residue Pair" in df.columns
        assert "Hydrophobic" in df.columns
        assert "Hydrogen Bond" in df.columns
        assert "Salt Bridge" in df.columns
        assert "Covariance" in df.columns

    def test_parse_results_filters_zeros(self, saltbridge_ppi):
        """Test that parse_results filters out all-zero entries."""
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

        df = saltbridge_ppi.parse_results(results)

        # Should only include the non-zero entry
        assert len(df) == 1
        assert "A_ALA1-B_LYS10" in df["Residue Pair"].to_list()


class TestEvaluateHBond:
    """Test the evaluate_hbond method on real donor/acceptor groups."""

    def test_evaluate_hbond_found(self, traj_ppi):
        """evaluate_hbond scores real Lys-donor / Asp-acceptor geometry."""
        lys = traj_ppi.u.select_atoms("chainID A and resname LYS")
        asp = traj_ppi.u.select_atoms("chainID B and resname ASP")

        donors, acceptors = traj_ppi.survey_donors_acceptors(lys, asp)
        result = traj_ppi.evaluate_hbond(donors, acceptors)

        # First frame has the amine donating to the carboxylate -> a hydrogen bond
        assert result == 1


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
    """Test compute_interactions method on the real Lys/Asp pair."""

    def test_compute_interactions(self, traj_ppi):
        """compute_interactions scores the real Lys(2)-Asp(5) interface."""
        result = traj_ppi.compute_interactions(2, 5)

        assert isinstance(result, dict)
        # Key format is 'A_<aa><resid>-B_<aa><resid>'
        assert "A_K2-B_D5" in result
        scores = result["A_K2-B_D5"]
        assert set(scores) == {"hydrophobic", "hbond", "saltbridge"}
        assert all(0.0 <= v <= 1.0 for v in scores.values())


class TestIdentifyInteractionType:
    """Test identify_interaction_type method (pure resname logic)."""

    def test_identify_interaction_type_polar(self, saltbridge_ppi):
        """Test interaction type identification for polar residues"""
        # Test SER-THR (should have hbond capability)
        functions, labels = saltbridge_ppi.identify_interaction_type("SER", "THR")
        assert "hydrophobic" in labels
        assert "hbond" in labels

    def test_identify_interaction_type_charged(self, saltbridge_ppi):
        """Test interaction type identification for charged residues"""
        # Test ASP-LYS (should have saltbridge capability)
        functions, labels = saltbridge_ppi.identify_interaction_type("ASP", "LYS")
        assert "hydrophobic" in labels
        assert "saltbridge" in labels

    def test_identify_interaction_type_hydrophobic(self, saltbridge_ppi):
        """Test interaction type identification for hydrophobic residues"""
        # Test ALA-VAL (hydrophobic only)
        functions, labels = saltbridge_ppi.identify_interaction_type("ALA", "VAL")
        assert "hydrophobic" in labels
        # ALA and VAL are not in the int_types dict, so only hydrophobic


class TestMakePlot:
    """Test make_plot method"""

    def test_make_plot(self, saltbridge_ppi, tmp_path):
        """Test make_plot renders a real figure file (Agg backend)."""
        data = pl.DataFrame(
            {
                "Residue Pair": ["A_ALA1-B_LYS10"],
                "Hydrophobic": [0.5],
                "Hydrogen Bond": [0.3],
                "Salt Bridge": [0.0],
                "Covariance": ["positive"],
            }
        )

        plot_path = tmp_path / "test_plot.png"
        saltbridge_ppi.make_plot(data, "Hydrophobic", plot_path)

        # A real PNG is written to disk
        assert plot_path.exists()
        assert plot_path.stat().st_size > 0


class TestPlotResults:
    """Test plot_results method (real figures via the Agg backend)."""

    def test_plot_results(self, saltbridge_ppi, tmp_path, monkeypatch):
        """plot_results writes a real PNG per non-zero interaction type."""
        # plot_results writes into a relative ./plots dir; sandbox it
        monkeypatch.chdir(tmp_path)

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

        saltbridge_ppi.plot_results(results)

        # Positive hydrophobic + hydrogen-bond columns are non-zero -> two PNGs
        pngs = sorted(p.name for p in (tmp_path / "plots").glob("*.png"))
        assert "Positive_Covariance_Hydrophobic.png" in pngs
        assert "Positive_Covariance_Hydrogen_Bond.png" in pngs


class TestSurveyDonorsAcceptors:
    """Test survey_donors_acceptors method on real residues."""

    def test_survey_donors_acceptors(self, traj_ppi):
        """survey_donors_acceptors finds real donor/acceptor atoms via bonds."""
        lys = traj_ppi.u.select_atoms("chainID A and resname LYS")
        asp = traj_ppi.u.select_atoms("chainID B and resname ASP")

        donors, acceptors = traj_ppi.survey_donors_acceptors(lys, asp)

        # Lys contributes amine N donors; Asp contributes carboxylate O acceptors
        assert donors.n_atoms > 0
        assert acceptors.n_atoms > 0


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
    """Test PPInteractions.run() end-to-end on the real trajectory."""

    def test_run_writes_results(self, traj_ppi):
        """run() drives covariance -> interpret -> compute -> save for real."""
        traj_ppi.run()

        assert traj_ppi.out.exists()
        data = json.loads(traj_ppi.out.read_text())
        assert set(data) == {"positive", "negative"}


class TestPPInteractionsComputeInteractions:
    """compute_interactions on the real Lys/Asp pair (cf. TestComputeInteractions)."""

    def test_compute_interactions_returns_data(self, traj_ppi):
        """compute_interactions returns a dict keyed by the residue pair."""
        result = traj_ppi.compute_interactions(2, 5)

        assert isinstance(result, dict)
        key = next(iter(result))
        assert "hydrophobic" in result[key]


class TestPPInteractionsAnalyzeSaltbridge:
    """analyze_saltbridge charge-compatibility (cf. TestAnalyzeSaltbridge)."""

    def test_saltbridge_non_charged_returns_zero(self, saltbridge_ppi):
        """Saltbridge returns 0 for a non-charged residue."""
        res1 = SimpleNamespace(resnames=["ALA"])
        res2 = SimpleNamespace(resnames=["GLU"])

        assert saltbridge_ppi.analyze_saltbridge(res1, res2) == 0.0

    def test_saltbridge_same_charge_returns_zero(self, saltbridge_ppi):
        """Saltbridge returns 0 for two like-charged residues."""
        both_positive = saltbridge_ppi.analyze_saltbridge(
            SimpleNamespace(resnames=["LYS"]), SimpleNamespace(resnames=["ARG"])
        )
        both_negative = saltbridge_ppi.analyze_saltbridge(
            SimpleNamespace(resnames=["ASP"]), SimpleNamespace(resnames=["GLU"])
        )

        assert both_positive == 0.0
        assert both_negative == 0.0


class TestPPInteractionsGetCovariance:
    """Test PPInteractions.get_covariance() on the real trajectory."""

    def test_get_covariance_returns_matrix(self, traj_ppi):
        """get_covariance returns a real chain-A x chain-B covariance matrix."""
        C = traj_ppi.get_covariance()

        assert isinstance(C, np.ndarray)
        assert C.ndim == 2
        # One titratable CA per chain (Lys / Asp) -> 1x1
        assert C.shape == (1, 1)


class TestPPInteractionsAnalyzeSaltbridgeWithTrajectory:
    """Test saltbridge with a real charged pair over the trajectory."""

    def test_saltbridge_valid_pair_with_trajectory(self, traj_ppi):
        """Real Lys-Asp saltbridge occupancy over the drifting trajectory."""
        lys = traj_ppi.u.select_atoms("chainID A and resname LYS")
        asp = traj_ppi.u.select_atoms("chainID B and resname ASP")

        result = traj_ppi.analyze_saltbridge(lys, asp)

        assert isinstance(result, float)
        # Present in the first frame, broken as chain B drifts away
        assert result == 0.2


class TestPPInteractionsRunWithPlot:
    """Test run() with plot=True drives the real plotting branch."""

    def test_run_with_plot(self, two_chain_trajectory, tmp_path, monkeypatch):
        """run(plot=True) completes and writes results (plots go to ./plots)."""
        from molecular_simulations.analysis.cov_ppi import PPInteractions

        # plot_results writes into a relative ./plots dir; sandbox it
        monkeypatch.chdir(tmp_path)

        ppi = PPInteractions(
            top=str(two_chain_trajectory["top"]),
            traj=str(two_chain_trajectory["traj"]),
            out=tmp_path / "results.json",
            plot=True,
        )
        ppi.run()

        assert (tmp_path / "results.json").exists()


class TestPPInteractionsSave:
    """Test PPInteractions.save() (cf. TestPPInteractions.test_save_results)."""

    def test_save_creates_json_file(self, saltbridge_ppi):
        """save writes the results dict as JSON round-trippable on disk."""
        results = {
            "positive": {"A_ALA1-B_GLY10": {"hydrophobic": 0.5}},
            "negative": {},
        }
        saltbridge_ppi.save(results)

        assert saltbridge_ppi.out.exists()
        loaded = json.loads(saltbridge_ppi.out.read_text())
        assert loaded == results


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
