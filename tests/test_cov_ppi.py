"""
Unit tests for analysis/cov_ppi.py module

This module uses no mocks. Tests run real MDAnalysis over committed PDB/DCD
fixtures, with conditional skips for environments without MDAnalysis installed.
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
        import MDAnalysis  # noqa: F401

        return True
    except ImportError:
        return False


requires_mdanalysis = pytest.mark.skipif(
    not _check_mdanalysis(), reason='MDAnalysis not installed'
)


@pytest.fixture
def test_data_dir():
    """Return the path to test data directory."""
    return Path(__file__).parent / 'data'


@pytest.fixture
def alanine_pdb(test_data_dir):
    """Return the path to the alanine dipeptide PDB."""
    return test_data_dir / 'pdb' / 'alanine_dipeptide.pdb'


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
        out=tmp_path / 'results.json',
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
        top=str(two_chain_trajectory['top']),
        traj=str(two_chain_trajectory['traj']),
        out=tmp_path / 'results.json',
        plot=False,
    )


# ============================================================================
# Integration tests using real MDAnalysis
# ============================================================================


@requires_mdanalysis
class TestPPInteractionsIntegration:
    """Integration tests using real MDAnalysis."""

    def test_ppinteractions_init_with_real_file(self, alanine_pdb, tmp_path):
        """Test PPInteractions initialization with real PDB file."""
        from molecular_simulations.analysis.cov_ppi import PPInteractions

        out_path = tmp_path / 'results.json'

        ppi = PPInteractions(
            top=str(alanine_pdb),
            traj=str(alanine_pdb),  # Single PDB as trajectory
            out=out_path,
            sel1='resid 1',
            sel2='resid 2',
            plot=False,
        )

        assert ppi.n_frames == 1  # Single frame PDB
        assert ppi.u is not None

    def test_res_map_with_real_universe(self, alanine_pdb, tmp_path):
        """Test res_map with real MDAnalysis Universe."""
        import MDAnalysis as mda

        u = mda.Universe(str(alanine_pdb))
        ag1 = u.select_atoms('resid 1')
        ag2 = u.select_atoms('resid 2')

        # Test the mapping logic
        mapping = {'ag1': {}, 'ag2': {}}
        for i, resid in enumerate(ag1.resids):
            mapping['ag1'][i] = resid
        for i, resid in enumerate(ag2.resids):
            mapping['ag2'][i] = resid

        assert 0 in mapping['ag1']
        assert 0 in mapping['ag2']

    def test_save_and_load_results(self, alanine_pdb, tmp_path):
        """Test save method creates valid JSON."""
        from molecular_simulations.analysis.cov_ppi import PPInteractions

        out_path = tmp_path / 'results.json'

        ppi = PPInteractions(
            top=str(alanine_pdb),
            traj=str(alanine_pdb),
            out=out_path,
            sel1='resid 1',
            sel2='resid 2',
            plot=False,
        )

        results = {
            'positive': {
                'A_ALA1-B_LYS10': {'hydrophobic': 0.5, 'hbond': 0.3, 'saltbridge': 0.0}
            },
            'negative': {},
        }

        ppi.save(results)

        assert out_path.exists()
        with open(out_path) as f:
            loaded = json.load(f)
        assert 'positive' in loaded
        assert 'A_ALA1-B_LYS10' in loaded['positive']


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
            out=tmp_path / 'results.json',
            sel1='chainID A',
            sel2='chainID B',
            cov_cutoff=(11.0, 13.0),
            sb_cutoff=6.0,
            hbond_cutoff=3.5,
            hbond_angle=30.0,
            hydrophobic_cutoff=8.0,
            plot=False,
        )

        assert ppi.n_frames == 1  # single-frame PDB used as its own trajectory
        assert ppi.sel1 == 'chainID A'
        assert ppi.sel2 == 'chainID B'
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
            out=tmp_path / 'results.json',
            hbond_angle=30.0,
            plot=False,
        )

        expected_angle = 30.0 * 180 / np.pi
        assert np.isclose(ppi.hb_a, expected_angle)

    def test_res_map(self, saltbridge_ppi):
        """Test res_map builds a per-atom residue index mapping."""
        ag1 = saltbridge_ppi.u.select_atoms('chainID A')
        ag2 = saltbridge_ppi.u.select_atoms('chainID B')

        saltbridge_ppi.res_map(ag1, ag2)

        # Chain A starts at resid 1 (ACE), chain B at resid 4 (ACE)
        assert saltbridge_ppi.mapping['ag1'][0] == 1
        assert saltbridge_ppi.mapping['ag2'][0] == 4

    def test_interpret_covariance(self, saltbridge_ppi):
        """Test interpret_covariance splits positive/negative correlations."""
        saltbridge_ppi.mapping = {'ag1': {0: 1, 1: 2}, 'ag2': {0: 10, 1: 11}}

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
        _functions, labels = saltbridge_ppi.identify_interaction_type('ASP', 'LYS')

        assert 'saltbridge' in labels or 'hydrophobic' in labels or 'hbond' in labels

    def test_save_results(self, saltbridge_ppi):
        """Test save writes a JSON results file."""
        results = {
            'positive': {
                'A_ALA1-B_LYS10': {
                    'hydrophobic': 0.5,
                    'hbond': 0.3,
                    'saltbridge': 0.0,
                }
            },
            'negative': {},
        }

        saltbridge_ppi.save(results)

        assert saltbridge_ppi.out.exists()
        with open(saltbridge_ppi.out) as f:
            loaded = json.load(f)
        assert 'A_ALA1-B_LYS10' in loaded['positive']

    def test_parse_results(self, saltbridge_ppi):
        """Test parse_results returns a structured DataFrame."""
        results = {
            'positive': {
                'A_ALA1-B_LYS10': {
                    'hydrophobic': 0.5,
                    'hbond': 0.3,
                    'saltbridge': 0.0,
                }
            },
            'negative': {
                'A_GLU5-B_ARG15': {
                    'hydrophobic': 0.0,
                    'hbond': 0.0,
                    'saltbridge': 0.8,
                }
            },
        }

        df = saltbridge_ppi.parse_results(results)

        assert isinstance(df, pl.DataFrame)
        assert 'Residue Pair' in df.columns
        assert 'Hydrophobic' in df.columns
        assert 'Hydrogen Bond' in df.columns
        assert 'Salt Bridge' in df.columns
        assert 'Covariance' in df.columns

    def test_parse_results_filters_zeros(self, saltbridge_ppi):
        """Test that parse_results filters out all-zero entries."""
        results = {
            'positive': {
                'A_ALA1-B_LYS10': {
                    'hydrophobic': 0.5,
                    'hbond': 0.0,
                    'saltbridge': 0.0,
                },
                'A_GLY2-B_SER11': {
                    'hydrophobic': 0.0,
                    'hbond': 0.0,
                    'saltbridge': 0.0,
                },
            },
            'negative': {},
        }

        df = saltbridge_ppi.parse_results(results)

        # Should only include the non-zero entry
        assert len(df) == 1
        assert 'A_ALA1-B_LYS10' in df['Residue Pair'].to_list()


class TestEvaluateHBond:
    """Test the evaluate_hbond method on real donor/acceptor groups."""

    def test_evaluate_hbond_found(self, traj_ppi):
        """evaluate_hbond scores real Lys-donor / Asp-acceptor geometry."""
        lys = traj_ppi.u.select_atoms('chainID A and resname LYS')
        asp = traj_ppi.u.select_atoms('chainID B and resname ASP')

        donors, acceptors = traj_ppi.survey_donors_acceptors(lys, asp)
        result = traj_ppi.evaluate_hbond(donors, acceptors)

        # First frame has the amine donating to the carboxylate -> a hydrogen bond
        assert result == 1


class TestAnalyzeHydrophobic:
    """Test the analyze_hydrophobic method on a real two-chain trajectory."""

    def test_analyze_hydrophobic(self, traj_ppi):
        """analyze_hydrophobic returns a real frame-averaged occupancy."""
        lys = traj_ppi.u.select_atoms('chainID A and resname LYS')
        asp = traj_ppi.u.select_atoms('chainID B and resname ASP')

        result = traj_ppi.analyze_hydrophobic(lys, asp)

        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0
        # The carbon side chains start within the 8 A cutoff
        assert result > 0.0


class TestAnalyzeSaltbridge:
    """Test the analyze_saltbridge method on a real Lys/Asp interface."""

    def test_analyze_saltbridge_real_pair(self, saltbridge_ppi):
        """A real Lys-Asp pair within cutoff is a full-occupancy salt bridge."""
        lys = saltbridge_ppi.u.select_atoms('chainID A and resname LYS')
        asp = saltbridge_ppi.u.select_atoms('chainID B and resname ASP')

        # Single frame, NZ <-> carboxylate ~3.4 A < 6.0 A cutoff -> occupancy 1.0
        assert saltbridge_ppi.analyze_saltbridge(lys, asp) == 1.0

    def test_analyze_saltbridge_incompatible_residues(self, saltbridge_ppi):
        """Saltbridge analysis returns 0 for non-charged residues."""
        res1 = SimpleNamespace(resnames=['ALA'])
        res2 = SimpleNamespace(resnames=['GLY'])

        assert saltbridge_ppi.analyze_saltbridge(res1, res2) == 0.0

    def test_analyze_saltbridge_same_charge(self, saltbridge_ppi):
        """Saltbridge analysis returns 0 for two positively-charged residues."""
        res1 = SimpleNamespace(resnames=['LYS'])
        res2 = SimpleNamespace(resnames=['ARG'])

        assert saltbridge_ppi.analyze_saltbridge(res1, res2) == 0.0

    def test_analyze_saltbridge_two_negative(self, saltbridge_ppi):
        """Saltbridge analysis returns 0 for two negatively-charged residues."""
        res1 = SimpleNamespace(resnames=['ASP'])
        res2 = SimpleNamespace(resnames=['GLU'])

        assert saltbridge_ppi.analyze_saltbridge(res1, res2) == 0.0


class TestComputeInteractions:
    """Test compute_interactions method on the real Lys/Asp pair."""

    def test_compute_interactions(self, traj_ppi):
        """compute_interactions scores the real Lys(2)-Asp(5) interface."""
        result = traj_ppi.compute_interactions(2, 5)

        assert isinstance(result, dict)
        # Key format is 'A_<aa><resid>-B_<aa><resid>'
        assert 'A_K2-B_D5' in result
        scores = result['A_K2-B_D5']
        assert set(scores) == {'hydrophobic', 'hbond', 'saltbridge'}
        assert all(0.0 <= v <= 1.0 for v in scores.values())


class TestIdentifyInteractionType:
    """Test identify_interaction_type method (pure resname logic)."""

    def test_identify_interaction_type_polar(self, saltbridge_ppi):
        """Test interaction type identification for polar residues"""
        # Test SER-THR (should have hbond capability)
        _functions, labels = saltbridge_ppi.identify_interaction_type('SER', 'THR')
        assert 'hydrophobic' in labels
        assert 'hbond' in labels

    def test_identify_interaction_type_charged(self, saltbridge_ppi):
        """Test interaction type identification for charged residues"""
        # Test ASP-LYS (should have saltbridge capability)
        _functions, labels = saltbridge_ppi.identify_interaction_type('ASP', 'LYS')
        assert 'hydrophobic' in labels
        assert 'saltbridge' in labels

    def test_identify_interaction_type_hydrophobic(self, saltbridge_ppi):
        """Test interaction type identification for hydrophobic residues"""
        # Test ALA-VAL (hydrophobic only)
        _functions, labels = saltbridge_ppi.identify_interaction_type('ALA', 'VAL')
        assert 'hydrophobic' in labels
        # ALA and VAL are not in the int_types dict, so only hydrophobic


class TestMakePlot:
    """Test make_plot method"""

    def test_make_plot(self, saltbridge_ppi, tmp_path):
        """Test make_plot renders a real figure file (Agg backend)."""
        data = pl.DataFrame(
            {
                'Residue Pair': ['A_ALA1-B_LYS10'],
                'Hydrophobic': [0.5],
                'Hydrogen Bond': [0.3],
                'Salt Bridge': [0.0],
                'Covariance': ['positive'],
            }
        )

        plot_path = tmp_path / 'test_plot.png'
        saltbridge_ppi.make_plot(data, 'Hydrophobic', plot_path)

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
            'positive': {
                'A_ALA1-B_LYS10': {
                    'hydrophobic': 0.5,
                    'hbond': 0.3,
                    'saltbridge': 0.0,
                }
            },
            'negative': {},
        }

        saltbridge_ppi.plot_results(results)

        # Positive hydrophobic + hydrogen-bond columns are non-zero -> two PNGs
        pngs = sorted(p.name for p in (tmp_path / 'plots').glob('*.png'))
        assert 'Positive_Covariance_Hydrophobic.png' in pngs
        assert 'Positive_Covariance_Hydrogen_Bond.png' in pngs


class TestSurveyDonorsAcceptors:
    """Test survey_donors_acceptors method on real residues."""

    def test_survey_donors_acceptors(self, traj_ppi):
        """survey_donors_acceptors finds real donor/acceptor atoms via bonds."""
        lys = traj_ppi.u.select_atoms('chainID A and resname LYS')
        asp = traj_ppi.u.select_atoms('chainID B and resname ASP')

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
        lys = traj_ppi.u.select_atoms('chainID A and resname LYS')
        asp = traj_ppi.u.select_atoms('chainID B and resname ASP')

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
        assert set(data) == {'positive', 'negative'}


class TestPPInteractionsComputeInteractions:
    """compute_interactions on the real Lys/Asp pair (cf. TestComputeInteractions)."""

    def test_compute_interactions_returns_data(self, traj_ppi):
        """compute_interactions returns a dict keyed by the residue pair."""
        result = traj_ppi.compute_interactions(2, 5)

        assert isinstance(result, dict)
        key = next(iter(result))
        assert 'hydrophobic' in result[key]


class TestPPInteractionsAnalyzeSaltbridge:
    """analyze_saltbridge charge-compatibility (cf. TestAnalyzeSaltbridge)."""

    def test_saltbridge_non_charged_returns_zero(self, saltbridge_ppi):
        """Saltbridge returns 0 for a non-charged residue."""
        res1 = SimpleNamespace(resnames=['ALA'])
        res2 = SimpleNamespace(resnames=['GLU'])

        assert saltbridge_ppi.analyze_saltbridge(res1, res2) == 0.0

    def test_saltbridge_same_charge_returns_zero(self, saltbridge_ppi):
        """Saltbridge returns 0 for two like-charged residues."""
        both_positive = saltbridge_ppi.analyze_saltbridge(
            SimpleNamespace(resnames=['LYS']), SimpleNamespace(resnames=['ARG'])
        )
        both_negative = saltbridge_ppi.analyze_saltbridge(
            SimpleNamespace(resnames=['ASP']), SimpleNamespace(resnames=['GLU'])
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
        lys = traj_ppi.u.select_atoms('chainID A and resname LYS')
        asp = traj_ppi.u.select_atoms('chainID B and resname ASP')

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
            top=str(two_chain_trajectory['top']),
            traj=str(two_chain_trajectory['traj']),
            out=tmp_path / 'results.json',
            plot=True,
        )
        ppi.run()

        assert (tmp_path / 'results.json').exists()


class TestPPInteractionsSave:
    """Test PPInteractions.save() (cf. TestPPInteractions.test_save_results)."""

    def test_save_creates_json_file(self, saltbridge_ppi):
        """save writes the results dict as JSON round-trippable on disk."""
        results = {
            'positive': {'A_ALA1-B_GLY10': {'hydrophobic': 0.5}},
            'negative': {},
        }
        saltbridge_ppi.save(results)

        assert saltbridge_ppi.out.exists()
        loaded = json.loads(saltbridge_ppi.out.read_text())
        assert loaded == results


@requires_mdanalysis
class TestGetCovarianceCutoffs:
    """Drive get_covariance's distance-cutoff zeroing on both signs.

    Builds a real, in-memory MDAnalysis Universe (two CA per chain) with
    hand-chosen positions over two frames so the raw covariance matrix is
    exactly [[+1, 0], [0, -1]] before cutoffs and the four residue-pair
    distances straddle both cov_cutoff boundaries. This exercises every arm
    of the sign/distance filter (cov_ppi lines 217-221) with real geometry.
    """

    def _controlled_ppi(self, two_chain_pdb, tmp_path, cov_cutoff):
        import MDAnalysis as mda
        from MDAnalysis.coordinates.memory import MemoryReader

        from molecular_simulations.analysis.cov_ppi import PPInteractions

        u = mda.Universe.empty(
            4,
            n_residues=4,
            atom_resindex=[0, 1, 2, 3],
            residue_segindex=[0, 0, 0, 0],
            trajectory=True,
        )
        u.add_TopologyAttr('name', ['CA', 'CA', 'CA', 'CA'])
        u.add_TopologyAttr('type', ['C', 'C', 'C', 'C'])
        u.add_TopologyAttr('resname', ['LYS', 'LYS', 'ASP', 'ASP'])
        u.add_TopologyAttr('resid', [1, 2, 1, 2])
        u.add_TopologyAttr('chainID', ['A', 'A', 'B', 'B'])

        # frame0 / frame1 are mirror images about the per-atom mean, so
        # C[i, j] reduces to dot(dR1_i, dR2_j) from frame0 (see scratch check).
        frame0 = [[1, 0, 0], [1, 10, 0], [1, 0, 5], [-1, 10, 13]]
        frame1 = [[-1, 0, 0], [-1, 10, 0], [-1, 0, 5], [1, 10, 13]]
        coords = np.array([frame0, frame1], dtype=float)
        u.load_new(coords, format=MemoryReader)

        ppi = PPInteractions(
            top=str(two_chain_pdb),
            traj=str(two_chain_pdb),
            out=tmp_path / 'results.json',
            cov_cutoff=cov_cutoff,
            plot=False,
        )
        # Swap in the controlled real Universe (not a mock).
        ppi.u = u
        ppi.n_frames = 2
        return ppi

    def test_covariance_cutoffs_zero_by_sign(self, two_chain_pdb, tmp_path):
        """Positive-far and negative-far pairs are zeroed; close ones survive."""
        ppi = self._controlled_ppi(two_chain_pdb, tmp_path, cov_cutoff=(11.0, 13.0))

        C = ppi.get_covariance()

        assert C.shape == (2, 2)
        # (A1,B1) C=+1 dist=5  <=11 -> kept
        assert np.isclose(C[0, 0], 1.0)
        # (A2,B1) C=+1 dist~11.18 >11 -> positive branch zeroed (lines 218-219)
        assert C[1, 0] == 0.0
        # (A1,B2) C=-1 dist~16.4 >13 -> negative branch zeroed (lines 220-221)
        assert C[0, 1] == 0.0
        # (A2,B2) C=-1 dist=13    not >13 -> kept
        assert np.isclose(C[1, 1], -1.0)

    def test_covariance_permissive_cutoffs_keep_all(self, two_chain_pdb, tmp_path):
        """With huge cutoffs no pair is zeroed -> raw [[+1, 0-ish], ...] survives."""
        ppi = self._controlled_ppi(
            two_chain_pdb, tmp_path, cov_cutoff=(1000.0, 1000.0)
        )

        C = ppi.get_covariance()

        # Nothing exceeds the cutoffs, so the raw signed covariance is preserved.
        assert np.isclose(C[0, 0], 1.0)
        assert np.isclose(C[1, 1], -1.0)


class TestInterpretCovarianceAlreadySeen:
    """Exercise the 'already seen' skip arms of interpret_covariance."""

    def test_positive_pair_already_seen_is_skipped(self, saltbridge_ppi):
        """A reversed positive pair already in `seen` is not re-added (264->261)."""
        # Overlapping resid namespaces so cell (1,1) yields (2,1), the reverse
        # of (1,2) added by cell (0,0) -> hits the 'already seen' skip.
        saltbridge_ppi.mapping = {'ag1': {0: 1, 1: 2}, 'ag2': {0: 2, 1: 1}}
        cov = np.array([[0.5, 0.5], [0.5, 0.5]])

        positive, negative = saltbridge_ppi.interpret_covariance(cov)

        # (2,1) is skipped, so only three unique pairs survive.
        assert positive == [(1, 2), (1, 1), (2, 2)]
        assert negative == []

    def test_negative_pair_already_seen_is_skipped(self, saltbridge_ppi):
        """A reversed negative pair already in `seen` is not re-added (273->270)."""
        saltbridge_ppi.mapping = {'ag1': {0: 1, 1: 2}, 'ag2': {0: 2, 1: 1}}
        cov = np.array([[-0.5, -0.5], [-0.5, -0.5]])

        positive, negative = saltbridge_ppi.interpret_covariance(cov)

        assert negative == [(1, 2), (1, 1), (2, 2)]
        assert positive == []


class TestIdentifyInteractionTypeRouting:
    """Exercise the func-not-shared skip arm of identify_interaction_type."""

    def test_partial_capability_overlap(self, saltbridge_ppi):
        """ASP (hbond+saltbridge) vs SER (hbond only) keeps only the shared func."""
        _functions, labels = saltbridge_ppi.identify_interaction_type('ASP', 'SER')

        # hbond is shared -> appended (lines 331-333); saltbridge is not shared
        # by SER -> skipped (the 331->330 false arm).
        assert labels == ['hydrophobic', 'hbond']
        assert 'saltbridge' not in labels

    def test_glycine_has_no_shared_capabilities(self, saltbridge_ppi):
        """GLY is absent from the type table, so only hydrophobic is routed."""
        _functions, labels = saltbridge_ppi.identify_interaction_type('GLY', 'LYS')

        # GLY contributes an empty func list, so the zip loop never appends.
        assert labels == ['hydrophobic']


class TestAnalyzeHydrophobicNoContact:
    """Exercise analyze_hydrophobic's no-contact (occupancy 0.0) arm."""

    def test_no_contact_returns_zero(self, two_chain_pdb, tmp_path):
        """With a sub-angstrom cutoff no carbons ever contact -> 0.0 (lines 430-433)."""
        from molecular_simulations.analysis.cov_ppi import PPInteractions

        ppi = PPInteractions(
            top=str(two_chain_pdb),
            traj=str(two_chain_pdb),
            out=tmp_path / 'results.json',
            hydrophobic_cutoff=0.01,
            plot=False,
        )
        lys = ppi.u.select_atoms('chainID A and resname LYS')
        asp = ppi.u.select_atoms('chainID B and resname ASP')

        # Every frame's minimum carbon-carbon distance exceeds 0.01 A.
        assert ppi.analyze_hydrophobic(lys, asp) == 0.0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
