"""
Unit tests for build/build_amber.py module
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


class TestImplicitSolvent:
    """Test suite for ImplicitSolvent class"""

    def test_implicit_solvent_init_with_path(self):
        """Test ImplicitSolvent initialization with path"""
        from molecular_simulations.build.build_amber import ImplicitSolvent

        with tempfile.TemporaryDirectory() as tmpdir:
            pdb_path = Path(tmpdir) / "test.pdb"
            pdb_path.write_text(
                "ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00\n"
            )

            with patch.dict(os.environ, {"AMBERHOME": "/fake/amber"}):
                builder = ImplicitSolvent(path=tmpdir, pdb=str(pdb_path), protein=True)

            assert builder.path == Path(tmpdir).resolve()
            assert "leaprc.protein.ff19SB" in builder.ffs

    def test_implicit_solvent_init_none_path(self):
        """Test ImplicitSolvent with path=None uses pdb directory"""
        from molecular_simulations.build.build_amber import ImplicitSolvent

        with tempfile.TemporaryDirectory() as tmpdir:
            pdb_path = Path(tmpdir) / "test.pdb"
            pdb_path.write_text(
                "ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00\n"
            )

            with patch.dict(os.environ, {"AMBERHOME": "/fake/amber"}):
                builder = ImplicitSolvent(path=None, pdb=str(pdb_path), protein=True)

            assert builder.path == Path(tmpdir).resolve()

    def test_implicit_solvent_no_amberhome(self):
        """Test ImplicitSolvent raises error when AMBERHOME not set"""
        from molecular_simulations.build.build_amber import ImplicitSolvent

        with tempfile.TemporaryDirectory() as tmpdir:
            pdb_path = Path(tmpdir) / "test.pdb"
            pdb_path.write_text(
                "ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00\n"
            )

            # Remove AMBERHOME if it exists
            env = os.environ.copy()
            env.pop("AMBERHOME", None)

            with patch.dict(os.environ, env, clear=True):
                with pytest.raises(ValueError, match="AMBERHOME is not set"):
                    ImplicitSolvent(path=tmpdir, pdb=str(pdb_path), amberhome=None)

    def test_implicit_solvent_custom_output(self):
        """Test ImplicitSolvent with custom output path"""
        from molecular_simulations.build.build_amber import ImplicitSolvent

        with tempfile.TemporaryDirectory() as tmpdir:
            pdb_path = Path(tmpdir) / "test.pdb"
            pdb_path.write_text(
                "ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00\n"
            )

            with patch.dict(os.environ, {"AMBERHOME": "/fake/amber"}):
                builder = ImplicitSolvent(
                    path=tmpdir, pdb=str(pdb_path), out="custom_output.pdb"
                )

            assert "custom_output.pdb" in str(builder.out)

    def test_implicit_solvent_forcefields(self):
        """Test forcefield selection based on switches"""
        from molecular_simulations.build.build_amber import ImplicitSolvent

        with tempfile.TemporaryDirectory() as tmpdir:
            pdb_path = Path(tmpdir) / "test.pdb"
            pdb_path.write_text(
                "ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00\n"
            )

            with patch.dict(os.environ, {"AMBERHOME": "/fake/amber"}):
                # Test with multiple forcefields
                builder = ImplicitSolvent(
                    path=tmpdir, pdb=str(pdb_path), protein=True, rna=True, dna=True
                )

            assert "leaprc.protein.ff19SB" in builder.ffs
            assert "leaprc.RNA.Shaw" in builder.ffs
            assert "leaprc.DNA.OL21" in builder.ffs

    def test_implicit_solvent_write_leap(self):
        """Test tleap_it method writes correct leap input"""
        from molecular_simulations.build.build_amber import ImplicitSolvent

        with tempfile.TemporaryDirectory() as tmpdir:
            pdb_path = Path(tmpdir) / "test.pdb"
            pdb_path.write_text(
                "ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00\n"
            )

            with patch.dict(os.environ, {"AMBERHOME": "/fake/amber"}):
                builder = ImplicitSolvent(path=tmpdir, pdb=str(pdb_path))

            builder.debug = True
            with patch("subprocess.run"):
                builder.tleap_it()

            leap_file = Path(tmpdir) / "tleap.in"
            assert leap_file.exists()
            content = leap_file.read_text()
            assert "leaprc.protein.ff19SB" in content
            assert "loadpdb" in content


class TestExplicitSolvent:
    """Test suite for ExplicitSolvent class"""

    def test_explicit_solvent_init(self):
        """Test ExplicitSolvent initialization"""
        from molecular_simulations.build.build_amber import ExplicitSolvent

        with tempfile.TemporaryDirectory() as tmpdir:
            pdb_path = Path(tmpdir) / "test.pdb"
            pdb_path.write_text(
                "ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00\n"
            )

            with patch.dict(os.environ, {"AMBERHOME": "/fake/amber"}):
                builder = ExplicitSolvent(path=tmpdir, pdb=str(pdb_path), padding=15.0)

            assert builder.pad == 15.0
            assert "leaprc.water.opc" in builder.ffs
            assert builder.water_box == "OPCBOX"

    def test_explicit_solvent_polarizable(self):
        """Test ExplicitSolvent with polarizable forcefield"""
        from molecular_simulations.build.build_amber import ExplicitSolvent

        with tempfile.TemporaryDirectory() as tmpdir:
            pdb_path = Path(tmpdir) / "test.pdb"
            pdb_path.write_text(
                "ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00\n"
            )

            with patch.dict(os.environ, {"AMBERHOME": "/fake/amber"}):
                builder = ExplicitSolvent(
                    path=tmpdir, pdb=str(pdb_path), polarizable=True
                )

            assert "leaprc.protein.ff15ipq" in builder.ffs
            assert "leaprc.water.spceb" in builder.ffs
            assert builder.water_box == "SPCBOX"

    def test_get_ion_numbers(self):
        """Test get_ion_numbers static method"""
        from molecular_simulations.build.build_amber import ExplicitSolvent

        # Test with known volume
        volume = 1000000  # 100 nm^3 in cubic Angstroms
        num_ions = ExplicitSolvent.get_ion_numbers(volume)

        # Should be a positive integer
        assert isinstance(num_ions, int)
        assert num_ions > 0

    def test_get_ion_numbers_different_volumes(self):
        """Test get_ion_numbers with different volumes"""
        from molecular_simulations.build.build_amber import ExplicitSolvent

        small_volume = 125000  # 50 Å cube
        large_volume = 1000000  # 100 Å cube

        small_ions = ExplicitSolvent.get_ion_numbers(small_volume)
        large_ions = ExplicitSolvent.get_ion_numbers(large_volume)

        # Larger volume should need more ions
        assert large_ions > small_ions

    def test_get_pdb_extent(self):
        """Test get_pdb_extent method"""
        from molecular_simulations.build.build_amber import ExplicitSolvent

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create PDB with known coordinates
            pdb_path = Path(tmpdir) / "test.pdb"
            pdb_content = """ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00
ATOM      2  CA  ALA A   1      10.000   5.000   3.000  1.00  0.00
ATOM      3  C   ALA A   1       5.000  15.000   8.000  1.00  0.00
"""
            pdb_path.write_text(pdb_content)

            with patch.dict(os.environ, {"AMBERHOME": "/fake/amber"}):
                builder = ExplicitSolvent(path=tmpdir, pdb=str(pdb_path), padding=10.0)

            # Need to set pdb path explicitly for the method
            builder.pdb = str(pdb_path)

            extent = builder.get_pdb_extent()

            # X extent: 10-0 = 10
            # Y extent: 15-0 = 15
            # Z extent: 8-0 = 8
            # Max is 15, plus 2*10 padding = 35
            assert extent == 35


class TestBuildWorkflow:
    """Test the build workflow methods"""

    @patch("molecular_simulations.build.build_amber.subprocess")
    def test_temp_tleap(self, mock_subprocess):
        """Test temp_tleap method"""
        from molecular_simulations.build.build_amber import ImplicitSolvent

        mock_subprocess.run.return_value = MagicMock(returncode=0)

        with tempfile.TemporaryDirectory() as tmpdir:
            pdb_path = Path(tmpdir) / "test.pdb"
            pdb_path.write_text(
                "ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00\n"
            )

            with patch.dict(os.environ, {"AMBERHOME": "/fake/amber"}):
                builder = ImplicitSolvent(
                    path=tmpdir, pdb=str(pdb_path), delete_temp_file=True
                )

            tleap_input = """source leaprc.protein.ff19SB
prot = loadpdb test.pdb
quit
"""
            builder.temp_tleap(tleap_input)

            # Should have called subprocess.run
            mock_subprocess.run.assert_called_once()

    @patch("molecular_simulations.build.build_amber.subprocess")
    def test_prep_pdb(self, mock_subprocess):
        """Test prep_pdb method"""
        from molecular_simulations.build.build_amber import ExplicitSolvent

        mock_subprocess.run.return_value = MagicMock(returncode=0)

        with tempfile.TemporaryDirectory() as tmpdir:
            pdb_path = Path(tmpdir) / "test.pdb"
            pdb_path.write_text(
                "ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00\n"
            )

            with patch.dict(os.environ, {"AMBERHOME": "/fake/amber"}):
                builder = ExplicitSolvent(path=tmpdir, pdb=str(pdb_path))

            builder.prep_pdb()

            # Should have run pdb4amber
            mock_subprocess.run.assert_called_once()

            # pdb should be updated to protein.pdb
            assert "protein.pdb" in builder.pdb

    def test_clean_up_directory(self):
        """Test clean_up_directory method"""
        from molecular_simulations.build.build_amber import ExplicitSolvent

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            pdb_path = tmpdir / "test.pdb"
            pdb_path.write_text(
                "ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00\n"
            )

            # Create some files that should be moved
            (tmpdir / "leap.log").write_text("log content")
            (tmpdir / "protein.pdb").write_text("pdb content")

            # Create files that should NOT be moved
            (tmpdir / "system.prmtop").write_text("topology")
            (tmpdir / "system.inpcrd").write_text("coordinates")

            with patch.dict(os.environ, {"AMBERHOME": "/fake/amber"}):
                builder = ExplicitSolvent(path=str(tmpdir), pdb=str(pdb_path))

            builder.clean_up_directory()

            # Check build directory exists
            assert (tmpdir / "build").exists()

            # prmtop and inpcrd should still be in main directory
            assert (tmpdir / "system.prmtop").exists()
            assert (tmpdir / "system.inpcrd").exists()


class TestKwargsHandling:
    """Test handling of extra keyword arguments"""

    def test_kwargs_set_as_attributes(self):
        """Test that kwargs are set as instance attributes"""
        from molecular_simulations.build.build_amber import ImplicitSolvent

        with tempfile.TemporaryDirectory() as tmpdir:
            pdb_path = Path(tmpdir) / "test.pdb"
            pdb_path.write_text(
                "ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00\n"
            )

            with patch.dict(os.environ, {"AMBERHOME": "/fake/amber"}):
                builder = ImplicitSolvent(
                    path=tmpdir,
                    pdb=str(pdb_path),
                    custom_param="custom_value",
                    another_param=42,
                )

            assert builder.custom_param == "custom_value"
            assert builder.another_param == 42


class TestExplicitSolventBuild:
    """Test suite for ExplicitSolvent build workflow methods."""

    @patch("molecular_simulations.build.build_amber.subprocess")
    def test_build_calls_workflow_steps(self, mock_subprocess):
        """Test build() orchestrates prep_pdb, get_pdb_extent, etc."""
        from molecular_simulations.build.build_amber import ExplicitSolvent

        mock_subprocess.run.return_value = MagicMock(returncode=0)
        mock_subprocess.DEVNULL = -1

        with tempfile.TemporaryDirectory() as tmpdir:
            pdb_path = Path(tmpdir) / "test.pdb"
            pdb_path.write_text(
                "ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00\n"
                "ATOM      2  CA  ALA A   1       1.458   0.000   0.000  1.00  0.00\n"
                "END\n"
            )

            with patch.dict(os.environ, {"AMBERHOME": "/fake/amber"}):
                builder = ExplicitSolvent(path=tmpdir, pdb=str(pdb_path), protein=True)

            with (
                patch.object(builder, "prep_pdb") as mock_prep,
                patch.object(
                    builder, "get_pdb_extent", return_value=50.0
                ) as mock_extent,
                patch.object(builder, "get_ion_numbers", return_value=10) as mock_ions,
                patch.object(builder, "assemble_system") as mock_assemble,
                patch.object(builder, "clean_up_directory") as mock_clean,
            ):
                builder.build()

            mock_prep.assert_called_once()
            mock_extent.assert_called_once()
            mock_ions.assert_called_once_with(50.0**3)
            mock_assemble.assert_called_once_with(50.0, 10)
            mock_clean.assert_called_once()

    @patch("molecular_simulations.build.build_amber.subprocess")
    def test_prep_pdb_runs_pdb4amber(self, mock_subprocess):
        """Test prep_pdb calls pdb4amber subprocess."""
        from molecular_simulations.build.build_amber import ExplicitSolvent

        mock_subprocess.run.return_value = MagicMock(returncode=0)
        mock_subprocess.DEVNULL = -1

        with tempfile.TemporaryDirectory() as tmpdir:
            pdb_path = Path(tmpdir) / "test.pdb"
            pdb_path.write_text(
                "ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00\n"
            )

            with patch.dict(os.environ, {"AMBERHOME": "/fake/amber"}):
                builder = ExplicitSolvent(path=tmpdir, pdb=str(pdb_path), protein=True)

            builder.prep_pdb()

            # Check subprocess was called with pdb4amber
            mock_subprocess.run.assert_called()

            # pdb should be updated to protein.pdb
            assert builder.pdb.endswith("protein.pdb")

    @patch("molecular_simulations.build.build_amber.subprocess")
    def test_assemble_system_calls_tleap(self, mock_subprocess):
        """Test assemble_system generates tleap input."""
        from molecular_simulations.build.build_amber import ExplicitSolvent

        mock_subprocess.run.return_value = MagicMock(returncode=0)
        mock_subprocess.DEVNULL = -1

        with tempfile.TemporaryDirectory() as tmpdir:
            pdb_path = Path(tmpdir) / "test.pdb"
            pdb_path.write_text(
                "ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00\n"
            )

            with patch.dict(os.environ, {"AMBERHOME": "/fake/amber"}):
                builder = ExplicitSolvent(path=tmpdir, pdb=str(pdb_path), protein=True)

            with patch.object(builder, "temp_tleap") as mock_tleap:
                builder.assemble_system(dim=60.0, num_ions=15)

            mock_tleap.assert_called_once()
            tleap_input = mock_tleap.call_args[0][0]
            assert "solvatebox" in tleap_input
            assert "addIonsRand" in tleap_input
            assert "15" in tleap_input
            assert "60.0" in tleap_input

    @patch("molecular_simulations.build.build_amber.subprocess")
    def test_assemble_system_debug_mode(self, mock_subprocess):
        """Test assemble_system uses debug_tleap in debug mode."""
        from molecular_simulations.build.build_amber import ExplicitSolvent

        mock_subprocess.run.return_value = MagicMock(returncode=0)
        mock_subprocess.DEVNULL = -1

        with tempfile.TemporaryDirectory() as tmpdir:
            pdb_path = Path(tmpdir) / "test.pdb"
            pdb_path.write_text(
                "ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00\n"
            )

            with patch.dict(os.environ, {"AMBERHOME": "/fake/amber"}):
                builder = ExplicitSolvent(
                    path=tmpdir, pdb=str(pdb_path), protein=True, debug=True
                )

            with patch.object(builder, "debug_tleap") as mock_debug:
                builder.assemble_system(dim=60.0, num_ions=15)

            mock_debug.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
