import importlib.util
import pathlib
import sys
import tempfile
import unittest


MODULE_PATH = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "05_evaluate.py"

spec = importlib.util.spec_from_file_location("evaluate_script", MODULE_PATH)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)


class EvaluateJoinTests(unittest.TestCase):
    def test_join_residues_raises_on_identity_mismatch(self):
        predicted = [
            {"number": 1, "name": "ALA"},
            {"number": 2, "name": "SER"},
        ]
        reference = [
            {"number": 1, "name": "TRP"},
            {"number": 2, "name": "SER"},
        ]
        mapping = {1: 1, 2: 2}

        with self.assertRaises(ValueError):
            module.join_residues(predicted, reference, mapping)

    def test_join_residues_allows_matching_residues(self):
        predicted = [
            {"number": 1, "name": "ALA"},
            {"number": 2, "name": "SER"},
        ]
        reference = [
            {"number": 1, "name": "ALA"},
            {"number": 2, "name": "SER"},
        ]
        mapping = {1: 1, 2: 2}

        rows = module.join_residues(predicted, reference, mapping)
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["predicted_residue_name"], "ALA")
        self.assertEqual(rows[0]["reference_residue_name"], "ALA")

    def test_extract_residues_from_mmcif(self):
        content = '''
data_test

loop_
_atom_site.group_PDB
_atom_site.id
_atom_site.type_symbol
_atom_site.label_atom_id
_atom_site.label_comp_id
_atom_site.label_seq_id
_atom_site.label_asym_id
_atom_site.auth_asym_id
_atom_site.Cartn_x
_atom_site.Cartn_y
_atom_site.Cartn_z
ATOM 1 C CA ALA 1 A A 0.0 0.0 0.0
ATOM 2 C CB ALA 1 A A 1.0 0.0 0.0
ATOM 3 C CA ALA 3 A A 2.0 0.0 0.0
'''
        with tempfile.NamedTemporaryFile("w", suffix=".cif", delete=False) as handle:
            handle.write(content)
            temp_path = pathlib.Path(handle.name)
        try:
            residues = module.extract_residues_from_cif(temp_path)
        finally:
            temp_path.unlink(missing_ok=True)

        self.assertEqual([res["number"] for res in residues], [1, 3])
        self.assertTrue(all(res["name"] == "UNK" for res in residues))


if __name__ == "__main__":
    unittest.main()
