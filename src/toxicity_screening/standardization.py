from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any

from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.Scaffolds import MurckoScaffold


@dataclass
class StandardizationResult:
    original_smiles: str | None
    standardized_smiles: str | None
    inchi: str | None
    inchikey: str | None
    scaffold: str | None
    standardization_status: str
    failure_reason: str | None
    transformation_log: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _fragment_key(mol: Chem.Mol) -> tuple[int, float, str]:
    return (
        mol.GetNumHeavyAtoms(),
        float(Descriptors.MolWt(mol)),
        Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True),
    )


def _principal_fragment(mol: Chem.Mol) -> Chem.Mol:
    fragments = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
    if not fragments:
        raise ValueError("No molecular fragments found")
    organic = [m for m in fragments if any(atom.GetAtomicNum() == 6 for atom in m.GetAtoms())]
    candidates = organic or list(fragments)
    return max(candidates, key=_fragment_key)


def standardize_smiles(
    smiles: str | None,
    *,
    uncharge: bool = True,
    remove_isotopes: bool = False,
    preserve_stereochemistry: bool = True,
) -> StandardizationResult:
    log: list[dict[str, Any]] = []
    original = None if smiles is None else str(smiles).strip()
    if not original:
        return StandardizationResult(original, None, None, None, None, "failed", "empty_smiles", "[]")
    try:
        mol = Chem.MolFromSmiles(original, sanitize=True)
        if mol is None:
            raise ValueError("RDKit could not parse SMILES")
        log.append({"step": "parse", "atoms": mol.GetNumAtoms(), "fragments": len(Chem.GetMolFrags(mol))})

        fragment = _principal_fragment(mol)
        if fragment.GetNumAtoms() != mol.GetNumAtoms():
            log.append({"step": "principal_fragment", "before_atoms": mol.GetNumAtoms(), "after_atoms": fragment.GetNumAtoms()})
        mol = fragment

        before = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
        mol = rdMolStandardize.Cleanup(mol)
        after = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
        if before != after:
            log.append({"step": "cleanup", "before": before, "after": after})

        normalizer = rdMolStandardize.Normalizer()
        before = after
        mol = normalizer.normalize(mol)
        after = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
        if before != after:
            log.append({"step": "normalize", "before": before, "after": after})

        reionizer = rdMolStandardize.Reionizer()
        before = after
        mol = reionizer.reionize(mol)
        after = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
        if before != after:
            log.append({"step": "reionize", "before": before, "after": after})

        if uncharge:
            before = after
            mol = rdMolStandardize.Uncharger().uncharge(mol)
            after = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
            if before != after:
                log.append({"step": "uncharge", "before": before, "after": after})

        if remove_isotopes:
            isotopes = 0
            for atom in mol.GetAtoms():
                if atom.GetIsotope():
                    atom.SetIsotope(0)
                    isotopes += 1
            if isotopes:
                log.append({"step": "remove_isotopes", "atoms_changed": isotopes})

        Chem.SanitizeMol(mol)
        standardized = Chem.MolToSmiles(
            mol, canonical=True, isomericSmiles=preserve_stereochemistry
        )
        try:
            inchi = Chem.MolToInchi(mol)
            inchikey = Chem.InchiToInchiKey(inchi)
        except Exception:
            inchi, inchikey = None, None
            log.append({"step": "inchi", "status": "unavailable"})
        scaffold_mol = MurckoScaffold.GetScaffoldForMol(mol)
        scaffold = (
            Chem.MolToSmiles(scaffold_mol, canonical=True, isomericSmiles=False)
            if scaffold_mol.GetNumAtoms() else ""
        )
        return StandardizationResult(
            original,
            standardized,
            inchi,
            inchikey,
            scaffold,
            "success",
            None,
            json.dumps(log, sort_keys=True),
        )
    except Exception as exc:
        log.append({"step": "failure", "error_type": type(exc).__name__, "message": str(exc)})
        return StandardizationResult(
            original,
            None,
            None,
            None,
            None,
            "failed",
            f"{type(exc).__name__}: {exc}",
            json.dumps(log, sort_keys=True),
        )
