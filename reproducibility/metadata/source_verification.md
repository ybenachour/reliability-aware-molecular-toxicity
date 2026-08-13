# Dataset source verification record

Verification date: **2026-07-23**

This file records access paths and unresolved licensing/provenance issues. It is not a substitute for legal review.

## hERG blockade

- ML-ready distribution: Therapeutics Data Commons, toxicity task `hERG_Karim`.
- Access: `pytdc.single_pred.Tox(name="hERG_Karim")` with `pytdc==1.1.15`.
- TDC endpoint statement: blocker when reported IC50 is below 10 µM; non-blocker at or above 10 µM.
- Primary cited source: Karim et al., *CardioTox net*, Journal of Cheminformatics 13, 60 (2021), DOI `10.1186/s13321-021-00541-z`.
- Qualification: integrated literature/database labels are not a uniform assay protocol. The exact threshold is recorded, but source-level protocols and units should be enriched before claims about assay-specific performance.
- Terms: the TDC web page shows ambiguous wording ("Not Specified" together with a CC BY 4.0 link). The project does not redistribute the table and requires original-source terms to be checked before redistribution.

## Ames mutagenicity

- ML-ready distribution: Therapeutics Data Commons toxicity task `AMES`.
- Access: `pytdc.single_pred.Tox(name="AMES")`.
- Label: binary mutagenic/non-mutagenic aggregate.
- Primary cited source: Xu et al., *In silico prediction of chemical Ames mutagenicity*, Journal of Chemical Information and Modeling 52 (2012) 2840–2847, DOI `10.1021/ci300400a`.
- Qualification: the distributed TDC table does not expose bacterial strain or metabolic-activation fields. The project must not claim strain-specific performance unless the source data are enriched with traceable assay metadata.
- Terms: same ambiguity as the hERG TDC distribution; no raw data are redistributed.

## Tox21 stress-response endpoints

- Primary scientific access page: NCATS Tox21 Data Challenge 2014.
- Download used by the reproducible workflow: the documented MoleculeNet/DeepChem `tox21.csv.gz` mirror.
- Encoding: `1` active, `0` inactive, blank unavailable/missing. Blanks remain missing and are masked.
- Endpoint references:
  - `SR-p53`: PubChem summary AID 720552; p53 stress-response reporter activity.
  - `SR-ATAD5`: PubChem summary AID 720516; genotoxicity-associated ATAD5 reporter activity in human cells.
  - `SR-ARE`: PubChem summary AID 743219; antioxidant response element reporter activity.
  - `SR-MMP`: PubChem summary AID 720637 (confirmatory project record AID 720635); disruption of mitochondrial membrane potential.
- Qualification: these are in-vitro assay-activity labels, not clinical toxicity outcomes. NCATS notes that assay artifacts and nonspecific signal effects are possible.
- Terms: no dataset-specific license statement was identified on the challenge page during verification. Public access is documented, but redistribution rights are not inferred.

## Independence rule

A dataset hosted elsewhere is not automatically external. Exact standardized-SMILES/InChIKey overlap, scaffold overlap, source/publication derivation, endpoint compatibility, assay thresholds, and collection independence must be audited before any external-validation claim.
