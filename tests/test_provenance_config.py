from pathlib import Path

from toxicity_screening.config import load_configs


def test_endpoint_and_source_configuration_complete():
    root = Path(__file__).resolve().parents[1]
    configs = load_configs(root)
    assert set(configs["endpoints"]["endpoints"]) == {
        "herg_blockade", "ames_mutagenicity", "SR-p53", "SR-ATAD5", "SR-ARE", "SR-MMP"
    }
    assert configs["data_config"]["sources"]["tox21"]["label_columns"] == [
        "SR-p53", "SR-ATAD5", "SR-ARE", "SR-MMP"
    ]
