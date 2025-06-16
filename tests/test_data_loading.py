import pytest
import pandas as pd
from src.core.data_loader import DataLoader, DataValidator
from src.config.paths import RAW_DATA_DIR
import os
import shutil

# Fixture to create temporary test files
@pytest.fixture
def temp_data_files(tmp_path):
    csv_file = tmp_path / "test.csv"
    csv_file.write_text("col1,col2\n1,A\n2,B")

    txt_file = tmp_path / "test.txt"
    txt_file.write_text("colA\tcolB\nval1\tvalX\nval2\tvalY")

    json_file = tmp_path / "test.json"
    json_file.write_text('[{"col1": 1, "col2": "A"}, {"col1": 2, "col2": "B"}]')

    excel_file = tmp_path / "test.xlsx"
    # Create a dummy Excel file
    pd.DataFrame({"col1": [1, 2], "col2": ["A", "B"]}).to_excel(excel_file, index=False)

    empty_csv = tmp_path / "empty.csv"
    empty_csv.write_text("")

    return {
        "csv": str(csv_file),
        "txt": str(txt_file),
        "json": str(json_file),
        "xlsx": str(excel_file),
        "empty_csv": str(empty_csv),
        "non_existent": str(tmp_path / "non_existent.csv")
    }

def test_data_loader_load_csv(temp_data_files):
    loader = DataLoader()
    df = loader.load_data(temp_data_files["csv"])
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert df.shape == (2, 2)
    assert list(df.columns) == ["col1", "col2"]

def test_data_loader_load_txt(temp_data_files):
    loader = DataLoader()
    df = loader.load_data(temp_data_files["txt"])
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert df.shape == (2, 2)
    assert list(df.columns) == ["colA", "colB"]

def test_data_loader_load_json(temp_data_files):
    loader = DataLoader()
    df = loader.load_data(temp_data_files["json"])
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert df.shape == (2, 2)
    assert list(df.columns) == ["col1", "col2"]

def test_data_loader_load_excel(temp_data_files):
    loader = DataLoader()
    df = loader.load_data(temp_data_files["xlsx"])
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert df.shape == (2, 2)
    assert list(df.columns) == ["col1", "col2"]

def test_data_loader_file_not_found(temp_data_files):
    loader = DataLoader()
    with pytest.raises(FileNotFoundError):
        loader.load_data(temp_data_files["non_existent"])

def test_data_loader_empty_file(temp_data_files):
    loader = DataLoader()
    with pytest.raises(ValueError, match="contains no data"):
        loader.load_data(temp_data_files["empty_csv"])

def test_data_loader_unsupported_format(tmp_path):
    loader = DataLoader()
    unsupported_file = tmp_path / "test.xyz"
    unsupported_file.write_text("some data")
    with pytest.raises(ValueError, match="Unsupported file format"):
        loader.load_data(str(unsupported_file))

def test_data_loader_save_raw_data(tmp_path, sample_dataframe):
    loader = DataLoader()
    test_file_name = "test_raw_save.csv"
    save_path = os.path.join(RAW_DATA_DIR, test_file_name)
    
    # Ensure RAW_DATA_DIR exists and is clean for this test
    if os.path.exists(RAW_DATA_DIR):
        shutil.rmtree(RAW_DATA_DIR)
    os.makedirs(RAW_DATA_DIR)

    loader.save_raw_data(sample_dataframe, test_file_name)
    assert os.path.exists(save_path)
    loaded_df = pd.read_csv(save_path)
    pd.testing.assert_frame_equal(loaded_df, sample_dataframe)
    
    # Clean up
    os.remove(save_path)

def test_data_validator_valid_df(sample_dataframe):
    validator = DataValidator()
    assert validator.validate_dataframe(sample_dataframe) == True

def test_data_validator_empty_df():
    validator = DataValidator()
    empty_df = pd.DataFrame()
    assert validator.validate_dataframe(empty_df) == False

def test_data_validator_too_few_rows_df():
    validator = DataValidator()
    small_df = pd.DataFrame({'col1': [1, 2]}) # Less than 5 rows
    assert validator.validate_dataframe(small_df) == False