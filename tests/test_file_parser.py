import pytest
import pandas as pd
from pandas.testing import assert_frame_equal, assert_series_equal
import io
from datetime import datetime
import openpyxl # Required to create in-memory excel data for testing

# Adjust import path if your project structure is different
# This assumes 'services' is a package discoverable in the PYTHONPATH
# If running tests from the root directory, this should work if 'services' is a top-level folder.
from services.file_parser import parse_file

@pytest.fixture
def valid_csv_data():
    return "2023-01-01,100\n2023-01-02,150\n2023-01-03,200"

@pytest.fixture
def valid_excel_data():
    # Create an in-memory Excel file
    output = io.BytesIO()
    workbook = openpyxl.Workbook()
    sheet = workbook.active
    sheet.append(["2023-01-01", 100])
    sheet.append(["2023-01-02", 150])
    sheet.append(["2023-01-03", 200])
    workbook.save(output)
    output.seek(0) # Rewind the buffer to the beginning for reading
    return output

@pytest.fixture
def expected_df():
    data = {
        'fecha': [datetime(2023, 1, 1), datetime(2023, 1, 2), datetime(2023, 1, 3)],
        'ventas': [100.0, 150.0, 200.0]
    }
    df = pd.DataFrame(data)
    df['fecha'] = pd.to_datetime(df['fecha'])
    return df

def test_parse_valid_csv(valid_csv_data, expected_df):
    """Test parsing a valid CSV string."""
    file_like_object = io.StringIO(valid_csv_data)
    df = parse_file(file_like_object)
    assert df is not None, "DataFrame should not be None for valid CSV"
    # Reset index for comparison if parse_file might produce a different index type/name
    assert_frame_equal(df.reset_index(drop=True), expected_df.reset_index(drop=True))

def test_parse_valid_excel(valid_excel_data, expected_df):
    """Test parsing a valid Excel file object."""
    # Here, valid_excel_data is already a BytesIO object
    df = parse_file(valid_excel_data)
    assert df is not None, "DataFrame should not be None for valid Excel"
    assert_frame_equal(df.reset_index(drop=True), expected_df.reset_index(drop=True))

def test_parse_unsupported_format_path():
    """Test parsing an unsupported file format by path (mocked by extension)."""
    # parse_file for paths checks extension first
    assert parse_file("data.json") is None 

def test_parse_invalid_column_structure_csv():
    """Test CSV with incorrect number of columns."""
    csv_data = "2023-01-01,100,extra\n2023-01-02,150,data"
    file_like_object = io.StringIO(csv_data)
    assert parse_file(file_like_object) is None

def test_parse_non_numeric_sales_csv():
    """Test CSV with non-numeric sales data."""
    csv_data = "2023-01-01,100\n2023-01-02,abc\n2023-01-03,200"
    file_like_object = io.StringIO(csv_data)
    df = parse_file(file_like_object)
    assert df is not None
    # Expecting rows with non-numeric sales to be dropped
    assert len(df) == 2 
    assert df['ventas'].iloc[1] == 200.0


def test_parse_non_date_fecha_csv():
    """Test CSV with non-date 'fecha' data."""
    csv_data = "2023-01-01,100\nnot-a-date,150\n2023-01-03,200"
    file_like_object = io.StringIO(csv_data)
    df = parse_file(file_like_object)
    assert df is not None
    # Expecting rows with unparseable dates to be dropped
    assert len(df) == 2
    assert df['fecha'].iloc[0] == pd.Timestamp('2023-01-01')
    assert df['fecha'].iloc[1] == pd.Timestamp('2023-01-03')


def test_parse_empty_file_csv():
    """Test parsing an empty CSV file."""
    csv_data = ""
    file_like_object = io.StringIO(csv_data)
    assert parse_file(file_like_object) is None

def test_parse_csv_header_names_not_fecha_ventas():
    """Test parsing CSV where data is fine but default read might assign other header names if not header=None."""
    # parse_file uses header=None, names=['fecha', 'ventas'], so this tests that behavior.
    # If the file itself had headers like "Date,Value", parse_file should still rename them.
    csv_data = "Date,Value\n2023-01-01,100\n2023-01-02,150" # This would be skipped by header=None
    # Correct test for this logic: just use valid_csv_data and ensure it's parsed as 'fecha', 'ventas'
    file_like_object = io.StringIO("2023-01-01,100\n2023-01-02,150")
    df = parse_file(file_like_object)
    assert df is not None
    assert 'fecha' in df.columns and 'ventas' in df.columns

def test_file_with_only_invalid_dates():
    """Test a file where all date entries are invalid."""
    csv_data = "invalid_date_1,100\ninvalid_date_2,200"
    file_like_object = io.StringIO(csv_data)
    df = parse_file(file_like_object)
    assert df is None # Expecting None as no valid rows would remain

def test_file_with_only_invalid_sales():
    """Test a file where all sales entries are non-numeric."""
    csv_data = "2023-01-01,sales_a\n2023-01-02,sales_b"
    file_like_object = io.StringIO(csv_data)
    df = parse_file(file_like_object)
    assert df is None # Expecting None as no valid rows would remain after dropping NaNs

def test_mixed_valid_invalid_data():
    """Test file with a mix of valid and invalid rows to ensure valid ones are kept."""
    csv_data = (
        "2023-01-01,100\n"
        "not-a-date,150\n"
        "2023-01-03,abc\n"  # Valid date, invalid sales
        "2023-01-04,200\n"
        "  ,300\n" # Invalid date (empty)
    )
    file_like_object = io.StringIO(csv_data)
    df = parse_file(file_like_object)
    assert df is not None
    assert len(df) == 2
    expected_dates = [pd.Timestamp('2023-01-01'), pd.Timestamp('2023-01-04')]
    expected_sales = [100.0, 200.0]
    assert list(df['fecha']) == expected_dates
    assert list(df['ventas']) == expected_sales

# To run these tests using pytest, navigate to the project root directory
# and run the command: `pytest`
# Ensure that the `services` package is in PYTHONPATH.
# If tests are in `tests/` and services in `services/`, running `pytest` from root should work.
# Or, you might need to adjust PYTHONPATH: `export PYTHONPATH=.` (Linux/macOS) or `set PYTHONPATH=.` (Windows)
# then run `pytest`.
# Also ensure 'openpyxl' is installed for the Excel tests (`pip install openpyxl`).
