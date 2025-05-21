import pandas as pd

def parse_file(file_path_or_object):
    """
    Parses an uploaded file (text, csv, or Excel) and extracts date and sales data.

    Args:
        file_path_or_object: Path to the file or a file-like object.

    Returns:
        A pandas DataFrame with 'fecha' (date) and 'ventas' (sales) columns,
        or None if parsing fails.
    """
    try:
        if isinstance(file_path_or_object, str):
            # It's a file path
            if file_path_or_object.endswith(('.xls', '.xlsx')):
                df = pd.read_excel(file_path_or_object, header=None, names=['fecha', 'ventas'])
            elif file_path_or_object.endswith(('.txt', '.csv')):
                df = pd.read_csv(file_path_or_object, header=None, names=['fecha', 'ventas'])
            else:
                raise ValueError("Unsupported file format. Please upload .txt, .csv, .xls, or .xlsx files.")
        else:
            # It's a file-like object
            # Attempt to determine type by sniffing, or try reading as excel then csv.
            # For simplicity, we'll assume if it's not excel, it's csv-like.
            # A more robust solution might involve checking file signatures or content.
            try:
                # Try reading as Excel first, as it's more specific
                df = pd.read_excel(file_path_or_object, header=None, names=['fecha', 'ventas'])
            except Exception: # Broad exception for non-excel files passed as objects
                # Rewind the file pointer if it has been read by read_excel
                if hasattr(file_path_or_object, 'seek'):
                    file_path_or_object.seek(0)
                df = pd.read_csv(file_path_or_object, header=None, names=['fecha', 'ventas'])


        # Validate columns: must have exactly two columns, named 'fecha' and 'ventas' by the read operation
        if len(df.columns) != 2 or 'fecha' not in df.columns or 'ventas' not in df.columns:
            # This check is based on the names assigned during read_csv/read_excel
            raise ValueError("File must contain exactly two columns. The first column should be dates ('fecha') and the second column sales figures ('ventas').")
        
        # Attempt to convert 'fecha' to datetime
        # errors='coerce' will turn unparseable dates into NaT (Not a Time)
        df['fecha'] = pd.to_datetime(df['fecha'], errors='coerce')
        
        # Drop rows where 'fecha' could not be parsed (NaT)
        df.dropna(subset=['fecha'], inplace=True)

        if df.empty:
            raise ValueError("No valid date entries found in the 'fecha' column or the file is empty after processing dates.")

        # Convert 'ventas' to numeric, coercing errors will turn non-numeric to NaN
        df['ventas'] = pd.to_numeric(df['ventas'], errors='coerce')
        # Drop rows where 'ventas' could not be converted to a number (became NaN)
        df.dropna(subset=['ventas'], inplace=True)

        if df.empty:
            raise ValueError("No valid sales entries found after processing 'ventas' column. Ensure sales are numeric.")

        return df

    except ValueError as ve:
        # Log this error for server-side diagnostics
        print(f"ValueError during parsing: {ve}") 
        return None # Return None to indicate failure due to format/content issues
    except Exception as e:
        # Log this error for server-side diagnostics
        print(f"An unexpected error occurred during parsing: {e}")
        return None # Return None for other unexpected errors
