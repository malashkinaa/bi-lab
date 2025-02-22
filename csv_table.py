import pandas as pd
import os

class CsvTable:
    """Class for handling CSV data operations: loading, processing, and validating."""

    def __init__(self, url:str, table_name:str=None):
        """Initializes the CsvTable with a URL and an optional table name."""
        self.url = url
        self.table_name = table_name if table_name else os.path.splitext(os.path.basename(url))[0]
        self.df = None

    def load_data(self):
        """Loads the data from the specified URL or local path."""
        try:
            self.df = pd.read_csv(self.url, sep=';', low_memory=False)
            print(f"Loaded data for table '{self.table_name}'")
            print(f"Columns: {self.df.columns.tolist()}")
            print(f"Rows: {len(self.df)}\n")
        except Exception as e:
            print(f"Error loading data for table '{self.table_name}': {e}")
            self.df = pd.DataFrame()

    # def check_columns(self, required_columns):
    #     """Checks for the presence of required columns in the dataframe."""
    #     if self.df is not None:
    #         missing = [col for col in required_columns if col not in self.df.columns]
    #         if missing:
    #             print(f"Missing columns in the table '{self.table_name}': {missing}")
    #         else:
    #             print(f"All required columns are present in the table '{self.table_name}'.")
    #     else:
    #         print("Dataframe is empty. Please load the data first.")

    # def preprocess_date(self, date_column, new_column_name):
    #     """Converts YYMMDD format to a datetime object and creates a new column."""
    #     if self.df is not None:
    #         self.df[date_column] = self.df[date_column].astype(str).str.zfill(6)
    #         self.df[new_column_name] = pd.to_datetime(
    #             self.df[date_column], format='%y%m%d', errors='coerce'
    #         )
    #     else:
    #         print("Dataframe is empty. Please load the data first.")