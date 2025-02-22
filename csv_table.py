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
