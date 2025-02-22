import os
import pandas as pd
from csv_table import CsvTable

class CvsDataset:
    """Class to manage dataset of CSV files as CsvTable instances."""

    def __init__(self, folder):
        self.folder = folder
        self.tables = {}

        # Ensure the target directory exists
        if not os.path.exists(folder):
            os.makedirs(folder)

    def load_csv_tables(self):
        """Loads CSV files from the local path into CsvTable instances."""
        for filename in os.listdir(self.folder):
            if filename.endswith('.csv'):
                path = os.path.join(self.folder, filename)
                table_name = os.path.splitext(filename)[0]
                csv_table = CsvTable(path, table_name)
                csv_table.load_data()
                self.tables[table_name] = csv_table

    def get_table(self, table_name):
        """Returns the CsvTable instance for the specified table name."""
        return self.tables.get(table_name, None)

    def list_tables(self):
        """Lists all tables available in the dataset."""
        return list(self.tables.keys())