from csv_dataset import CvsDataset
from bi_dataset import BiDataset
from dm_datamining import Datamining
from dm_html_page import HTMLPage

def main():
    local_path = "./1999-czech-financial-dataset"

    # Create an instance of dataset
    source_ds = CvsDataset(local_path)

    # Download and load the dataset
    source_ds.load_csv_tables() # E. (ETL)
    
    # Create an instance of the BI dataset and inject the source dataset to generate BI tables
    dest_ds = BiDataset(source_ds)
    dest_ds.generate_bi_tables() # T. (ETL)

    # List all available tables
    table_names = dest_ds.list_tables()
    print(f"Available tables: {table_names}")

    for name, table in dest_ds.tables.items():
        print(f"\nTable {name} rows: {len(table)}") 
        print("Columns:", list(table.columns))

    datamining = Datamining(dest_ds)
    datamining.processing()

    # Create an instance of the HTML page and inject the datamining
    html_page = HTMLPage(datamining)
    html_page.generate_page()

if __name__ == "__main__":
    main()