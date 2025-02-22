from csv_dataset import CvsDataset
from bi_dataset import BiDataset
from html_page import HTMLPage

def main():
    local_path = "./1999-czech-financial-dataset"

    # Create an instance of dataset
    source_ds = CvsDataset(local_path)

    # Download and load the dataset
    source_ds.load_csv_tables() # E. (ETL)

    # List all available tables
    table_names = source_ds.list_tables()
    print(f"Available tables: {table_names}")
    
    # Create an instance of the BI dataset and inject the source dataset to generate BI tables
    dest_ds = BiDataset(source_ds)
    dest_ds.generate_bi_tables() # T. (ETL)

    for name, table in dest_ds.tables.items():
        print(f"\nRows from  {name}: {len(table)}") 
        print(table.head())  # Display the first few rows

    # dest_ds.create_pivot_table()

    # Create an instance of the HTML page and inject the BI dataset
    html_page = HTMLPage(dest_ds) #L. (in the memory)
    html_page.generate_page()

if __name__ == "__main__":
    main()