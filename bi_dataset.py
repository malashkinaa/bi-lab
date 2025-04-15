import pandas as pd

class BiDataset:
    """Class to handle BI operations and generate BI tables."""

    def __init__(self, csvDataset):
        """
        Initializes the BiDataset with a csvDataset instance.
        
        Args:
            csvDataset (CvsDataset): An instance of CvsDataset containing CSV data.
        """
        self.csvDataset = csvDataset
        self.tables = {}

    def generate_bi_tables(self):
        """Generates dimension tables"""
        self._create_date_table()
        self._create_district_table()
        self._create_client_table()
        self._create_card_table()
        self._create_account_table()
        """Generates various fact tables""" # 2 measures (trans.amount, loan.amount)
        self._create_trans_loan_table()
        """Generates various pivot tables"""
        self._create_pivot_trans_by_year_district()
        self._create_pivot_trans_by_year_card()
        self._create_pivot_loan_by_year_district()
        self._create_pivot_loan_avarage_by_year_district()
    
    def _create_district_table(self):
        self.tables['dim_district'] = self.csvDataset.tables['district'].df[['A1', 'A2', 'A3']].rename(columns={
            'A1': 'district_id',
            'A2': 'region',
            'A3': 'district_name'
        })

    def _create_client_table(self):
        self.tables['dim_client'] = self.csvDataset.tables['client'].df[['client_id', 'birth_number']]

    def _create_card_table(self):
        self.tables['dim_card'] = self.csvDataset.tables['card'].df

    def _create_account_table(self):
        self.tables['dim_account'] = self.csvDataset.tables['account'].df

    def _create_trans_loan_table(self):
        trans = self.csvDataset.tables["trans"].df
        account = self.csvDataset.tables["account"].df
        card = self.csvDataset.tables["card"].df
        # get unique disposition to keep the trans summable or aggregatable
        disp = self.csvDataset.tables["disp"].df.drop_duplicates(subset=['account_id'], keep='first')

        # flatten the dimension hierarchy by denormalizing it into the transaction fact table
        # attach district_id
        fact_trans = pd.merge(trans, account[["account_id", "district_id"]], on="account_id")
        # attach client_id
        fact_trans = pd.merge(fact_trans, disp[["account_id", "client_id", "disp_id"]], on="account_id")
        # attach card
        fact_trans = pd.merge(fact_trans, card[["card_id", "disp_id"]], on="disp_id", how="left")

        self.tables['fact_trans'] = fact_trans
        #################
        fact_loan = self.csvDataset.tables['loan'].df
        # flatten the dimension hierarchy by denormalizing it into the loan fact table
        # attach district_id
        fact_loan = pd.merge(fact_loan, account[["account_id", "district_id"]], on="account_id")
        # attach client_id
        fact_loan = pd.merge(fact_loan, disp[["account_id", "client_id", "disp_id"]], on="account_id")
        # attach card
        fact_loan = pd.merge(fact_loan, card[["card_id", "disp_id"]], on="disp_id", how="left")              
        self.tables['fact_loan'] = fact_loan
    
    def _create_pivot_trans_by_year_district(self):
        pivot_facts = self.tables['fact_trans'][['date', 'amount', 'district_id']].copy()
        dim_date = self.tables['dim_date'][['date', 'date_full', 'year']]
        dim_district = self.tables['dim_district'][['district_id', 'district_name', 'region']]
        pivot_facts = pd.merge(pivot_facts, dim_date, on='date')
        pivot_facts = pd.merge(pivot_facts, dim_district, on='district_id')

        pivot_table = pivot_facts.pivot_table(
            values='amount',
            index='region',
            columns='year',
            aggfunc='sum',
            fill_value=0
        )
        self.tables['pivot_by_year_district'] = pivot_table

    def _create_pivot_trans_by_year_card(self):
        pivot_facts = self.tables['fact_trans'][['date', 'amount', 'district_id', 'disp_id']].copy()
        dim_date = self.tables['dim_date'][['date', 'date_full', 'year']]
        dim_card = self.tables['dim_card'][['card_id', 'type', 'disp_id']].rename(columns={'type': 'card_type'})
        pivot_facts = pd.merge(pivot_facts, dim_date, on='date')
        pivot_facts = pd.merge(pivot_facts, dim_card, on='disp_id')

        pivot_table = pivot_facts.pivot_table(
            values='amount',
            index='card_type',
            columns='year',
            aggfunc='sum',
            fill_value=0
        )
        self.tables['pivot_by_year_card'] = pivot_table

    def _create_pivot_loan_by_year_district(self):
        pivot_facts = self.tables['fact_loan'].copy()
        dim_date = self.tables['dim_date'][['date', 'date_full', 'year']]
        dim_district = self.tables['dim_district'][['district_id', 'region', 'district_name']]
        pivot_facts = pd.merge(pivot_facts, dim_date, on='date')
        pivot_facts = pd.merge(pivot_facts, dim_district, on='district_id')

        pivot_table = pivot_facts.pivot_table(
            values='amount',
            index='region',
            columns='year',
            aggfunc='sum',
            fill_value=0
        )
        self.tables['pivot_loan_by_year_district'] = pivot_table

    def _create_pivot_loan_avarage_by_year_district(self):
        pivot_facts = self.tables['fact_loan'].copy()
        dim_date = self.tables['dim_date'][['date', 'date_full', 'year']]
        dim_district = self.tables['dim_district'][['district_id', 'region', 'district_name']]
        pivot_facts = pd.merge(pivot_facts, dim_date, on='date')
        pivot_facts = pd.merge(pivot_facts, dim_district, on='district_id')

        pivot_table = pivot_facts.pivot_table(
            values='duration',
            index='district_name',
            columns='year',
            aggfunc='mean',
            fill_value=0
        )
        pivot_table = pivot_table.round(2)
        self.tables['pivot_loan_avarage_by_year_district'] = pivot_table

    def _create_date_table(self):
        """Generates a dim_date table based on date columns in the format 'yymmdd'."""
        all_dates = []

        # Collect all dates
        for csv_table in self.csvDataset.tables.values():
            if 'date' in csv_table.df.columns:
                dates = pd.to_datetime(csv_table.df['date'], format='%y%m%d', errors='coerce')
                all_dates.extend(dates.dropna().tolist())

        if all_dates:
            # Determine min and max date
            min_date = min(all_dates)
            max_date = max(all_dates)

            # Create a complete range of dates
            date_range = pd.date_range(start=min_date, end=max_date)

            # Format date_range to 'yymmdd'
            formatted_date_ids = date_range.strftime('%y%m%d').astype('int64')

            # Construct the dim_date DataFrame
            dim_date_df = pd.DataFrame({
                'date': formatted_date_ids,
                'date_full': date_range,
                'year': date_range.year,
                'quarter': date_range.quarter,
                'month': date_range.month,
                'day': date_range.day
            })

            self.tables['dim_date'] = dim_date_df

        else:
            print("No valid date entries found in any table to create a dim_date table.")

    def list_tables(self):
        """Lists all tables available in the dataset."""
        return list(self.tables.keys())