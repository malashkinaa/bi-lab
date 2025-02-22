from dash import Dash, dcc, html, Input, Output, dash_table
import dash_cytoscape as cyto
import pandas as pd

class HTMLPage:
    """Class for generating HTML content for the financial dashboard."""

    def __init__(self, bi_dataset):
        """Initializes the HTMLPage with a BI dataset."""
        self.tables = bi_dataset.tables

        """Dash Cytoscape elements for the star schema visualization."""
        self.nodes = [
            # Fact Table
            {'data': {'id': 'fact_trans', 'label': 'fact_trans'}, 'classes': 'fact'},
            {'data': {'id': 'fact_loan', 'label': 'fact_loan'}, 'classes': 'fact'},

            # Dimension Tables
            {'data': {'id': 'dim_date', 'label': 'dim_date'}, 'classes': 'dimension'},
            {'data': {'id': 'dim_client', 'label': 'dim_client'}, 'classes': 'dimension'},
            {'data': {'id': 'dim_district', 'label': 'dim_district'}, 'classes': 'dimension'},
            {'data': {'id': 'dim_account', 'label': 'dim_account'}, 'classes': 'dimension'},
            {'data': {'id': 'dim_card', 'label': 'dim_card'}, 'classes': 'dimension'}
        ]

        self.edges = [
            {'data': {'source': 'fact_trans', 'target': 'dim_date'}},
            {'data': {'source': 'fact_trans', 'target': 'dim_district'}},
            {'data': {'source': 'fact_trans', 'target': 'dim_client'}},
            {'data': {'source': 'fact_trans', 'target': 'dim_account'}},
            {'data': {'source': 'fact_trans', 'target': 'dim_card'}},
            {'data': {'source': 'fact_loan', 'target': 'dim_date'}},
            {'data': {'source': 'fact_loan', 'target': 'dim_district'}},
            {'data': {'source': 'fact_loan', 'target': 'dim_client'}},
            {'data': {'source': 'fact_loan', 'target': 'dim_account'}},
            {'data': {'source': 'fact_loan', 'target': 'dim_card'}},            
        ]
        self.elements = self.nodes + self.edges

        self.stylesheet = [
            {
                'selector': 'node',
                'style': {
                    'label': 'data(label)',
                    'text-valign': 'center',
                    'color': 'white',
                    'text-outline-width': 2,
                    'text-outline-color': '#888',
                    'font-size': '12px'
                }
            },
            {
                'selector': 'node.fact',
                'style': {
                    'background-color': '#FF4136',
                    'shape': 'ellipse',
                    'width': '60px',
                    'height': '60px'
                }
            },
            {
                'selector': 'node.dimension',
                'style': {
                    'background-color': '#0074D9',
                    'shape': 'roundrectangle',
                    'width': '50px',
                    'height': '50px'
                }
            },
            {
                'selector': 'edge',
                'style': {
                    'width': 2,
                    'line-color': '#ccc',
                    'target-arrow-color': '#ccc',
                    'target-arrow-shape': 'triangle',
                    'curve-style': 'bezier'
                }
            }
        ]

    def generate_page(self):
        """Generates the HTML content for the financial dashboard."""
        app = Dash(__name__)
        app.title = "1999 Czech Bank Financial"

        app.layout = html.Div([
            html.H1("1999 Czech Bank Financial", style={'textAlign': 'center'}),

            html.H2("Star Schema Structure", style={'textAlign': 'center'}),
            html.Div([
                cyto.Cytoscape(
                    id='star-schema',
                    elements=self.elements,
                    stylesheet=self.stylesheet,
                    layout={'name': 'cose'},
                    # Можна змінити на інший макет, наприклад, 'breadthfirst', 'cose', 'grid', 'circle', etc.
                    style={'width': '100%', 'height': '400px'}
                )
            ]),
            html.Hr(),

            html.H2("Pivot transactions by year and region", style={'textAlign': 'center'}),
            dash_table.DataTable(
                id='pivot-table-region',
                columns=[
                    {"name": str(i) if not pd.isna(i) else "Undefined", 
                     "id": str(i) if not pd.isna(i) else "Undefined"}
                    for i in self.tables['pivot_by_year_district'].reset_index().columns
                ],
                data=self.tables['pivot_by_year_district'].reset_index().to_dict('records'),
                style_table={'overflowX': 'auto'},
                style_cell={
                    'minWidth': '100px', 'width': '150px', 'maxWidth': '180px',
                    'whiteSpace': 'normal',
                    'textAlign': 'center'
                },
                page_size=20
            ),

            html.H2("Pivot transactions by year and card", style={'textAlign': 'center'}),
            dash_table.DataTable(
                id='pivot-table-card',
                columns=[
                    {"name": str(i) if not pd.isna(i) else "Undefined", 
                     "id": str(i) if not pd.isna(i) else "Undefined"}
                    for i in self.tables['pivot_by_year_card'].reset_index().columns
                ],
                data=self.tables['pivot_by_year_card'].reset_index().to_dict('records'),
                style_table={'overflowX': 'auto'},
                style_cell={
                    'minWidth': '100px', 'width': '150px', 'maxWidth': '180px',
                    'whiteSpace': 'normal',
                    'textAlign': 'center'
                },
                page_size=20
            ),

            html.H2("Pivot loan amount by year and district", style={'textAlign': 'center'}),
            dash_table.DataTable(
                id='pivot-loan-region',
                columns=[
                    {"name": str(i) if not pd.isna(i) else "Undefined", 
                     "id": str(i) if not pd.isna(i) else "Undefined"}
                    for i in self.tables['pivot_loan_by_year_district'].reset_index().columns
                ],
                data=self.tables['pivot_loan_by_year_district'].reset_index().to_dict('records'),
                style_table={'overflowX': 'auto'},
                style_cell={
                    'minWidth': '100px', 'width': '150px', 'maxWidth': '180px',
                    'whiteSpace': 'normal',
                    'textAlign': 'center'
                },
                page_size=20
            ),

            html.H2("Pivot loan duration avarage by year and district", style={'textAlign': 'center'}),
            dash_table.DataTable(
                id='pivot-loan-avg-region',
                columns=[
                    {"name": str(i) if not pd.isna(i) else "Undefined", 
                     "id": str(i) if not pd.isna(i) else "Undefined"}
                    for i in self.tables['pivot_loan_avarage_by_year_district'].reset_index().columns
                ],
                data=self.tables['pivot_loan_avarage_by_year_district'].reset_index().to_dict('records'),
                style_table={'overflowX': 'auto'},
                style_cell={
                    'minWidth': '100px', 'width': '150px', 'maxWidth': '180px',
                    'whiteSpace': 'normal',
                    'textAlign': 'center'
                },
                page_size=20
            ),            
        ])


        app.run_server(debug=True, port=8888)



        
            