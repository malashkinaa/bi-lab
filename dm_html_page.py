import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, dcc, html, dash_table
import dash_bootstrap_components as dbc
import base64
from io import BytesIO

class HTMLPage:
    """Class for generating HTML content for the financial dashboard."""

    def __init__(self, datamining):
        """Initializes the HTMLPage with datamining."""
        self.datamining = datamining

    def generate_page(self):
        app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
        app.title = "Data Mining Dashboard"
                
        app.layout = dbc.Container([
            dbc.Row([
                dbc.Col(html.H1("Data Mining Dashboard"), className="text-center mb-4")
            ]),

            # Task 1: Classification of Loan Statuses
            dbc.Card([
                dbc.CardHeader("Task 1: Classification of Loan Statuses"),
                dbc.CardBody([
                    dash_table.DataTable(
                        columns=[{"name": i, "id": i} for i in self.datamining.df_classification.columns],
                        data=self.datamining.df_classification.to_dict('records'),
                        style_cell={'textAlign': 'center'},
                        style_header={'backgroundColor': '#f1f1f1', 'fontWeight': 'bold'},
                        page_size=5
                    )
                ])
            ], className="mb-4"),

            # Task 2: Clustering of customers
            dbc.Card([
                dbc.CardHeader("Task 2: Clustering of customers"),
                dbc.CardBody([
                    dash_table.DataTable(
                        columns=[{"name": i, "id": i} for i in self.datamining.df_clustering.columns],
                        data=self.datamining.df_clustering.to_dict('records'),
                        style_cell={'textAlign': 'center'},
                        style_header={'backgroundColor': '#f1f1f1', 'fontWeight': 'bold'},
                        page_size=5
                    )
                ])
            ], className="mb-4"),

            # Task 3: Prediction of Transaction Amount
            dbc.Card([
                dbc.CardHeader("Task 3: Prediction of Transaction Amount"),
                dbc.CardBody([
                    dash_table.DataTable(
                        columns=[{"name": i, "id": i} for i in self.datamining.df_forecasting.columns],
                        data=self.datamining.df_forecasting.to_dict('records'),
                        style_cell={'textAlign': 'center'},
                        style_header={'backgroundColor': '#f1f1f1', 'fontWeight': 'bold'},
                        page_size=5
                    )
                ])
            ], className="mb-4"),


            # Task 4: Dependency Between Transaction Volume and Credit Reliability
            dbc.Card([
                dbc.CardHeader("Task 4: Dependency Between Transaction Volume and Credit Reliability"),
                dbc.CardBody([
                    dash_table.DataTable(
                        columns=[{"name": i, "id": i} for i in self.datamining.df_dependencies.columns],
                        data=self.datamining.df_dependencies.to_dict('records'),
                        style_cell={'textAlign': 'center'},
                        style_header={'backgroundColor': '#f1f1f1', 'fontWeight': 'bold'},
                        page_size=5
                    )
                ]),
                # confusion_matrix_image(self.datamining.cm_dependencies, "Dependencies Confusion Matrix"),
            ], className="mb-4"),

        ], fluid=True)

        app.run(debug=True, port=8888)


