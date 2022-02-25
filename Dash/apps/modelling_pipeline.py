import pandas as pd
from dash import dcc
from dash import html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate

import plotly.graph_objects as go




tab1_content = dbc.Card(
    dbc.CardBody(
        [
            html.P("Fill in preprocessing ...", className="card-text"),
            dbc.Button("Click here", color="success"),
        ]
    ),
    className="mt-3",
)

tab2_content = dbc.Card(
    dbc.CardBody(
        [
            html.P("Fill in Modelling ... ", className="card-text"),
            dbc.Button("Don't click here", color="danger"),
        ]
    ),
    className="mt-3",
)


layout = html.Div([
    dbc.Container([
        dbc.Row([
            dbc.Col([html.H2(children='Welcome to the Modelling pipeline', className='')]
                    , className="mb-2 mt-5")
        ]),
        dbc.Row([
            dbc.Col([html.P(
                children='Please start with the preprocessing option.')]
                , className="")
        ]),
    ]),
    dbc.Container([
        dbc.Row([
            dbc.Col([
                dbc.Tabs(
                    [
                        dbc.Tab(tab1_content, label="Preprocessing"),
                        dbc.Tab(tab2_content, label="Modelling")
                    ]
                )
            ])
        ])
    ]),
    html.Div([
        dcc.Store(id="memory", data=[])
    ])
])

