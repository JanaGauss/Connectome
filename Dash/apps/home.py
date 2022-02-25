from dash import dcc
from dash import html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State

from app import app

layout = html.Div([
    dbc.Container([
        dbc.Row([
            dbc.Col(html.H1("Welcome to the Connectome dashboard", className="text-center"),
                    className="mb-5 mt-5")
        ]),
        dbc.Row([
            dbc.Col(html.H5(children='This app blablabla ',
                            className='')
                    , className="mb-2")
        ]),

        dbc.Row([
            dbc.Col(html.P(
                children='It consists of x pages: blablablabl')
                , className="mb-3")
        ]),


        html.A("Link to our github repo",
               href="https://www.youtube.com/watch?v=dQw4w9WgXcQ&ab_channel=RickAstley")

    ])
])
