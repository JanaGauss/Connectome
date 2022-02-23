from dash import dcc
from dash import html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State

from app import server
from app import app
from apps import home, modelling_pipeline, delcode_analysis

navbar = dbc.Navbar(
    dbc.Container([
        html.Div(
            dbc.Row(
                [
                    dbc.Col(html.Img(src="/assets/stonks.png", height="30px")),
                    dbc.Col(dbc.NavbarBrand("Connectome", className="ml-2")),
                ],
                align="center",
                className="g-2",
            ),
        ),
        dbc.Nav([dbc.NavLink("Delcode Analysis", href = '/delcode_analysis', active = "exact")]),
        dbc.Nav([dbc.NavLink("Pipeline", href = '/pipeline', active = "exact")]),
        html.A(
            dbc.Col(html.Img(src="/assets/home.png", height="30px")),
            href="/home",
        )

    ], className= 'navbar navbar-expand-lg navbar-dark bg-primary')
)

# embedding the navigation bar
app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    navbar,
    html.Div(id='page-content')
])


@app.callback(Output('page-content', 'children'),
              [Input('url', 'pathname')])
def display_page(pathname):
    if pathname == '/delcode_analysis':
        return delcode_analysis.layout
    elif pathname == '/pipeline':
        return modelling_pipeline.layout
    else:
        return home.layout


if __name__ == '__main__':
    app.run_server(debug=True)
