import dash
import dash_bootstrap_components as dbc

# https://bootswatch.com/cerulian/
external_stylesheets = [dbc.themes.SANDSTONE]

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

server = app.server
app.config.suppress_callback_exceptions = True

