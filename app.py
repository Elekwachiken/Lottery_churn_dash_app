import dash
from dash import dcc, html, Input, Output, State, dash_table, ctx
import dash_bootstrap_components as dbc
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go
import base64
import io
import os
from dotenv import load_dotenv
from datetime import datetime
import numpy as np

# Load environment variables for credentials
load_dotenv()

# --- Load Model and Encoders ---
# Ensure the model and encoder files are in a 'model' directory
try:
    model = joblib.load('model/churn_model.pkl')
    gender_encoder = joblib.load('model/gender_encoder.pkl')
    country_encoder = joblib.load('model/country_encoder.pkl')
except FileNotFoundError as e:
    print(f"Error loading model/encoders: {e}. Make sure the 'model' directory and its contents exist.")
    # Exit or handle gracefully if model files are essential for startup
    model, gender_encoder, country_encoder = None, None, None

# --- App Initialization ---
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], suppress_callback_exceptions=True, assets_folder='assets')
app.title = 'Lottery Churn Prediction App'
server = app.server

# --- Helper Functions ---
def is_valid_user(username, password):
    """Checks for correct login credentials from environment variables."""
    i = 1
    while True:
        user_key = f"APP_USER{i}"
        pass_key = f"APP_PASS{i}"
        stored_user = os.getenv(user_key)
        stored_pass = os.getenv(pass_key)
        if not stored_user:
            break
        if stored_user == username and stored_pass == password:
            return True
        i += 1
    return False

def generate_visuals(df):
    """Creates a tabbed component with various charts from the dataframe."""
    if df.empty:
        return html.Div("No data to display visuals.", className="text-center p-4")

    # A simple, elegant style for chart titles
    def styled_title(text):
        return dict(title=text, title_x=0.5, font=dict(family='Nunito, sans-serif', size=18, color='#333'))

    tabs_children = []
    
    # Chart 1: Bar Chart
    try:
        df_counts = df['Predicted_Churn'].value_counts().reset_index()
        df_counts.columns = ['Predicted_Churn', 'count']
        bar_fig = px.bar(df_counts, x='Predicted_Churn', y='count', color='Predicted_Churn', labels={'count':'Number of Players', 'Predicted_Churn':'Churn Prediction'})
        bar_fig.update_layout(**styled_title('Player Churn Count'))
        tabs_children.append(dcc.Tab(label="Churn Count", children=[dcc.Graph(figure=bar_fig)]))
    except Exception as e:
        print(f"Bar Chart Error: {e}")

    # Chart 2: Pie Chart
    try:
        pie_fig = px.pie(df, names='Predicted_Churn', hole=0.4, title='Churn Proportions')
        pie_fig.update_layout(**styled_title('Player Churn Proportions'))
        tabs_children.append(dcc.Tab(label="Churn Breakdown", children=[dcc.Graph(figure=pie_fig)]))
    except Exception as e:
        print(f"Pie Chart Error: {e}")

    # Add more charts here following the same try/except pattern
    # (e.g., Boxplot, Histogram, etc.)

    return dcc.Tabs(tabs_children) if tabs_children else html.Div("Could not generate visuals.")


# --- Component Definitions ---

# Sidebar with collapsible submenus
sidebar = html.Div([
    html.H2([html.Span("🎲", className="icon"), html.Span("Churn Predictor", className="label")], className="sidebar-title"),
    html.Hr(),
    
    # Home Menu
    html.Div([
        html.Button([html.Span("🏠", className="icon"), html.Span("Home", className="label")], id="home-toggle", className="menu-header-button"),
        dbc.Collapse(
            dbc.Nav([
                dbc.NavLink("Dashboard", href="/", active="exact"),
            ], vertical=True, pills=True, className="submenu-nav"),
            id="home-collapse",
        ),
    ]),
    
    # Upload & Predict Menu
    html.Div([
        html.Button([html.Span("📤", className="icon"), html.Span("Upload & Predict", className="label")], id="upload-toggle", className="menu-header-button"),
        dbc.Collapse(
            dbc.Nav([
                dbc.NavLink("Upload CSV", href="/upload", active="exact"),
            ], vertical=True, pills=True, className="submenu-nav"),
            id="upload-collapse",
        ),
    ]),

    # About Menu
    html.Div([
        html.Button([html.Span("ℹ️", className="icon"), html.Span("About", className="label")], id="about-toggle", className="menu-header-button"),
        dbc.Collapse(
            dbc.Nav([
                dbc.NavLink("About App", href="/about", active="exact"),
            ], vertical=True, pills=True, className="submenu-nav"),
            id="about-collapse",
        ),
    ]),
    
    # Theme Toggle
    html.Div([
        html.Div([html.Span("🌓", className="icon"), html.Span("Dark Mode", className="label")], className="theme-label-row"),
        dbc.Checklist(options=[{"label": "", "value": "dark"}], value=[], id="theme-switch", switch=True)
    ], className="theme-toggle-section"),

    html.Footer("© 2024 Kenneth", className="sidebar-footer")
], id="sidebar", className="expanded")

# Top Navbar with User Dropdown
navbar = html.Div([
    html.Button("☰", id="toggle-sidebar", className="menu-toggle"),
    html.Div([html.Img(src='/assets/logo.png', className='logo'), html.H4("Lottery Churn Prediction")], className="logo-title"),
    html.Div(id="user-info-display", className="user-panel")
], className="navbar")

# Main Content Area
content = html.Div(id="page-content", className="content")

# --- Page Layouts ---
login_layout = html.Div([
    html.Div([
        html.Div([
            html.H1("🎯 Lottery Churn Prediction", className="login-title"),
            html.Img(src="/assets/lottery_banner.png", className="login-image")
        ], className="login-left"),
        html.Div([
            html.Div([
                html.H2("🔐 Admin Login"),
                dcc.Input(id="username", type="text", placeholder="Username", autoComplete="username", className="login-input"),
                dcc.Input(id="password", type="password", placeholder="Password", autoComplete="current-password", className="login-input"),
                html.Button("Login", id="login-button", className="login-btn"),
                html.Div(id="login-output", className="login-message")
            ], className="login-form-box")
        ], className="login-right"),
    ], className="login-wrapper")
])

home_layout = html.Div([
    html.H2("Welcome to the Online Lottery Churn Predictor"),
    html.P("This tool helps you identify players likely to churn based on behavioral data. Navigate using the sidebar."),
    html.P("To begin, head to the 'Upload & Predict' section to upload your CSV file and see predictions.")
])

upload_layout = html.Div([
    html.H4("Upload Your CSV File for Prediction", className="page-title"),
    html.A("📥 Download Sample CSV Template", href='/assets/sample_input.csv', download='sample_input.csv', className="template-btn"),
    dcc.Upload(
        id='upload-data',
        children=html.Div(['📤 Drag & Drop or ', html.A('Click to Upload CSV File')]),
        className="upload-box",
        multiple=False
    ),
    dcc.Loading(id="loading-spinner", type="circle", children=html.Div(id='output-data-upload'))
])

about_layout = html.Div([
    html.H2("About This App"),
    html.P("This Dash app uses a trained machine learning model to predict churn in online lottery players."),
    html.P("Built with ❤️ using Dash, Plotly, and scikit-learn.")
])

# --- Main App Layout ---
app.layout = html.Div([
    dcc.Store(id="auth-status", storage_type="local"),
    dcc.Store(id="theme-store", storage_type="local", data={'theme': 'light'}),
    dcc.Store(id="current-user", storage_type="local"),
    dcc.Download(id="download-csv"),
    dcc.Location(id="url", refresh=False),
    html.Div(id="app-container")
], id="main-container")

# --- Callbacks ---

# Main layout renderer: Login page vs App
@app.callback(Output('app-container', 'children'), Input('auth-status', 'data'))
def render_app_or_login(is_logged_in):
    if is_logged_in:
        return html.Div([sidebar, html.Div([navbar, content], className="main-body")], id="body-wrapper", className="sidebar-expanded")
    return login_layout

# Page router
@app.callback(Output('page-content', 'children'), Input('url', 'pathname'))
def display_page(pathname):
    if pathname == '/upload': return upload_layout
    if pathname == '/about': return about_layout
    return home_layout

# Login authentication
@app.callback(
    [Output('auth-status', 'data'), Output('login-output', 'children'), Output('current-user', 'data')],
    Input('login-button', 'n_clicks'),
    [State('username', 'value'), State('password', 'value')],
    prevent_initial_call=True
)
def authenticate(n_clicks, username, password):
    if not username or not password:
        return dash.no_update, "Please enter all credentials.", dash.no_update
    if is_valid_user(username.strip(), password.strip()):
        return True, f"✅ Welcome, {username}!", username.strip()
    return False, "❌ Invalid credentials.", dash.no_update

# User dropdown and logout
@app.callback(Output('user-info-display', 'children'), Input('current-user', 'data'))
def update_user_info(username):
    if username:
        return dbc.DropdownMenu(
            children=[
                dbc.DropdownMenuItem("View Profile", disabled=True),
                dbc.DropdownMenuItem("My Attendance", disabled=True),
                dbc.DropdownMenuItem(divider=True),
                dbc.DropdownMenuItem("Logout", id="logout-button"),
            ],
            nav=True, in_navbar=True, label=f"Welcome, {username}"
        )
    return []

@app.callback(
    [Output('auth-status', 'clear_data'), Output('current-user', 'clear_data'), Output('url', 'pathname', allow_duplicate=True)],
    Input('logout-button', 'n_clicks'),
    prevent_initial_call=True
)
def logout(n_clicks):
    return True, True, '/'

# Sidebar toggling
@app.callback(
    [Output('sidebar', 'className'), Output('body-wrapper', 'className')],
    Input('toggle-sidebar', 'n_clicks'),
    State('sidebar', 'className'),
    prevent_initial_call=True
)
def toggle_sidebar_class(n, current_class):
    if n:
        if 'collapsed' in current_class:
            return 'expanded', 'sidebar-expanded'
        return 'collapsed', 'sidebar-collapsed'
    return dash.no_update, dash.no_update

# Submenu toggling
def toggle_collapse(n, is_open):
    if n:
        return not is_open
    return is_open

for menu in ["home", "upload", "about"]:
    app.callback(
        Output(f"{menu}-collapse", "is_open"),
        Input(f"{menu}-toggle", "n_clicks"),
        State(f"{menu}-collapse", "is_open"),
        prevent_initial_call=True,
    )(toggle_collapse)

# Theme switching
@app.callback(Output('theme-store', 'data'), Input('theme-switch', 'value'))
def switch_theme(switch_value):
    return {'theme': "dark"} if switch_value else {'theme': "light"}

@app.callback(Output('main-container', 'className'), Input('theme-store', 'data'))
def update_main_container_theme(theme_data):
    return theme_data.get('theme', 'light') if theme_data else 'light'

# File upload and prediction
@app.callback(
    Output('output-data-upload', 'children'),
    Input('upload-data', 'contents'),
    State('upload-data', 'filename'),
    prevent_initial_call=True
)
def handle_file_upload(contents, filename):
    if not contents: return html.Div()
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    try:
        if 'csv' not in filename:
            return html.Div(['Invalid file type. Please upload a .csv file.'], className="error-message")

        df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
        original_df = df.copy()

        df['Gender'] = gender_encoder.transform(df['Gender'])
        df['Country'] = country_encoder.transform(df['Country'])
        X = df.drop(columns=['Player_ID'], errors='ignore')
        
        original_df['Predicted_Churn'] = model.predict(X)
        stored_data = original_df.to_json(orient='split')

        return html.Div([
            dcc.Store(id='predicted-data-store', data=stored_data),
            html.Button("⬇ Download Predictions as CSV", id="download-btn", className="download-btn"),
            html.Hr(),
            html.H5(f"Prediction Results for: {filename}"),
            dash_table.DataTable(
                columns=[{"name": i, "id": i} for i in original_df.columns],
                data=original_df.to_dict('records'), page_size=10, style_table={'overflowX': 'auto'}
            ),
            html.Hr(),
            html.H5("Visual Analytics"),
            generate_visuals(original_df)
        ])
    except Exception as e:
        return html.Div([f'An error occurred: {e}'], className="error-message")

# Download handler
@app.callback(
    Output('download-csv', 'data'),
    Input('download-btn', 'n_clicks'),
    State('predicted-data-store', 'data'),
    prevent_initial_call=True
)
def download_predictions(n_clicks, stored_data):
    if stored_data:
        df = pd.read_json(stored_data, orient='split')
        return dcc.send_data_frame(df.to_csv, "churn_predictions.csv", index=False)
    return dash.no_update

if __name__ == '__main__':
    app.run(debug=True)
