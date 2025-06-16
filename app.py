import dash
from dash import dcc, html, Input, Output, State, dash_table, ctx, callback_context
import dash_bootstrap_components as dbc
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objs as go
import base64
import io
import os
from dotenv import load_dotenv
from datetime import datetime
import numpy as np


load_dotenv() # For environment variables credentials

# Load model and encoders
model = joblib.load('model/churn_model.pkl')
gender_encoder = joblib.load('model/gender_encoder.pkl')
country_encoder = joblib.load('model/country_encoder.pkl')

# External stylesheet: custom CSS and Bootstrap for responsiveness
external_stylesheets = [dbc.themes.FLATLY, '/assets/custom.css']

# Initialize app
app = dash.Dash(__name__, external_stylesheets=external_stylesheets, suppress_callback_exceptions=True)
app.title = 'Lottery Churn Prediction App'
server = app.server  # for deployment


# === Helper Functions ===
# To check for correct login credentials
def is_valid_user(username, password):
    i = 1
    while True:
        user_key = f"APP_USER{i}"
        pass_key = f"APP_PASS{i}"
        stored_user = os.getenv(user_key)
        stored_pass = os.getenv(pass_key)
        if not stored_user or not stored_pass:
            break
        if stored_user == username and stored_pass == password:
            return True
        i += 1
    return False

# Reusable layout wrapper
def wrap_layout(content, is_logged_in, theme, username=None, login_time=None):
    class_name = f"page-container {'dark-mode' if theme == 'dark' else ''}"
    greeting = f"Welcome, {username}" if username else "Churn Predictor"
    session_info = f" | Logged in at: {login_time}" if login_time else ""

    header = html.Div([
        html.H2(greeting + session_info, style={"margin": "10px"}),
        html.Button("🔓 Logout", id="logout-button", n_clicks=0, className="logout-btn")
    ], style={"display": "flex", "justifyContent": "space-between", "alignItems": "center"},
    ) if is_logged_in else None

    return html.Div([header, content] if header else [content], 
                    id="page-container", 
                    className=class_name)


# === Robust Visual Charts Builder ===
# === Also Store CSV globally for download ===
predicted_csv = pd.DataFrame()

def generate_visuals(df):
    visuals = []

    def styled_title(text):
        return dict(title=f"<b style='font-family:Nunito;font-size:20px;color:#444;'>{text}</b>",
                    title_x=0.5, font=dict(family='Nunito'))

    # 1. Bar chart of churn count
    try:
        df_counts = df['Predicted_Churn'].value_counts().reset_index(name='count').rename(columns={'index': 'Predicted_Churn'})
        bar_fig = px.bar(df_counts, x='Predicted_Churn', y='count', color='Predicted_Churn', title='Bar Chart - Churn Count')
        bar_fig.update_layout(**styled_title('Churn Count (Bar Chart)'))
    except Exception as e:
        bar_fig = go.Figure().update_layout(title=f"Bar Chart Error: {e}")

    # 2. Pie chart
    try:
        pie_fig = px.pie(df, names='Predicted_Churn', title='Pie Chart - Churn Breakdown')
        pie_fig.update_layout(**styled_title('Churn Breakdown (Pie Chart)'))
    except Exception as e:
        pie_fig = go.Figure().update_layout(title=f"Pie Chart Error: {e}")

    # 3. Boxplot of Total_Deposits by Churn
    try:
        if 'Total_Deposits' in df.columns:
            box_fig = px.box(df, x='Predicted_Churn', y='Total_Deposits', title='Boxplot - Total Deposits by Churn')
            box_fig.update_layout(**styled_title('Deposits by Churn (Boxplot)'))
        else:
            raise ValueError("Column 'Total_Deposits' not found")
    except Exception as e:
        box_fig = go.Figure().update_layout(title=f"Boxplot Error: {e}")

    # 4. Histogram of session duration
    try:
        if 'Average_Session_Duration_Minutes' in df.columns:
            hist_fig = px.histogram(df[df['Predicted_Churn'] == 1], x='Average_Session_Duration_Minutes', nbins=30, title='Histogram - Session Duration for Churned Users')
            hist_fig.update_layout(**styled_title('Churned Sessions (Histogram)'))
        else:
            raise ValueError("Column 'Average_Session_Duration_Minutes' not found")
    except Exception as e:
        hist_fig = go.Figure().update_layout(title=f"Histogram Error: {e}")

    # 5. Heatmap of correlations
    try:
        numeric_df = df.select_dtypes(include=[np.number]).drop(columns=['Player_ID'], errors='ignore')
        corr = numeric_df.corr()
        heatmap_fig = px.imshow(corr, text_auto=True, title='Heatmap - Feature Correlation')
        heatmap_fig.update_layout(**styled_title('Feature Correlation (Heatmap)'))
    except Exception as e:
        heatmap_fig = go.Figure().update_layout(title=f"Heatmap Error: {e}")

    # 6. Stacked bar chart - Gender vs Churn
    try:
        if 'Gender' in df.columns:
            stack_fig = px.histogram(df, x='Gender', color='Predicted_Churn', barmode='stack', title='Stacked Bar - Gender vs Churn')
            stack_fig.update_layout(**styled_title('Gender vs Churn (Stacked Bar)'))
        else:
            raise ValueError("Column 'Gender' not found")
    except Exception as e:
        stack_fig = go.Figure().update_layout(title=f"Stacked Bar Error: {e}")

    # 7. Country-wise churn
    try:
        if 'Country' in df.columns:
            country_counts = df[df['Predicted_Churn'] == 1]['Country'].value_counts().reset_index(name='count').rename(columns={'index': 'Country'})
            country_fig = px.bar(country_counts, x='Country', y='count', title='Bar Chart - Country-wise Churn')
            country_fig.update_layout(**styled_title('Country-wise Churn'))
        else:
            raise ValueError("Column 'Country' not found")
    except Exception as e:
        country_fig = go.Figure().update_layout(title=f"Country Chart Error: {e}")

    # 8. Scatter Plot
    try:
        if all(col in df.columns for col in ['Game_Sessions_Last_30_Days', 'Total_Deposits']):
            scatter_fig = px.scatter(df, x='Game_Sessions_Last_30_Days', y='Total_Deposits', color='Predicted_Churn', title='Scatter - Sessions vs Deposits by Churn')
            scatter_fig.update_layout(**styled_title('Sessions vs Deposits (Scatter)'))
        else:
            raise ValueError("Columns 'Game_Sessions_Last_30_Days' and/or 'Total_Deposits' not found")
    except Exception as e:
        scatter_fig = go.Figure().update_layout(title=f"Scatter Plot Error: {e}")

    # 9. Feature Importance
    try:
        if hasattr(model, 'feature_importances_'):
            features = df.drop(columns=['Player_ID', 'Predicted_Churn'], errors='ignore').columns
            feat_imp_fig = px.bar(x=model.feature_importances_, y=features, orientation='h', title='Feature Importance')
            feat_imp_fig.update_layout(**styled_title('Feature Importance'))
        else:
            raise ValueError("Model does not support feature_importances_")
    except Exception as e:
        feat_imp_fig = go.Figure().update_layout(title=f"Feature Importance Error: {e}")

    return dcc.Tabs([
        dcc.Tab(label="Churn Bar Chart", children=[dcc.Graph(figure=bar_fig)]),
        dcc.Tab(label="Churn Pie Chart", children=[dcc.Graph(figure=pie_fig)]),
        dcc.Tab(label="Deposits Boxplot", children=[dcc.Graph(figure=box_fig)]),
        dcc.Tab(label="Session Histogram", children=[dcc.Graph(figure=hist_fig)]),
        dcc.Tab(label="Feature Correlation", children=[dcc.Graph(figure=heatmap_fig)]),
        dcc.Tab(label="Gender vs Churn", children=[dcc.Graph(figure=stack_fig)]),
        dcc.Tab(label="Country-wise Churn", children=[dcc.Graph(figure=country_fig)]),
        dcc.Tab(label="Behavioral Scatter", children=[dcc.Graph(figure=scatter_fig)]),
        dcc.Tab(label="Feature Importance", children=[dcc.Graph(figure=feat_imp_fig)])
    ])


# === Application Components ===
# === Enhanced Sidebar with Submenus and Navigation Interactivity ===
# Sidebar structure with nested submenus for Home, Upload, About
sidebar = html.Div([
    html.H2([html.Span("🎲", className="icon"), html.Span("Churn Predictor App", className="label")], className="sidebar-title"),
    html.Hr(),
    # Vertical Navigation pages
    html.Div([html.Span("🏠", className="icon"), html.Span("Home", className="label")], className="menu-header", id="home-header"), 
    html.Div(id="home-submenu", className="", children=dbc.Nav([
        dbc.NavLink([html.Span("🏡", className="icon"), html.Span("Home", className="label")], href="/", active="exact", className="submenu-item"),
        dbc.NavLink([html.Span("📊", className="icon"), html.Span("Dashboard", className="label")], href="/dashboard", active="exact", className="submenu-item"),
        dbc.NavLink([html.Span("🌟", className="icon"), html.Span("Wall of Fame", className="label")], href="/wall-of-fame", active="exact", className="submenu-item")
    ], vertical=True, pills=True, className="menu-section")),

    html.Div([html.Span("📤", className="icon"), html.Span("Upload & Predict", className="label")], className="menu-header collapsible", id="upload-header"),
    html.Div(id="upload-submenu", className="", children=dbc.Nav([
        dbc.NavLink([html.Span("📥", className="icon"), html.Span("Upload CSV", className="label")], href="/upload", active="exact", className="submenu-item"),
        dbc.NavLink([html.Span("🔎", className="icon"), html.Span("Predictions", className="label")], href="/predictions", active="exact", className="submenu-item"),
        dbc.NavLink([html.Span("📈", className="icon"), html.Span("KPI", className="label")], href="/kpi", active="exact", className="submenu-item")
    ], vertical=True, pills=True, className="menu-section")),

    html.Div([html.Span("ℹ️", className="icon"), html.Span("About", className="label")], className="menu-header collapsible", id="about-header"),
    html.Div(id="about-submenu", className="", children=dbc.Nav([
        dbc.NavLink([html.Span("ℹ️", className="icon"), html.Span("About", className="label")], href="/about", active="exact", className="submenu-item"),
        dbc.NavLink([html.Span("🔍", className="icon"), html.Span("About App", className="label")], href="/about-app", active="exact", className="submenu-item")
    ], vertical=True, pills=True, className="menu-section")),

    html.Div([
        html.Div([
        html.Div([
            html.Span("🌓", className="icon"),
            html.Span("Theme", className="label")
        ], className="theme-label-row"),

        # dbc.Checklist(
        #     options=[{"label": "", "value": "dark"}],
        #     value=[],
        #     id="theme-switch",
        #     switch=True,
        #     className="theme-toggle-switch"
        # )
        # Use simpler dbc.Switch since we just need on/off, and not a list of toggle options
        dbc.Switch(
            id="theme-switch",
            label="Dark Mode",
            value=False,
            className="theme-toggle-switch"
        )
    ], className="theme-toggle-col"),

    dcc.Store(id="theme-store", storage_type="local")
], className="theme-section"),

    html.Footer("© 2025 Lottery Analytics App. Built by Kenneth")
], id="sidebar", className="sidebar expanded")

# Top navbar with user dropdown and sidebar toggle
navbar = html.Div([
    html.Button("☰", id="toggle-sidebar", className="menu-toggle"),
    html.Div([html.Img(src='/assets/logo.png', className='logo'), html.H4("🎲 Lottery Churn Prediction")], className="logo-title"),
    html.Div([
        html.Span("●", className="status-indicator"),
        html.Div([
            html.Span("KENNETH ▼", className="user-name"), 
            html.Div([
                html.Div("View Profile"),
                html.Div("My Attendance"),
                html.Div("Logout", className="logout-link", id="logout-button")
            ], className="user-dropdown")
        ], id="user-dropdown-container", className="user-panel")
    ], className="user-panel")
], className="navbar")

# Main content wrapper (Content Area)
content = html.Div(id="page-content", className="content")

# Application layout wrapper with responsive body wrapper
app.layout = html.Div([
    dcc.Location(id="redirect", refresh=True),
    dcc.Store(id="auth-status", storage_type="local", data=False), #tells Dash to store data in browser local storage, persisting it across refreshes or browser reopenings.
    dcc.Store(id="theme-store", storage_type="local", data={"theme": "light"}),
    dcc.Store(id="login-clicked", data=False, storage_type="session"), # Store to track login button clicks
    dcc.Store(id="current-user", storage_type="session"), # Stores the currently logged-in username
    dcc.Store(id="login-time", storage_type="session"), # Stores the currently login time
    html.Div(id="layout-container"),

    # html.Div([
    #     sidebar,
    #     html.Div([navbar, content], className="main-body")
    # ], className="body-wrapper")
])


# === Pages Layout ===
# Login Page Layout
login_layout = html.Div([
    html.Div([
        # Left column with background and title
        html.Div([
            html.H1("🎯 Lottery Churn Prediction", className="login-title"),
            html.Img(src="/assets/lottery_banner.png", className="login-image")
        ], className="login-left"),
        # Right column with login box
        html.Div([
            html.Div([
                html.H2("🔐 Admin Login"),
                dcc.Input(id="username", type="text", placeholder="Username",  autoComplete="username", className="login-input"),
                dcc.Input(id="password", type="password", placeholder="Password", autoComplete="current-password", className="login-input"),
                html.Button("Login", id="login-button", className="login-btn"),
                html.Div([
                    html.Div(id="login-output"),
                    dcc.Interval(id="clear-login-msg", interval=3000, n_intervals=0, disabled=True)
                ]),
            ], className="login-form-box")
        ], className="login-right"),
    ], className="login-wrapper")
])

# Logout button design/placement
logout_button = html.Div(
        html.Button("🔓 Logout", id="logout-button", n_clicks=0, className="logout-btn"),
        style={"textAlign": "right", "margin": "10px 20px"}
)


# Home page layout
home_layout = html.Div(
    id="page-container",
    children=[
        html.H2("Welcome to the Online Lottery Churn Predictor"),
        html.P("This tool helps you identify players likely to churn based on behavioral data. Navigate using the sidebar."),
        html.P("This app helps you predict whether lottery players are likely to churn based on uploaded customer data. You can navigate between the tabs/sidebar to upload your CSV file, see predictions, and read more about the model."),
        html.P("To begin, head to the 'Upload & Predict' tab.")
    ]
)

# Upload & Predict page layout
upload_layout = html.Div(
    id="page-container",
    className="page-layout",
    children=[
        html.H4("Upload Your CSV File for Prediction", className="page-title"),
        dcc.Upload(
            id='upload-data',
            children=html.Div(['📤 Drag & Drop or ', html.A('Click to Upload CSV File')]),
            className="upload-box",
            multiple=False
        ),
        html.Div(id='output-data-upload'),
        dcc.Loading(
            id="loading-spinner",
            type="circle",
            children=dcc.Graph(id='churn-graph', className="graph-section")
        ),
        html.Div([
            html.A("📥 Download Sample CSV Template", href='/assets/sample_input.csv', download='sample_input.csv', className="template-btn")
            ], style={'textAlign': 'center'}, className="download-link"),
        html.Footer("© 2025 Lottery Analytics App. Built by Kenneth", className="footer")
    ]
)

# About page layout
about_layout = html.Div(
    id="page-container",
    children=[
        html.H2("About This App"),
        html.P("This Dash app uses a trained machine learning model to predict churn in online lottery players."),
        html.P("This churn prediction model is trained using real-world anonymized data from lottery player behavior. The model uses features like demographics, transaction history, and gaming behavior to predict churn likelihood."),
        html.P("The predictions help marketing and CRM teams design better re-engagement campaigns for players at risk of churning."),
        html.P("Built with ❤️ using Dash, Plotly, and Render."),
        html.Footer("Built using Dash and Plotly • Model: Random Forest Classifier • Maintained by Kenneth")
    ]
)




# === Callbacks ===
# Callback to render either login or main layout
# === Layout Control Callback: Render login or full app ===
@app.callback(
    Output("layout-container", "children"),
    [Input("auth-status", "data"), Input("current-user", "data"), Input("login-time", "data")]
)
def render_layout(is_logged_in, username, login_time):
    if is_logged_in:
        return html.Div([
            html.Div([
                sidebar,
                html.Div([navbar, content], className="main-body")
                ], className="body-wrapper")
        ])
    return login_layout


# # Main Layout
# def main_layout(username, login_time):
#     return html.Div([
#         dcc.Location(id="url", refresh=False),
#         sidebar,
#         content, 
#         dcc.Store(id="username-store", data=username)
#     ])


# Callback to set "login-clicked" Store to True when login is attempted
@app.callback(
    Output("login-clicked", "data"),
    Input("login-button", "n_clicks"),
    prevent_initial_call=True
    )
def record_login_click(n):
    return True


# Login Authentication Callback (and clearing Positive/Negative message callback)
@app.callback([
    Output("auth-status", "data"),
    Output("login-output", "children"),
    Output("redirect", "pathname"),  # New Output to trigger redirect to homepage
    Output("clear-login-msg", "disabled"),  # Enable timer Interval component
    Output("current-user", "data"),
    Output("login-time", "data")
    ], [Input("login-button", "n_clicks"), 
        Input("clear-login-msg", "n_intervals")], 
        [State("username", "value"), State("password", "value"), State("login-clicked", "data")], 
        prevent_initial_call=True
        )
def authenticate(n_clicks, n_intervals, u, p, login_clicked):
    trigger_id = ctx.triggered_id

    if trigger_id == "login-button":
        if not login_clicked:
            # Don’t authenticate unless manually clicked
            raise dash.exceptions.PreventUpdate
        if not u or not p:
            return False, "Enter Username and Password", dash.no_update, False, None, dash.no_update
        # print(f"🔍 Username: {u}, Password: {p}")  # For debug
        # if u and p and u.strip() == "admin" and p.strip() == "secret123": # Replace with env vars in prod!
        #  if u and p and u.strip() == os.getenv("APP_USER") and p.strip() == os.getenv("APP_PASS"):
        if is_valid_user(u.strip(), p.strip()):
        # If login credentials is correct/authorized(among .env file variables)
            return True, "✅ Logged in successfully", "/", False, u.strip(), datetime.now().strftime("%Y-%m-%d %H:%M:%S") # Login and goes straight to Home page, and also enables timer to clear positive message popup
        return False, "❌ Invalid credentials", dash.no_update, False, None, dash.no_update # Denies login, no change in tab, and also enables timer to clear negative message popup
    elif trigger_id == "clear-login-msg":
        return dash.no_update, "", dash.no_update, True, dash.no_update, dash.no_update # Clear message, disable timer


# Logout Callback
@app.callback(
    [Output("auth-status", "data", allow_duplicate=True),
     Output("login-clicked", "data", allow_duplicate=True), # Reset `login-clicked` Store on logout
     Output("current-user", "data", allow_duplicate=True),
     Output("login-time", "data", allow_duplicate=True)], 
    Input("logout-button", "n_clicks"),
    prevent_initial_call=True
)
def logout(n_clicks):
    if n_clicks:
        return False, False, None, None # This logs the user out. Clears auth status.
    return dash.no_update, dash.no_update, dash.no_update, dash.no_update



# === Sidebar & Submenu Toggle Callbacks ===
# Sidebar Collapse Toggle
@app.callback(
    Output("sidebar", "className"),
    Input("toggle-sidebar", "n_clicks"),
    State("sidebar", "className"),
    prevent_initial_call=True
)
def toggle_sidebar(n_clicks, current_class):
    if "collapsed" in current_class:
        return "sidebar expanded"
    else:
        return "sidebar collapsed"

# # Submenu Collapse: Auto-collapse others when one expands
# @app.callback(
#     Output("home-submenu", "style"),
#     Output("upload-submenu", "style"),
#     Output("about-submenu", "style"),
#     Input("home-header", "n_clicks"),
#     Input("upload-header", "n_clicks"),
#     Input("about-header", "n_clicks"),
#     prevent_initial_call=True
# )
# def toggle_submenus(home_click, upload_click, about_click):
#     ctx_id = callback_context.triggered[0]['prop_id'].split('.')[0]
#     if ctx_id == "home-header":
#         return {"height": "auto", "overflow": "hidden", "transition": "all 0.3s ease"}, {"height": 0, "overflow": "hidden", "transition": "all 0.3s ease"}, {"height": 0, "overflow": "hidden", "transition": "all 0.3s ease"}
#     elif ctx_id == "upload-header":
#         return {"height": 0, "overflow": "hidden", "transition": "all 0.3s ease"}, {"height": "auto", "overflow": "hidden", "transition": "all 0.3s ease"}, {"height": 0, "overflow": "hidden", "transition": "all 0.3s ease"}
#     elif ctx_id == "about-header":
#         return {"height": 0, "overflow": "hidden", "transition": "all 0.3s ease"}, {"height": 0, "overflow": "hidden", "transition": "all 0.3s ease"}, {"height": "auto", "overflow": "hidden", "transition": "all 0.3s ease"}
#     return dash.no_update

# Submenu Toggle: Allow multiple menus open at once (CSS handles animation)
@app.callback(
    Output("home-submenu", "className"),
    Input("home-header", "n_clicks"),
    State("home-submenu", "className"),
    prevent_initial_call=True
)
def toggle_home_submenu(n, current_class):
    return "" if "open" in (current_class or "") else "open"

@app.callback(
    Output("upload-submenu", "className"),
    Input("upload-header", "n_clicks"),
    State("upload-submenu", "className"),
    prevent_initial_call=True
)
def toggle_upload_submenu(n, current_class):
    return "" if "open" in (current_class or "") else "open"

@app.callback(
    Output("about-submenu", "className"),
    Input("about-header", "n_clicks"),
    State("about-submenu", "className"),
    prevent_initial_call=True
)
def toggle_about_submenu(n, current_class):
    return "" if "open" in (current_class or "") else "open"


# Merged `display_page` callback for the three of login, dark theme, and Navlink
@app.callback(
    Output("page-content", "children"),
    [Input("redirect", "pathname"), Input("theme-store", "data"), Input("current-user", "data"), Input("login-time", "data")]
)
def display_page(pathname, theme_data, username, login_time):
    theme = theme_data.get("theme", "light") if theme_data else "light"

    layout_map = {
        "/": home_layout,
        "/dashboard": html.Div([html.H2("📊 Dashboard"), html.P("Analytics overview.")]),
        "/wall-of-fame": html.Div([html.H2("🌟 Wall of Fame"), html.P("Top performing users.")]),
        "/upload": upload_layout,
        "/kpi": html.Div([html.H2("📈 KPI"), html.P("Key performance indicators.")]),
        "/predictions": html.Div([html.H2("📁 Predictions"), html.P("Saved prediction summaries.")]),
        "/about": about_layout,
        "/about-app": html.Div([html.H2("🔍 About Prediction App"), html.P("Details and credits.")])
    }

    layout = layout_map.get(pathname, html.Div([html.H2("404"), html.P("Page not found.")]))
    return wrap_layout(layout, is_logged_in=True, theme=theme, username=username, login_time=login_time)


# Dark theme callback
@app.callback(
    Output("theme-store", "data"),
    Input("theme-switch", "value")
)
def update_theme(value):
    return {"theme": "dark"} if value else {"theme": "light"}


# === CSV Download ===
@app.callback(
    Output("download-csv", "data"),
    Input("download-btn", "n_clicks"),
    prevent_initial_call=True
)
def download_csv(n):
    if not predicted_csv.empty:
        return dcc.send_data_frame(predicted_csv.to_csv, filename="predictions.csv", index=False)
    return dash.no_update

# Visual Contents callback
@app.callback(
    [Output('output-data-upload', 'children'),
     Output('churn-graph', 'figure')], 
     [Input('upload-data', 'contents')], State('upload-data', 'filename')
)
def update_output(contents, filename):
    global predicted_csv

    if contents is not None:
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
        try:
            df['Gender'] = gender_encoder.transform(df['Gender'])
            df['Country'] = country_encoder.transform(df['Country'])
        except Exception as e:
            return html.Div(f"⚠️ Error encoding categories: {str(e)}"), dash.no_update

        X = df.drop(columns=['Player_ID'], errors='ignore')
        df['Predicted_Churn'] = model.predict(X)

        # Decode back Gender and Country to show their text
        try:
            df['Gender'] = gender_encoder.inverse_transform(df['Gender'])
            df['Country'] = country_encoder.inverse_transform(df['Country'])
        except Exception:
            pass

        predicted_csv = df.copy()  # save for download

        table = dash_table.DataTable(
            columns=[{"name": i, "id": i} for i in df.columns],
            data=df.head(6).to_dict('records'), # Shows only a sample of 6 rows
            style_table={'overflowX': 'auto', 'maxHeight': '500px'},
            style_cell={'textAlign': 'left', 'padding': '8px', 'fontSize': '13px', 'fontFamily': 'Nunito'},
            style_header={'backgroundColor': '#0077b6', 'color': 'white', 'fontWeight': 'bold'},
            style_data={'whiteSpace': 'normal', 'height': 'auto'}
        )

        visuals_tabs = generate_visuals(df)

        download_btn = html.Div([
            html.Br(),
            html.Button("⬇ Download Predictions CSV", id="download-btn", className="template-btn"),
            dcc.Download(id="download-csv")
        ], style={'textAlign': 'center'})

        # Return table + visual dashboard
        return html.Div([table, download_btn, html.Br(), visuals_tabs]), dash.no_update
    return html.Div(), {}


if __name__ == '__main__':
    app.run(debug=True, port=8051)
