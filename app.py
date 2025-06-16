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

def generate_visuals(df, theme):
    """
    Creates tabbed component with various charts from the dataframe,
    adapting colors based on the selected theme.
    """
    if df.empty:
        return html.Div("No data to display visuals.", className="text-center p-4")

    # Define chart aesthetics based on theme
    if theme == 'dark':
        text_color = '#f1f1f1'
        bg_color = '#22223b'
        plot_bg_color = '#22223b'
        grid_color = '#555'
    else:
        text_color = '#333'
        bg_color = '#ffffff'
        plot_bg_color = '#ffffff'
        grid_color = '#e0e0e0'

    # A simple, elegant style for chart titles consistent with app theme
    def styled_title(text):
        return dict(text=text, x=0.5, xanchor='center', font=dict(family='Nunito, sans-serif', size=18, color=text_color))

    # Default layout updates for Plotly figures
    def update_figure_layout(fig):
        fig.update_layout(
            paper_bgcolor=bg_color,
            plot_bgcolor=plot_bg_color,
            font=dict(color=text_color),
            xaxis=dict(gridcolor=grid_color, zerolinecolor=grid_color),
            yaxis=dict(gridcolor=grid_color, zerolinecolor=grid_color),
        )
        return fig

    # Initialize list to hold chart tabs
    tabs_children = []
    
    # 1. Bar chart of churn count
    try:
        df_counts = df['Predicted_Churn'].value_counts().reset_index(name='count').rename(columns={'index': 'Predicted_Churn'})
        bar_fig = px.bar(df_counts, x='Predicted_Churn', y='count', color='Predicted_Churn',
                         color_discrete_map={0: '#28a745', 1: '#dc3545'}) # Green for No Churn, Red for Churn
        bar_fig = update_figure_layout(bar_fig)
        bar_fig.update_layout(title=styled_title('Churn Count (Bar Chart)'))
        tabs_children.append(dcc.Tab(label="Churn Bar Chart", children=[dcc.Graph(figure=bar_fig, id='churn-count-graph')], className='custom-tab', selected_className='custom-tab--selected'))
    except Exception as e:
        print(f"Bar Chart Error: {e}")
        tabs_children.append(dcc.Tab(label="Churn Bar Chart", children=[html.Div(f"Failed to load Churn Count chart: {e}", className="error-message")], className='custom-tab', selected_className='custom-tab--selected'))

    # 2. Pie chart
    try:
        pie_fig = px.pie(df, names='Predicted_Churn', hole=0.4,
                         color_discrete_map={0: '#28a745', 1: '#dc3545'})
        pie_fig = update_figure_layout(pie_fig)
        pie_fig.update_layout(title=styled_title('Churn Breakdown (Pie Chart)'))
        tabs_children.append(dcc.Tab(label="Churn Pie Chart", children=[dcc.Graph(figure=pie_fig, id='churn-breakdown-graph')], className='custom-tab', selected_className='custom-tab--selected'))
    except Exception as e:
        print(f"Pie Chart Error: {e}")
        tabs_children.append(dcc.Tab(label="Churn Pie Chart", children=[html.Div(f"Failed to load Churn Breakdown chart: {e}", className="error-message")], className='custom-tab', selected_className='custom-tab--selected'))

    # 3. Boxplot of Total_Deposits by Churn
    try:
        if 'Total_Deposits' in df.columns:
            box_fig = px.box(df, x='Predicted_Churn', y='Total_Deposits',
                             color='Predicted_Churn',
                             color_discrete_map={0: '#28a745', 1: '#dc3545'})
            box_fig = update_figure_layout(box_fig)
            box_fig.update_layout(title=styled_title('Deposits by Churn (Boxplot)'))
        else:
            raise ValueError("Column 'Total_Deposits' not found in uploaded data for Boxplot.")
        tabs_children.append(dcc.Tab(label="Deposits Boxplot", children=[dcc.Graph(figure=box_fig)], className='custom-tab', selected_className='custom-tab--selected'))
    except Exception as e:
        print(f"Boxplot Error: {e}")
        tabs_children.append(dcc.Tab(label="Deposits Boxplot", children=[html.Div(f"Failed to load Deposits Boxplot: {e}", className="error-message")], className='custom-tab', selected_className='custom-tab--selected'))

    # 4. Histogram of session duration for churned users
    try:
        if 'Average_Session_Duration_Minutes' in df.columns:
            # Filter for churned users only
            churned_df = df[df['Predicted_Churn'] == 1]
            if not churned_df.empty:
                hist_fig = px.histogram(churned_df, x='Average_Session_Duration_Minutes', nbins=30,
                                        color_discrete_sequence=['#dc3545']) # Red for churned
                hist_fig = update_figure_layout(hist_fig)
                hist_fig.update_layout(title=styled_title('Churned Sessions (Histogram)'))
            else:
                hist_fig = go.Figure().update_layout(title=styled_title('No Churned Users to display Session Duration (Histogram)'))
        else:
            raise ValueError("Column 'Average_Session_Duration_Minutes' not found in uploaded data for Histogram.")
        tabs_children.append(dcc.Tab(label="Session Histogram", children=[dcc.Graph(figure=hist_fig)], className='custom-tab', selected_className='custom-tab--selected'))
    except Exception as e:
        print(f"Histogram Error: {e}")
        tabs_children.append(dcc.Tab(label="Session Histogram", children=[html.Div(f"Failed to load Session Histogram: {e}", className="error-message")], className='custom-tab', selected_className='custom-tab--selected'))

    # 5. Heatmap of correlations
    try:
        numeric_df = df.select_dtypes(include=[np.number]).drop(columns=['Player_ID'], errors='ignore')
        # Ensure 'Predicted_Churn' is treated as numeric for correlation calculation
        if 'Predicted_Churn' in numeric_df.columns:
             numeric_df['Predicted_Churn'] = numeric_df['Predicted_Churn'].astype(float)
        
        if not numeric_df.empty and len(numeric_df.columns) > 1:
            corr = numeric_df.corr()
            heatmap_fig = px.imshow(corr, text_auto=True, color_continuous_scale=px.colors.sequential.Viridis)
            heatmap_fig = update_figure_layout(heatmap_fig)
            heatmap_fig.update_layout(title=styled_title('Feature Correlation (Heatmap)'),
                                      # --- NEW HEATMAP LAYOUT ADJUSTMENTS ---
                                      height=700, # Increase height of the plot
                                      width=800,  # Increase width if needed, or let it be responsive
                                      xaxis_showgrid=False, # Hide grid lines for cleaner look
                                      yaxis_showgrid=False,
                                      xaxis=dict(tickangle=45), # Angle x-axis labels for readability
                                      yaxis=dict(tickangle=0),   # Keep y-axis labels horizontal
                                      margin=dict(l=100, r=100, b=100, t=100) # Increase margins for labels
                                      # --- END NEW ADJUSTMENTS ---
                                     )
        else:
            raise ValueError("Not enough numeric columns for correlation heatmap or data is empty.")
        tabs_children.append(dcc.Tab(label="Feature Correlation", children=[dcc.Graph(figure=heatmap_fig)], className='custom-tab', selected_className='custom-tab--selected'))
    except Exception as e:
        print(f"Heatmap Error: {e}")
        tabs_children.append(dcc.Tab(label="Feature Correlation", children=[html.Div(f"Failed to load Feature Correlation: {e}", className="error-message")], className='custom-tab', selected_className='custom-tab--selected'))

    # 6. Stacked bar chart - Gender vs Churn
    try:
        if 'Gender' in df.columns:
            stack_fig = px.histogram(df, x='Gender', color='Predicted_Churn', barmode='stack',
                                     color_discrete_map={0: '#28a745', 1: '#dc3545'})
            stack_fig = update_figure_layout(stack_fig)
            stack_fig.update_layout(title=styled_title('Gender vs Churn (Stacked Bar)'))
        else:
            raise ValueError("Column 'Gender' not found in uploaded data for Stacked Bar Chart.")
        tabs_children.append(dcc.Tab(label="Gender vs Churn", children=[dcc.Graph(figure=stack_fig)], className='custom-tab', selected_className='custom-tab--selected'))
    except Exception as e:
        print(f"Stacked Bar Error: {e}")
        tabs_children.append(dcc.Tab(label="Gender vs Churn", children=[html.Div(f"Failed to load Gender vs Churn chart: {e}", className="error-message")], className='custom-tab', selected_className='custom-tab--selected'))

    # 7. Country-wise churn (Bar Chart)
    try:
        if 'Country' in df.columns:
            churn_by_country = df.groupby(['Country', 'Predicted_Churn']).size().unstack(fill_value=0).reset_index()
            churn_by_country.columns = ['Country', 'Not Churn', 'Churn']
            country_fig = px.bar(churn_by_country, x='Country', y=['Not Churn', 'Churn'],
                                 barmode='group',
                                 color_discrete_map={'Not Churn': '#28a745', 'Churn': '#dc3545'})
            country_fig = update_figure_layout(country_fig)
            country_fig.update_layout(title=styled_title('Country-wise Churn'))
        else:
            raise ValueError("Column 'Country' not found in uploaded data for Country-wise Churn chart.")
        tabs_children.append(dcc.Tab(label="Country-wise Churn", children=[dcc.Graph(figure=country_fig)], className='custom-tab', selected_className='custom-tab--selected'))
    except Exception as e:
        print(f"Country Chart Error: {e}")
        tabs_children.append(dcc.Tab(label="Country-wise Churn", children=[html.Div(f"Failed to load Country-wise Churn chart: {e}", className="error-message")], className='custom-tab', selected_className='custom-tab--selected'))

    # 8. Scatter Plot - Game Sessions vs Total Deposits by Churn
    try:
        if all(col in df.columns for col in ['Game_Sessions_Last_30_Days', 'Total_Deposits']):
            scatter_fig = px.scatter(df, x='Game_Sessions_Last_30_Days', y='Total_Deposits', color='Predicted_Churn',
                                     color_discrete_map={0: '#28a745', 1: '#dc3545'},
                                     hover_data=['Player_ID'])
            scatter_fig = update_figure_layout(scatter_fig)
            scatter_fig.update_layout(title=styled_title('Sessions vs Deposits (Scatter)'))
        else:
            raise ValueError("Columns 'Game_Sessions_Last_30_Days' and/or 'Total_Deposits' not found for Scatter Plot.")
        tabs_children.append(dcc.Tab(label="Behavioral Scatter", children=[dcc.Graph(figure=scatter_fig)], className='custom-tab', selected_className='custom-tab--selected'))
    except Exception as e:
        print(f"Scatter Plot Error: {e}")
        tabs_children.append(dcc.Tab(label="Behavioral Scatter", children=[html.Div(f"Failed to load Behavioral Scatter Plot: {e}", className="error-message")], className='custom-tab', selected_className='custom-tab--selected'))

    # 9. Feature Importance
    try:
        # Features should be derived from the input X used for prediction
        # Ensure we have the model and it has feature importances
        if model and hasattr(model, 'feature_importances_') and hasattr(model, 'feature_names_in_'):
            features = model.feature_names_in_ # Use feature names from the model if available
            feature_importances = model.feature_importances_
            
            # Create a DataFrame for sorting
            feature_imp_df = pd.DataFrame({
                'Feature': features,
                'Importance': feature_importances
            }).sort_values(by='Importance', ascending=True) # Sort for a nice bar chart

            feat_imp_fig = px.bar(feature_imp_df, x='Importance', y='Feature', orientation='h',
                                  color_discrete_sequence=['#0077b6'])
            feat_imp_fig = update_figure_layout(feat_imp_fig)
            feat_imp_fig.update_layout(title=styled_title('Feature Importance'))
        else:
            raise ValueError("Model does not have 'feature_importances_' or 'feature_names_in_' attribute.")
        tabs_children.append(dcc.Tab(label="Feature Importance", children=[dcc.Graph(figure=feat_imp_fig)], className='custom-tab', selected_className='custom-tab--selected'))
    except Exception as e:
        print(f"Feature Importance Error: {e}")
        tabs_children.append(dcc.Tab(label="Feature Importance", children=[html.Div(f"Failed to load Feature Importance chart: {e}", className="error-message")], className='custom-tab', selected_className='custom-tab--selected'))


    return dcc.Tabs(tabs_children, id='visual-analytics-tabs') if tabs_children else html.Div("Could not generate visuals.")




# --- Component Definitions ---
# === Enhanced Sidebar with Submenus and Navigation Interactivity ===
# Sidebar structure with nested submenus for Home, Upload, About
sidebar = html.Div([
    html.H2([html.Span("🎲", className="icon"), html.Span("Churn Predictor App", className="label")], className="sidebar-title"),
    html.Hr(),
    # Vertical Navigation Pages
    # Home Menu Section
    html.Div([
        html.Div([html.Span("🏠", className="icon"), html.Span("Home", className="label")], className="menu-header", id="home-header"), 
        html.Div(id="home-submenu", className="expanded-submenu menu-section", children=dbc.Nav([ # Initial state expanded
            dbc.NavLink([html.Span("🏡", className="icon"), html.Span("Home", className="label")], href="/", active="exact", className="submenu-item"),
            dbc.NavLink([html.Span("📊", className="icon"), html.Span("Dashboard", className="label")], href="/dashboard", active="exact", className="submenu-item"),
            dbc.NavLink([html.Span("🌟", className="icon"), html.Span("Wall of Fame", className="label")], href="/wall-of-fame", active="exact", className="submenu-item")
        ], vertical=True, pills=True))
    ], className="sidebar-menu-group"), # Added class for grouping in CSS

    # Upload & Predict Menu Section
    html.Div([
        html.Div([html.Span("📤", className="icon"), html.Span("Upload & Predict", className="label")], className="menu-header collapsible", id="upload-header"),
        html.Div(id="upload-submenu", className="collapsed-submenu menu-section", children=dbc.Nav([ # Initial state collapsed
            dbc.NavLink([html.Span("📥", className="icon"), html.Span("Upload CSV", className="label")], href="/upload", active="exact", className="submenu-item"),
            dbc.NavLink([html.Span("🔎", className="icon"), html.Span("Predictions", className="label")], href="/predictions", active="exact", className="submenu-item"),
            dbc.NavLink([html.Span("📈", className="icon"), html.Span("KPI", className="label")], href="/kpi", active="exact", className="submenu-item")
        ], vertical=True, pills=True))
    ], className="sidebar-menu-group"), # Added class for grouping in CSS

    # About Menu Section
    html.Div([
        html.Div([html.Span("ℹ️", className="icon"), html.Span("About", className="label")], className="menu-header collapsible", id="about-header"),
        html.Div(id="about-submenu", className="collapsed-submenu menu-section", children=dbc.Nav([ # Initial state collapsed
            dbc.NavLink([html.Span("ℹ️", className="icon"), html.Span("About", className="label")], href="/about", active="exact", className="submenu-item"),
            dbc.NavLink([html.Span("🔍", className="icon"), html.Span("About App", className="label")], href="/about-app", active="exact", className="submenu-item")
        ], vertical=True, pills=True))
    ], className="sidebar-menu-group"), # Added class for grouping in CSS

    # Theme Toggle Section
    html.Div([
        html.Div([
            html.Div([ # This div wraps the icon and label to keep them stacked above the switch
                html.Span("🌓", className="icon"),
                html.Span("Theme", className="label")
            ], className="theme-label-row"),
            dbc.Switch( # Changed to dbc.Switch
                id="theme-switch",
                label="Dark Mode", # Label for the switch itself
                value=False, # Initial value will be False (light mode)
                className="theme-toggle-switch" # Custom class for styling
            )
        ], className="theme-toggle-col"), # New wrapper for stacking label above switch

        # dcc.Store for theme preference ('light' or 'dark') is in app.layout
    ], className="theme-section"), # Overall container for the theme toggle

    html.Footer("© 2025 Lottery Analytics App. Built by Kenneth", className="sidebar-footer")
], id="sidebar", className="sidebar expanded") # Initial class for sidebar


# Top navbar with user dropdown and sidebar toggle
navbar = html.Div([
    html.Button("☰", id="toggle-sidebar", className="menu-toggle"),
    html.Div([html.Img(src='/assets/logo.png', className='logo'), html.H4("🎲 Lottery Churn Prediction")], className="logo-title"),
    html.Div(id="user-info-display", className="user-panel") # This will be dynamically updated by a callback
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
    html.H2("Welcome to the Online Lottery Churn Predictor", className="page-title"),
    html.P("This tool helps you identify players likely to churn based on behavioral data. Navigate using the sidebar."),
    html.P("To begin, head to the 'Upload & Predict' section to upload your CSV file and see predictions.")
], className="page-content-wrapper") # Added wrapper for consistent padding/margin

# Placeholder layouts for new pages
dashboard_layout = html.Div([html.H2("Dashboard", className="page-title"), html.P("This is the Dashboard page.")], className="page-content-wrapper")
wall_of_fame_layout = html.Div([html.H2("Wall of Fame", className="page-title"), html.P("This is the Wall of Fame page.")], className="page-content-wrapper")
predictions_layout = html.Div([html.H2("Predictions Overview", className="page-title"), html.P("This page shows detailed predictions.")], className="page-content-wrapper")
kpi_layout = html.Div([html.H2("Key Performance Indicators (KPI)", className="page-title"), html.P("This page displays key performance indicators.")], className="page-content-wrapper")
about_app_layout = html.Div([html.H2("About This Application", className="page-title"), html.P("More detailed information about the application can be found here.")], className="page-content-wrapper")


upload_layout = html.Div([
    html.H4("Upload Your CSV File for Prediction", className="page-title"),
    html.A("📥 Download Sample CSV Template", href='/assets/sample_input.csv', download='sample_input.csv', className="template-btn"),
    dcc.Upload(
        id='upload-data',
        children=html.Div(['📤 Drag & Drop or ', html.A('Click to Upload CSV File')]),
        className="upload-box",
        multiple=False # Only allow single file upload
    ),
    html.Div(id='output-data-upload') # Added this div to display upload results and prediction table
], className="page-content-wrapper") # Added wrapper

about_layout = html.Div([
    html.H2("About This App", className="page-title"),
    html.P("This Dash app uses a trained machine learning model to predict churn in online lottery players. "
            "It leverages historical player data to forecast which users are at risk of churning, "
            "allowing for proactive retention strategies."),
    html.P("The application is built with Dash for interactive web dashboards, Plotly for compelling visualizations, "
            "and scikit-learn for the underlying machine learning model."),
    html.P("Built with ❤️ using Dash, Plotly, and scikit-learn.")
], className="page-content-wrapper") # Added wrapper



# --- Main App Layout ---
# This is the root layout of the entire application.
app.layout = html.Div([
    # dcc.Store for authentication status (True/False for logged in)
    dcc.Store(id="auth-status", storage_type="local"),
    # dcc.Store for theme preference ('light' or 'dark')
    dcc.Store(id="theme-store", storage_type="local", data={'theme': 'light'}),
    # dcc.Store to hold the current logged-in username
    dcc.Store(id="current-user", storage_type="local"),
    # dcc.Store to hold the login time
    dcc.Store(id="login-time", storage_type="local"), 
    # dcc.Store to hold the predicted data for download and dynamic visual analytics
    dcc.Store(id='predicted-data-store', storage_type='session'), # Changed to session storage
    # dcc.Download component to trigger CSV file downloads
    dcc.Download(id="download-csv"),
    # dcc.Location component to track and update the browser's URL
    dcc.Location(id="url", refresh=False),
    # This div will conditionally render the login page or the main application layout
    html.Div(id="app-container"),

        # Global container for prediction output and loading spinner
    # This ensures 'output-prediction-table-and-data' always exists in the DOM
    # but its visibility is controlled by a callback
    html.Div(
        id='global-prediction-output-container',
        children=[
            dcc.Loading(
                id="loading-spinner",
                type="circle",
                children=html.Div(id='output-prediction-table-and-data')
            )
        ],
        style={'display': 'none'} # Hidden by default
    ),
    # Dummy output for client-side callback (FIXED: Always present in layout)
    html.Div(id='dummy-output', style={'display': 'none'}) 
], id="main-container") # This ID is used for theme switching in CSS









#     # --- NEW PLACEMENT FOR LOADING SPINNER ---
#     # Move dcc.Loading to wrap the actual content update area for predictions
#     dcc.Loading(
#         id="loading-predictions", # New ID for clarity
#         type="circle",
#         fullscreen=True, # Optional: Make it full screen for a more prominent effect
#         children=html.Div(id='output-data-upload-and-spinner-wrapper', children=[
#             # This is where the upload layout will be rendered,
#             # and within it, the 'output-data-upload' will receive content.
#             # We will modify the handle_file_upload callback to target this wrapper directly.
#             html.Div(id='output-data-upload') # The actual content from file upload will land here
#         ])
#     ),
#     # The 'global-prediction-output-container' is now redundant if we place spinner like this
#     # You can remove 'global-prediction-output-container' and its style output from the callback
#     # as the loading spinner now directly wraps the area where the upload results appear.

#     html.Div(id='dummy-output', style={'display': 'none'})
# ], id="main-container")






# --- CALLBACKS ---
# Callback to render either the login page or the main application layout
@app.callback(
    Output('app-container', 'children'),
    Input('auth-status', 'data')
)
def render_app_or_login(is_logged_in):
    """Renders the main application or the login page based on auth status."""
    # If logged in, show the main app structure (sidebar + navbar + content)
    # The 'sidebar-expanded' class will control the layout behavior based on sidebar state
    if is_logged_in:
        return html.Div([
            sidebar, 
            html.Div([navbar, content], 
                     className="main-body")], id="body-wrapper", className="sidebar-expanded"
        )
    # If auth_status is None (initial load or localStorage cleared), force login page
    return login_layout



# Callback to route between different pages based on URL pathname
@app.callback(
    Output('page-content', 'children'),
    Input('url', 'pathname')
)
def display_page(pathname):
    """Displays the content for the selected page."""
    # Depending on the pathname, return the appropriate layout.
    # The global-prediction-output-container will be controlled by a separate callback.
    if pathname == '/upload':
        return upload_layout
    elif pathname == '/about':
        return about_layout
    elif pathname == '/dashboard': # New page
        return dashboard_layout
    elif pathname == '/wall-of-fame': # New page
        return wall_of_fame_layout
    elif pathname == '/predictions': # New page
        return predictions_layout
    elif pathname == '/kpi': # New page
        return kpi_layout
    elif pathname == '/about-app': # New page
        return about_app_layout
    else:
        # Default to home layout if path is unrecognized or root
        return home_layout



# Callback for user authentication on login button click
@app.callback(
    [Output('auth-status', 'data'),  # Updates login status
     Output('login-output', 'children'), # Displays login message (success/failure)
     Output('url', 'pathname'),
     Output('current-user', 'data'),   # Stores the logged-in username
     Output('login-time', 'data')],
    Input('login-button', 'n_clicks'),
    [State('username', 'value'),
     State('password', 'value')],
    prevent_initial_call=True
)
def authenticate(n_clicks, username, password):
    """Handles user login authentication."""
    if not username or not password:
        return dash.no_update, "Please enter username and password.", dash.no_update, dash.no_update, dash.no_update
    
    if is_valid_user(username.strip(), password.strip()):
        login_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return True, f"✅ Welcome, {username}!", "/", username.strip(), login_time
    else:
        return False, "❌ Invalid credentials.", dash.no_update, dash.no_update, dash.no_update



# Callback to handle user logout
@app.callback(
    [Output('auth-status', 'clear_data'), # Clears authentication status
     Output('current-user', 'clear_data'), # Clears stored username
     Output('predicted-data-store', 'clear_data'), # Clear stored data on logout
     Output('output-prediction-table-and-data', 'children', allow_duplicate=True), # Clear displayed data
     Output('url', 'pathname', allow_duplicate=True)], # Redirects to login page
    Input('logout-button', 'n_clicks'),
    prevent_initial_call=True
)
def logout(n_clicks):
    if n_clicks is None: # Only proceed if a real click occurred
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update

    return True, True, True, html.Div(), '/' # Logs out the user and Clears output-prediction-table-and-data as well



# Callback to update the user dropdown menu in the navbar
@app.callback(
    Output('user-info-display', 'children'),
    [Input('current-user', 'data'),
     Input('auth-status', 'data')] # Added auth-status to determine indicator color
)
def update_user_info(username, is_logged_in):
    """Updates the navbar with user info and a logout button if a user is logged in."""
    status_color = 'green' if is_logged_in else 'red'
    user_name_display = username if username else "Guest"

    # Use dbc.DropdownMenu for standard dropdown behavior
    if is_logged_in:
        return dbc.DropdownMenu(
            children=[
                dbc.DropdownMenuItem("View Profile", disabled=True, className="user-dropdown-item"),
                dbc.DropdownMenuItem("My Attendance", disabled=True, className="user-dropdown-item"),
                dbc.DropdownMenuItem(divider=True),
                dbc.DropdownMenuItem("Logout", id="logout-button", className="logout-link user-dropdown-item"),
            ],
            nav=True,
            in_navbar=True,
            # The label now includes the status indicator and dynamic username
            label=html.Div([
                html.Span("●", className="status-indicator", style={'color': status_color, 'margin-right': '5px'}),
                html.Span(f"Welcome, {user_name_display}")
            ], className="user-dropdown-label"), # Custom class for the label's styling
            className="user-dropdown-menu" # Custom class for styling the dropdown toggle and menu
        )
    else:
        # For non-logged-in users, simply show "Guest" or a simplified representation
        return html.Div([
            html.Span("●", className="status-indicator", style={'color': status_color, 'margin-right': '5px'}),
            html.Span("Guest")
        ], className="guest-user-panel") # Different class for guest styling


# Callback to toggle the sidebar's collapsed/expanded state
@app.callback(
    [Output('sidebar', 'className'),
     Output('body-wrapper', 'className')],
    Input('toggle-sidebar', 'n_clicks'),
    State('sidebar', 'className'),
    prevent_initial_call=True
)
def toggle_sidebar(n, current_class):
    """Toggles the sidebar between expanded and collapsed states."""
    if n:
        if 'collapsed' in (current_class or ''):
            return 'sidebar expanded', 'sidebar-expanded main-body'
        else:
            return 'sidebar collapsed', 'sidebar-collapsed main-body'
    return dash.no_update, dash.no_update



# Callback to toggle submenu visibility using class names
@app.callback(
    [Output("home-submenu", "className"),
     Output("upload-submenu", "className"),
     Output("about-submenu", "className")],
    [Input("home-header", "n_clicks"),
     Input("upload-header", "n_clicks"),
     Input("about-header", "n_clicks")],
    [State("home-submenu", "className"),
     State("upload-submenu", "className"),
     State("about-submenu", "className")],
    prevent_initial_call=False # Allow initial run to set classes if not already done
)
def toggle_submenu_visibility(n_home, n_upload, n_about, home_class, upload_class, about_class):
    """
    Toggles the visibility of submenus by changing their class names.
    This replaces the dbc.Collapse logic for custom HTML submenus.
    """
    ctx_id = ctx.triggered_id if ctx.triggered else None

    # Default to current state if no click triggered, or based on initial setup
    new_home_class = home_class
    new_upload_class = upload_class
    new_about_class = about_class

    # Function to toggle a class between 'expanded-submenu' and 'collapsed-submenu'
    def toggle_class(current_cls):
        # Ensure 'menu-section' is always present
        if 'expanded-submenu' in current_cls:
            return "collapsed-submenu menu-section"
        elif 'collapsed-submenu' in current_cls:
            return "expanded-submenu menu-section"
        # Default to collapsed if no specific state, assuming hidden by default in CSS
        return "collapsed-submenu menu-section" # This handles initial load if classes are not set

    if ctx_id == "home-header":
        new_home_class = toggle_class(home_class)
    elif ctx_id == "upload-header":
        new_upload_class = toggle_class(upload_class)
    elif ctx_id == "about-header":
        new_about_class = toggle_class(about_class)
    
    return new_home_class, new_upload_class, new_about_class


# # Callback to switch theme preference and store it
# @app.callback(
#     Output('theme-store', 'data'),
#     Input('theme-switch', 'value') # dbc.Switch value is boolean
# )
# def switch_theme(is_dark_mode):
#     """Switches the theme between light and dark."""
#     theme = "dark" if is_dark_mode else "light"
#     return {'theme': theme}




# # NEW CALLBACK: To update the visual state of the theme switch on initial load/store change
# # This callback explicitly sets the dbc.Switch value based on the theme-store.
# @app.callback(
#     Output('theme-switch', 'value'), # Output to the switch's value (boolean)
#     Input('theme-store', 'data'),    # Input from the theme data store
#     prevent_initial_call=False       # Essential to set initial state correctly
# )
# def update_theme_switch_state(theme_data):
#     """
#     Updates the value of the dbc.Switch (theme switch) based on the theme stored in dcc.Store.
#     This ensures the switch visually reflects the current theme and can be toggled.
#     """
#     if theme_data: # Ensure theme_data is not None on initial load
#         current_theme_in_store = theme_data.get('theme', 'light')
#         target_switch_value = True if current_theme_in_store == 'dark' else False
#         return target_switch_value
#     return False # Default to light mode switch state if no theme_data



@app.callback(
    Output('theme-store', 'data'),
    Input('theme-switch', 'value'),
    prevent_initial_call=True
)
def update_store_from_switch(value):
    return {'theme': 'dark'} if value else {'theme': 'light'}




@app.callback(
    Output('theme-switch', 'value'),
    Input('theme-store', 'data'),
    prevent_initial_call=False
)
def sync_switch_with_store(theme_data):
    return theme_data.get('theme') == 'dark' if theme_data else False




# The clientside callback remains the same, it listens to theme-store and applies the class to the body.
app.clientside_callback(
    """
    function(themeData) {
        if (themeData && themeData.theme === 'dark') {
            document.body.classList.add('dark-mode');
        } else {
            document.body.classList.remove('dark-mode');
        }
        return window.dash_clientside.no_update; // No output needed for a direct DOM manipulation
    }
    """,
    Output('dummy-output', 'children'), # A dummy output is needed for clientside_callback
    Input('theme-store', 'data'),
    prevent_initial_call=False # Run on initial load to set theme
)


@app.callback(
    Output('main-container', 'className'),
    Input('theme-store', 'data')
)
def update_main_container_theme(theme_data):
    """Applies the theme class to the main container."""
    return theme_data.get('theme', 'light') if theme_data else 'light'



# Callback to handle file upload and store predicted data
@app.callback(
    [Output('output-data-upload', 'children'),
     Output('predicted-data-store', 'data', allow_duplicate=True), # Allow duplicate output
     Output('global-prediction-output-container', 'style')],       # Show/hide container
    Input('upload-data', 'contents'),
    State('upload-data', 'filename'),
    State('theme-store', 'data'),
    prevent_initial_call=True
)
def handle_file_upload(contents, filename, theme_data):
    """Processes the uploaded CSV file and displays predictions and visuals."""
    # # Show the prediction output container during processing
    container_style = {'display': 'block'} # Removed container_style output as Spinner needs to wrap predicted output table in app.layout

    if contents:
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        try:
            if 'csv' not in filename:
                # Hide the container if there's an error and no valid data
                return html.Div(['Invalid file type. Please upload a .csv file.'], className="error-message"), dash.no_update \
            , {'display': 'none'}

            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
            
            # Store original data for display
            original_df = df.copy()

            # Preprocessing for prediction
            # Check if encoders were loaded successfully
            if gender_encoder is None or country_encoder is None or model is None:
                return html.Div(['Error: ML model or encoders not loaded. Please check server logs.'], className="error-message"), dash.no_update \
                , {'display': 'none'}
            
            # Ensure columns exist before transforming
            # Use a try-except block for transformations to be more robust
            try:
                if 'Gender' in df.columns:
                    df['Gender'] = gender_encoder.transform(df['Gender'])
                else:
                    print("Warning: 'Gender' column not found in uploaded data. Skipping gender encoding.")
                
                if 'Country' in df.columns:
                    df['Country'] = country_encoder.transform(df['Country'])
                else:
                    print("Warning: 'Country' column not found in uploaded data. Skipping country encoding.")
            except Exception as transform_error:
                return html.Div([f"Error during data transformation: {transform_error}. Check if your CSV columns match the model's expected inputs (Gender, Country, etc.) and if your encoders are properly trained to handle the values."], className="error-message"), \
                       dash.no_update \
                , {'display': 'none'}

            
            X = df.drop(columns=['Player_ID'], errors='ignore')
            predictions = model.predict(X)
            
            # Add predictions to the original display dataframe
            original_df['Predicted_Churn'] = predictions
            
            # Generate results table
            table = dash_table.DataTable(
                columns=[{"name": i, "id": i} for i in original_df.columns],
                data=original_df.to_dict('records'),
                page_size=10,
                style_table={'overflowX': 'auto'},
                style_cell={'textAlign': 'left', 'fontFamily': 'Nunito', 'padding': '8px'},
                style_header={'backgroundColor': '#0077b6', 'color': 'white', 'fontWeight': 'bold'},
                # Add conditional styling for churn prediction rows
                style_data_conditional=[
                    {
                        'if': {'filter_query': '{Predicted_Churn} = 1'}, # If Predicted_Churn is 1 (churn)
                        'backgroundColor': '#ffe0e0', # Light red background
                        'color': 'black',
                    },
                    {
                        'if': {'filter_query': '{Predicted_Churn} = 0'}, # If Predicted_Churn is 0 (no churn)
                        'backgroundColor': '#e0ffe0', # Light green background
                        'color': 'black',
                    },
                ]
            )

            # Get the current theme from theme_data
            current_theme = theme_data.get('theme', 'light') 
            visuals = generate_visuals(original_df, current_theme)
            
            # Store dataframe in a dcc.Store for downloading
            # Serialize to JSON
            stored_data = original_df.to_json(date_format='iso', orient='split')

            return html.Div([
                html.Button("⬇ Download Predictions as CSV", id="download-btn", className="download-btn"),
                html.Hr(),
                html.H5(f"Prediction Results for: {filename}"),
                table,
                html.Hr(),
                html.H5("Visual Analytics"),
                visuals
            ]), stored_data \
        , container_style  # Removed container_style output

        except Exception as e:
            print(f"File upload/prediction error: {e}")
            return html.Div([f'An error occurred during file processing or prediction: {e}'], className="error-message"), dash.no_update \
            , {'display': 'none'} # Hide container on error
    return html.Div(), dash.no_update \
, {'display': 'none'} # Hide container if no contents


@app.callback(
    Output('download-csv', 'data'),
    Input('download-btn', 'n_clicks'),
    State('predicted-data-store', 'data'),
    prevent_initial_call=True
)
def download_predictions(n_clicks, stored_data):
    """Triggers the download of the prediction results."""
    if n_clicks and stored_data: # Ensure n_clicks is not None and stored_data exists
        print(f"Download button clicked {n_clicks} times. Preparing download...")
        df = pd.read_json(stored_data, orient='split')
        return dcc.send_data_frame(df.to_csv, "churn_predictions.csv", index=False)
    return dash.no_update

if __name__ == '__main__':
    app.run(debug=True)