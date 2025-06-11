import dash
from dash import dcc, html, Input, Output, State, dash_table
import pandas as pd
import joblib
import plotly.express as px
import numpy as np
import base64
import io


# Load model and trained encoders
model = joblib.load('model/churn_model.pkl')
gender_encoder = joblib.load('model/gender_encoder.pkl')
country_encoder = joblib.load('model/country_encoder.pkl')

# # Define app
# app = dash.Dash(__name__)
# server = app.server  # for deployment
# app.title = 'Lottery Churn Prediction App'

# App Layout
# app.layout = html.Div([
#     html.H2("🎲 Online Lottery Player Churn Prediction"),
    
#     dcc.Upload(
#         id='upload-data',
#         children=html.Div(['📤 Drag & Drop or Click to Upload CSV File']),
#         style={
#             'width': '100%',
#             'height': '60px',
#             'lineHeight': '60px',
#             'borderWidth': '2px',
#             'borderStyle': 'dashed',
#             'borderRadius': '5px',
#             'textAlign': 'center',
#             'margin': '10px'
#         },
#         multiple=False
#     ),

#     html.Div(id='output-data-upload'),

#     dcc.Graph(id='churn-graph'),

#     html.Hr(),
#     html.A("📥 Download Template", href='/assets/sample_input.csv', download='sample_input.csv')
# ])

# app.layout = html.Div([
#     html.Div([
#         html.H2("🎲 Online Lottery Player Churn Prediction", style={'textAlign': 'center'}),

#         dcc.Upload(
#             id='upload-data',
#             children=html.Div(['📤 Drag & Drop or Click to Upload CSV File']),
#             style={
#                 'width': '100%',
#                 'height': '60px',
#                 'lineHeight': '60px',
#                 'borderWidth': '2px',
#                 'borderStyle': 'dashed',
#                 'borderRadius': '10px',
#                 'textAlign': 'center',
#                 'margin': '10px auto',
#                 'backgroundColor': '#f9f9f9',
#                 'color': '#444'
#             },
#             multiple=False
#         ),
#     ], style={'maxWidth': '800px', 'margin': 'auto'}),

#     html.Div(id='output-data-upload', style={'margin': '20px auto', 'maxWidth': '1000px'}),

#     dcc.Graph(id='churn-graph', style={'marginTop': '40px'}),

#     html.Hr(),
#     html.Div([
#         html.A("📥 Download Prediction Template (CSV)", href='/assets/sample_input.csv', download='sample_input.csv',
#                style={'fontSize': '18px', 'color': '#007BFF'})
#     ], style={'textAlign': 'center'})
# ])

# app.layout = html.Div([
#     html.Div(className="navbar", children=[
#         html.Img(src='/assets/logo.png') if 'logo.png' else '',
#         html.H2("🎲 Online Lottery Churn Prediction")
#     ]),

#     html.Div([
#         dcc.Upload(
#             id='upload-data',
#             children=html.Div(['📤 Drag & Drop or Click to Upload CSV File']),
#             className='upload-box',
#             multiple=False
#         ),
        
#         dcc.Loading(
#             id="loading-output",
#             type="circle",
#             children=[
#                 html.Div(id='output-data-upload'),
#                 dcc.Graph(id='churn-graph')
#             ]
#         ),

#         html.A("📥 Download Template CSV", href='/assets/sample_input.csv', download='sample_input.csv')
#     ]),

#     html.Footer("© 2025 Lottery Analytics App. Built by Kenneth")
# ])

# Define app
app = dash.Dash(__name__, suppress_callback_exceptions=True)
server = app.server

# Page Layouts
landing_page = html.Div([
    html.Div(className="navbar", children=[
        html.Img(src='/assets/logo.png') if 'logo.png' else '',
        html.H2("🎲 Lottery Churn Prediction")
    ]),
    html.Div([
        html.H3("Welcome to the Churn Prediction App"),
        html.P("This app helps you predict whether lottery players are likely to churn based on uploaded customer data. You can navigate between the tabs above to upload your CSV file, see predictions, and read more about the model."),
        html.P("To begin, head to the 'Upload & Predict' tab.")
    ], style={'padding': '20px'}),
    html.Footer("© 2025 Lottery Analytics App. Built by Kenneth")
])

upload_page = html.Div([
    html.H3("Upload Data & View Predictions"),
    dcc.Upload(
        id='upload-data',
        children=html.Div(['📤 Drag & Drop or Click to Upload CSV File']),
        className='upload-box',
        multiple=False
    ),
    dcc.Loading(
        id="loading-output",
        type="circle",
        children=[
            html.Div(id='output-data-upload'),
            dcc.Graph(id='churn-graph')
        ]
    ),
    html.A("📥 Download Template CSV", href='/assets/sample_input.csv', download='sample_input.csv'),
    html.Footer("© 2025 Lottery Analytics App. Built by Kenneth")
])

about_page = html.Div([
    html.H3("About This App"),
    html.P("This churn prediction model is trained using real-world anonymized data from lottery player behavior. The model uses features like demographics, transaction history, and gaming behavior to predict churn likelihood."),
    html.P("The predictions help marketing and CRM teams design better re-engagement campaigns for players at risk of churning."),
    html.Footer("Built using Dash and Plotly • Model: Random Forest Classifier • Maintained by Kenneth")
])

# Main App Layout with Tabs
app.layout = html.Div([
    html.Div(className="navbar", children=[
        html.Img(src='/assets/logo.png') if 'logo.png' else '',
        html.H2("🎲 Lottery Churn Prediction")
    ]),

    dcc.Tabs(id='tabs', value='tab-home', children=[
        dcc.Tab(label='🏠 Home', value='tab-home'),
        dcc.Tab(label='📤 Upload & Predict', value='tab-upload'),
        dcc.Tab(label='ℹ️ About', value='tab-about')
    ]),

    html.Div(id='tabs-content')
])

@app.callback(Output('tabs-content', 'children'), [Input('tabs', 'value')])
def render_content(tab):
    if tab == 'tab-upload':
        return upload_page
    elif tab == 'tab-about':
        return about_page
    return landing_page

# Callback
@app.callback(
    [Output('output-data-upload', 'children'),
     Output('churn-graph', 'figure')],
    [Input('upload-data', 'contents')],
    [State('upload-data', 'filename')]
)
def update_output(contents, filename):
    if contents is not None:
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))

        try:
            # Transform categorical columns using previously saved encoders
            df['Gender'] = gender_encoder.transform(df['Gender'])
            df['Country'] = country_encoder.transform(df['Country'])
        except Exception as e:
            return html.Div(f"⚠️ Error encoding categories: {str(e)}"), {}

        # Drop non-feature columns if needed
        X = df.drop(columns=['Player_ID'], errors='ignore')

        # Predict
        df['Predicted_Churn'] = model.predict(X)

        # Table output
        table = dash_table.DataTable(
            columns=[{"name": i, "id": i} for i in df.columns],
            data=df.to_dict('records'),
            style_table={'overflowX': 'scroll'},
            style_cell={'textAlign': 'left', 'padding': '5px', 'minWidth': '120px'},
            style_header={'backgroundColor': 'lightgrey', 'color': 'white', 'fontWeight': 'bold'}
        )

        # Visualization
        fig = px.histogram(df, x='Predicted_Churn', title='Churn Prediction Distribution', 
                           labels={'Predicted_Churn': 'Churn Prediction'},
                        #    color='Predicted_Churn',
                        color_discrete_map={0: '#2ecc71', 1: '#e74c3c'}, 
                           barmode='group')

        fig.update_layout(plot_bgcolor='#f9f9f9', paper_bgcolor='white')

        return table, fig

    return html.Div(), {}

# # Run local server
# if __name__ == '__main__':
#     app.run(debug=True)
# app.run(host='0.0.0.0', port=8080)
# For Render server deployment
server = app.server  # Expose Flask server for Gunicorn

if __name__ == '__main__':
    app.run(debug=True)

