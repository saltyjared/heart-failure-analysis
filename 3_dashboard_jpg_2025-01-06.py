import dash
from dash import dcc, html, Input, Output
import plotly.express as px
import pandas as pd
from dash import dash_table
import joblib

# Load the dataset
data_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00519/heart_failure_clinical_records_dataset.csv"
df = pd.read_csv(data_url)

# Load the model
model_path = 'heart_failure_predictor.pkl'
model = joblib.load(model_path)

# Clean column names
df.rename(columns={
    "age": "Age",
    "anaemia": "Anaemia",
    "creatinine_phosphokinase": "Creatine Phosphokinase",
    "diabetes": "Diabetes",
    "ejection_fraction": "Ejection Fraction",
    "high_blood_pressure": "High Blood Pressure",
    "platelets": "Platelets",
    "serum_creatinine": "Serum Creatinine",
    "serum_sodium": "Serum Sodium",
    "sex": "Sex",
    "smoking": "Smoking",
    "time": "Follow-up Period",
    "DEATH_EVENT": "Death Event"
}, inplace=True)

# Initialize the Dash app
app = dash.Dash(__name__)
app.title = "Heart Failure Dashboard"

# Calculate mean values grouped by Death Event
mean_values = df.groupby("Death Event")[["Age", "Creatine Phosphokinase", "Ejection Fraction", 
                                         "Platelets", "Serum Creatinine", "Serum Sodium", 
                                         "Follow-up Period"]].mean().round(2).reset_index()
mean_values.rename(columns={"Death Event": "Death Event (0 = Surviving, 1 = Died)"}, inplace=True)

# Calculate min and max for continuous variables
min_max_values = df[["Age", "Creatine Phosphokinase", "Ejection Fraction", 
                     "Platelets", "Serum Creatinine", "Serum Sodium", 
                     "Follow-up Period"]].agg(["min", "max"]).to_dict()

# Universal styling
universal_style = {
    "fontFamily": "Roboto, sans-serif",
    "color": "#003366",
    "backgroundColor": "#f4f4f9",
    "padding": "10px",
    "border": "1px solid #ddd",
    "borderRadius": "5px"
}

app.layout = html.Div([
    html.Header([
        html.H1("Heart Failure Analysis", style={"textAlign": "center", "marginBottom": "20px"}),
        html.Div([
            html.A("EDA", href="/assets/1_eda_jpg_2024-12-13.html", target="_blank", style={"marginRight": "20px", "textDecoration": "underline", "color": "#0066cc"}),
            html.A("Modeling", href="/assets/2_modeling_jpg_2024-12-17.html", target="_blank", style={"textDecoration": "underline", "color": "#0066cc"})
        ], style={"textAlign": "center", "marginBottom": "20px"})
    ]),

    # Mean values table
    html.Div([
        html.Label("Mean Values of Continuous Variables by Death Event:", style={"fontWeight": "bold", "fontSize": "14px", "marginBottom": "10px"}),
        dash_table.DataTable(
            id="mean-table",
            columns=[{"name": col, "id": col} for col in mean_values.columns],
            data=mean_values.to_dict("records"),
            style_table={"width": "100%", "overflowX": "auto"},
            style_cell={"textAlign": "center", "padding": "5px"},
            style_header={"backgroundColor": "#003366", "color": "white", "fontWeight": "bold"},
            style_data={"backgroundColor": "#f4f4f9", "color": "#333"}
        )
    ], style={"margin": "20px"}),

    # Distribution plot and pie chart
    html.Div([
        html.Div([
            html.Label("Select a Feature for Distribution:", style={"fontWeight": "bold"}),
            dcc.Dropdown(
                id="feature-dropdown",
                options=[{"label": col, "value": col} for col in df.columns if col in ["Age", "Creatine Phosphokinase", "Ejection Fraction", "Platelets", "Serum Creatinine", "Serum Sodium", "Follow-up Period"]],
                value="Age",
                clearable=False
            ),
            dcc.Graph(id="distribution-plot", style={'width': '95%', 'height': 'auto'})
        ], style={**universal_style, 'flex': 1, 'padding': '10px'}),
        html.Div([
            html.Label("Survival Rate and Categorical Breakdown:", style={"fontWeight": "bold"}),
            dcc.Dropdown(
                id="combined-pie-chart-dropdown",
                options=[{"label": col, "value": col} for col in df.columns if col in ["Anaemia", "Diabetes", "High Blood Pressure", "Sex", "Smoking"]],
                value="Sex",
                clearable=False
            ),
            dcc.Graph(id="combined-pie-chart", style={'width': '95%', 'height': 'auto'})
        ], style={**universal_style, 'flex': 1, 'padding': '10px'}),
    ], style={"display": "flex", "gap": "20px"}),

    # Scatter plot on its own row
    html.Div([
        html.Label("Select X-axis and Y-axis for Scatter Plot:", style={"fontWeight": "bold"}),
        dcc.Dropdown(
            id="x-axis-dropdown",
            options=[{"label": col, "value": col} for col in df.columns if col in ["Age", "Creatine Phosphokinase", "Ejection Fraction", "Platelets", "Serum Creatinine", "Serum Sodium", "Follow-up Period"]],
            value="Age",
            clearable=False,
            style={"marginBottom": "10px"}
        ),
        dcc.Dropdown(
            id="y-axis-dropdown",
            options=[{"label": col, "value": col} for col in df.columns if col in ["Age", "Creatine Phosphokinase", "Ejection Fraction", "Platelets", "Serum Creatinine", "Serum Sodium", "Follow-up Period"]],
            value="Serum Creatinine",
            clearable=False
        ),
        dcc.Graph(id="scatter-plot")
    ], style={**universal_style, "marginTop": "20px"}),

    # Predictor portion
    html.Div([
        html.H2("Predict Patient Outcome", style={"fontWeight": "bold"}),

        html.Div([
            html.Label("Enter Inputs for Prediction:", style={"fontWeight": "bold"}),
            html.Div([
                html.Label("Age:"),
                dcc.Input(id="age-input", type="number", placeholder="Enter Age", min=min_max_values["Age"]["min"], max=min_max_values["Age"]["max"]),

                html.Label("Creatine Phosphokinase:"),
                dcc.Input(id="cpk-input", type="number", placeholder="Enter Creatine Phosphokinase", min=min_max_values["Creatine Phosphokinase"]["min"], max=min_max_values["Creatine Phosphokinase"]["max"]),

                html.Label("Ejection Fraction:"),
                dcc.Input(id="ef-input", type="number", placeholder="Enter Ejection Fraction", min=min_max_values["Ejection Fraction"]["min"], max=min_max_values["Ejection Fraction"]["max"]),

                html.Label("Platelets:"),
                dcc.Input(id="platelets-input", type="number", placeholder="Enter Platelets", min=min_max_values["Platelets"]["min"], max=min_max_values["Platelets"]["max"]),

                html.Label("Serum Creatinine:"),
                dcc.Input(id="serum-creatinine-input", type="number", placeholder="Enter Serum Creatinine", min=min_max_values["Serum Creatinine"]["min"], max=min_max_values["Serum Creatinine"]["max"]),

                html.Label("Serum Sodium:"),
                dcc.Input(id="serum-sodium-input", type="number", placeholder="Enter Serum Sodium", min=min_max_values["Serum Sodium"]["min"], max=min_max_values["Serum Sodium"]["max"]),

                html.Label("Follow-up Period:"),
                dcc.Input(id="time-input", type="number", placeholder="Enter Follow-up Period", min=min_max_values["Follow-up Period"]["min"], max=min_max_values["Follow-up Period"]["max"]),

                html.Label("Anaemia:"),
                dcc.Dropdown(
                    id="anaemia-input",
                    options=[{"label": "No", "value": 0}, {"label": "Yes", "value": 1}],
                    placeholder="Select Anaemia Status"
                ),

                html.Label("Diabetes:"),
                dcc.Dropdown(
                    id="diabetes-input",
                    options=[{"label": "No", "value": 0}, {"label": "Yes", "value": 1}],
                    placeholder="Select Diabetes Status"
                ),

                html.Label("High Blood Pressure:"),
                dcc.Dropdown(
                    id="high-bp-input",
                    options=[{"label": "No", "value": 0}, {"label": "Yes", "value": 1}],
                    placeholder="Select High Blood Pressure Status"
                ),

                html.Label("Sex:"),
                dcc.Dropdown(
                    id="sex-input",
                    options=[{"label": "Female", "value": 0}, {"label": "Male", "value": 1}],
                    placeholder="Select Sex"
                ),

                html.Label("Smoking:"),
                dcc.Dropdown(
                    id="smoking-input",
                    options=[{"label": "No", "value": 0}, {"label": "Yes", "value": 1}],
                    placeholder="Select Smoking Status"
                )
                        ], style={"display": "flex", "flexWrap": "wrap", "flexDirection": "row", "gap": "10px"}),

                        html.Button("Predict", id="predict-button"),
                        html.Div(id="prediction-output", style={"marginTop": "10px"})
                    ], style=universal_style)
    ], style={"marginTop": "20px"})
])

# Callbacks
@app.callback(
    Output("distribution-plot", "figure"),
    Input("feature-dropdown", "value")
)
def update_distribution(selected_feature):
    fig = px.histogram(
        df, x=selected_feature, nbins=11, color="Death Event", title=f"Distribution of {selected_feature}",
        histnorm="density", labels={selected_feature: selected_feature}
    )
    fig.update_layout(bargap=0.1)
    fig.update_yaxes(title_text="Density")
    return fig

@app.callback(
    Output("scatter-plot", "figure"),
    [Input("x-axis-dropdown", "value"), Input("y-axis-dropdown", "value")]
)
def update_scatter(x_feature, y_feature):
    fig = px.scatter(
        df, x=x_feature, y=y_feature, color="Death Event",
        title=f"Scatter Plot of {x_feature} vs {y_feature}",
        labels={x_feature: x_feature, y_feature: y_feature}
    )
    return fig

@app.callback(
    Output("combined-pie-chart", "figure"),
    Input("combined-pie-chart-dropdown", "value")
)
def update_combined_pie_chart(selected_category):
    temp_df = df.copy()
    temp_df["Category"] = temp_df.apply(
        lambda row: f"Death with {selected_category}" if row["Death Event"] == 1 and row[selected_category] == 1 else
                    f"No Death with {selected_category}" if row["Death Event"] == 0 and row[selected_category] == 1 else
                    f"Death without {selected_category}" if row["Death Event"] == 1 and row[selected_category] == 0 else
                    f"No Death without {selected_category}", axis=1
    )
    fig = px.pie(
        temp_df, names="Category", title=f"Breakdown of {selected_category} by Survival",
    )
    return fig

@app.callback(
    Output("prediction-output", "children"),
    Input("predict-button", "n_clicks"),
    [Input("age-input", "value"), Input("cpk-input", "value"), Input("ef-input", "value"),
     Input("platelets-input", "value"), Input("serum-creatinine-input", "value"), Input("serum-sodium-input", "value"),
     Input("time-input", "value"), Input("anaemia-input", "value"), Input("diabetes-input", "value"),
     Input("high-bp-input", "value"), Input("sex-input", "value"), Input("smoking-input", "value")]
)
def predict_outcome(n_clicks, age, cpk, ef, platelets, sc, ss, time, anaemia, diabetes, hbp, sex, smoking):
    if n_clicks and n_clicks > 0:
        if None in [age, cpk, ef, platelets, sc, ss, time, anaemia, diabetes, hbp, sex, smoking]:
            return "Please fill out all fields before predicting."
        
        # Create input array
        input_features = [[age, anaemia, cpk, diabetes, ef, hbp, platelets, sc, ss, sex, smoking, time]]
        
        # Make prediction
        prediction = model.predict(input_features)[0]
        return f"The predicted outcome is: {'Death' if prediction == 1 else 'Survival'}."
    return ""

# Run the app
if __name__ == "__main__":
    app.run_server(debug=False)

