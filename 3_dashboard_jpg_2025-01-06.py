import dash
from dash import dcc, html, Input, Output
import plotly.express as px
import pandas as pd

# Load the dataset
data_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00519/heart_failure_clinical_records_dataset.csv"
df = pd.read_csv(data_url)

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

# Define app layout
app.layout = html.Div([
    html.H1("Heart Failure Clinical Records Dashboard", style={"textAlign": "center", "fontFamily": "Roboto, sans-serif", "color": "#003366", "marginBottom": "20px", "fontSize": "28px"}),

    html.Div([
        html.Div([
            html.Label("Select a Feature for Distribution:", style={"fontWeight": "bold", "fontFamily": "Roboto, sans-serif", "fontSize": "12px"}),
            dcc.Dropdown(
                id="feature-dropdown",
                options=[{"label": col, "value": col} for col in df.columns if col in ["Age", "Creatine Phosphokinase", "Ejection Fraction", "Platelets", "Serum Creatinine", "Serum Sodium", "Follow-up Period"]],
                value="Age",
                clearable=False,
                style={"width": "100%", "fontFamily": "Roboto, sans-serif"}
            ),
            dcc.Graph(id="distribution-plot", style={"height": "300px"})
        ], className="card", style={"flex": "1", "margin": "5px"}),

        html.Div([
            html.Label("Survival Rate and Categorical Breakdown:", style={"fontWeight": "bold", "fontFamily": "Roboto, sans-serif", "fontSize": "12px"}),
            dcc.Dropdown(
                id="combined-pie-chart-dropdown",
                options=[{"label": col, "value": col} for col in df.columns if col in ["Anaemia", "Diabetes", "High Blood Pressure", "Sex", "Smoking"]],
                value="Sex",
                clearable=False,
                style={"marginBottom": "10px", "fontFamily": "Roboto, sans-serif"}
            ),
            dcc.Graph(id="combined-pie-chart", style={"height": "300px"})
        ], className="card", style={"flex": "1", "margin": "5px"}),

        html.Div([
            html.Label("Select X-axis and Y-axis for Scatter Plot:", style={"fontWeight": "bold", "fontFamily": "Roboto, sans-serif", "fontSize": "12px"}),
            dcc.Dropdown(
                id="x-axis-dropdown",
                options=[{"label": col, "value": col} for col in df.columns if col in ["Age", "Creatine Phosphokinase", "Ejection Fraction", "Platelets", "Serum Creatinine", "Serum Sodium", "Follow-up Period"]],
                value="Age",
                clearable=False,
                style={"marginBottom": "10px", "fontFamily": "Roboto, sans-serif"}
            ),
            dcc.Dropdown(
                id="y-axis-dropdown",
                options=[{"label": col, "value": col} for col in df.columns if col in ["Age", "Creatine Phosphokinase", "Ejection Fraction", "Platelets", "Serum Creatinine", "Serum Sodium", "Follow-up Period"]],
                value="Serum Creatinine",
                clearable=False,
                style={"fontFamily": "Roboto, sans-serif"}
            ),
            dcc.Graph(id="scatter-plot", style={"height": "300px"})
        ], className="card", style={"flex": "1", "margin": "5px"}),

    ], className="dashboard", style={"display": "flex", "flexWrap": "wrap", "justifyContent": "space-between"}),

], style={"backgroundColor": "#f4f4f9", "fontFamily": "Roboto, sans-serif"})

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

# Run the app
if __name__ == "__main__":
    app.run_server(debug=True)

