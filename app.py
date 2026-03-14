import dash
from dash import dcc, html
import pandas as pd
import plotly.express as px
import joblib

df = pd.read_csv("simulated_hospital_equipment_data.csv")
model = joblib.load("maintenance_regression_model.pkl")
scaler = joblib.load("feature_scaler.pkl")

df['last_maintenance_date'] = pd.to_datetime(df['last_maintenance_date'])
df['days_since_maintenance'] = (pd.Timestamp.now() - df['last_maintenance_date']).dt.days

features = df[['equipment_type', 'usage_hours_per_day', 'hospital_section', 'days_since_maintenance']]
df['predicted_days_until_failure'] = model.predict(scaler.transform(features))

app = dash.Dash(__name__)
server = app.server

fig1 = px.scatter(df, x="usage_hours_per_day", y="predicted_days_until_failure", color="equipment_type",
                  title="Predicted Days Until Failure vs. Usage Hours")

fig2 = px.bar(df.groupby("equipment_type")["predicted_days_until_failure"].mean().reset_index(),
              x="equipment_type", y="predicted_days_until_failure",
              title="Average Predicted Days Until Failure by Equipment Type")

app.layout = html.Div(children=[
    html.H1("🔧 Predictive Maintenance Dashboard"),
    dcc.Graph(figure=fig1),
    dcc.Graph(figure=fig2)
])

if __name__ == "__main__":
    app.run_server(debug=True)
