import pandas
import streamlit
import plotly.graph_objs


# Load the data from a CSV file
df = pandas.read_csv("data.csv")
df.ffill()

streamlit.set_page_config(page_title="Energy Metrics", page_icon="⚡", layout="wide")
streamlit.title("Country-wise Energy Metrics Comparison")

# Get the unique country names for the dropdown
countries = df["Country"].unique()

# Multi-select dropdown for countries
selected_countries = streamlit.multiselect(
    "Select countries:", countries, default=["China"]
)

# Create Plotly figure
fig = plotly.graph_objs.Figure()

# Loop through each selected country and add traces
for country in selected_countries:
    country_data = df[df["Country"] == country]

    fig.add_trace(
        plotly.graph_objs.Scatter(
            x=country_data["Year"],
            y=country_data["Access to electricity (% of population)"],
            mode="lines+markers",
            name=f"{country} - Access to electricity (% of population)",
            visible="legendonly"
        )
    )

    fig.add_trace(
        plotly.graph_objs.Scatter(
            x=country_data["Year"],
            y=country_data["Access to clean fuels for cooking"],
            mode="lines+markers",
            name=f"{country} - Access to clean fuels for cooking",
            visible="legendonly"
        )
    )

    fig.add_trace(
        plotly.graph_objs.Scatter(
            x=country_data["Year"],
            y=country_data["Renewable-electricity-generating-capacity-per-capita"],
            mode="lines+markers",
            name=f"{country} - Renewable electricity generating capacity per capita",
            visible="legendonly"
        )
    )

    fig.add_trace(
        plotly.graph_objs.Scatter(
            x=country_data["Year"],
            y=country_data["Primary energy consumption per capita (kWh/person)"],
            mode="lines+markers",
            name=f"{country} - Primary energy consumption per capita (kWh/person)",
        )
    )

# Plotly layout
fig.update_layout(
    title="Country-wise Energy Metrics Comparison",
    xaxis_title="Year",
    yaxis_title="Percentage / Capacity (kW)",
    legend_title="Metrics",
    template="plotly_dark",
    plot_bgcolor="#0d1017",
    paper_bgcolor="#0d1017",
    font=dict(
        family="Iosevka",
        size=14,
        color="#bfbdb6",
    ),
    height=600,
    width=1800,
)

streamlit.plotly_chart(fig)
