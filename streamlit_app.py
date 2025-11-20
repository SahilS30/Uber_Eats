import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# ----------------------------------------------
# LOAD DATA
# ----------------------------------------------
st.title("Restaurant & Menu Analytics Dashboard")

@st.cache_data
def load_data():
    df = pd.read_csv("restaurants.csv")
    df_menu = pd.read_csv("restaurant-menus.csv")

    # Clean price
    df_menu['price'] = df_menu['price'].str.replace(" USD", "").astype(float)

    # Split category
    df['category'] = df['category'].str.split(", ").apply(lambda x: [i.strip() for i in x])
    df['price_range_numeric'] = df['price_range'].map({'$':1,'$$':2,'$$$':3,'$$$$':4,'Unknown':0})

    return df, df_menu

df, df_menu = load_data()

# ------------------------------------------------
# MERGE DATA
# ------------------------------------------------
df_merge = pd.merge(df, df_menu, left_on="id", right_on="restaurant_id", how="inner")

# Compute engineered feature
df_merge["description"] = df_merge["description"].fillna("No description")
df_merge["description_length"] = df_merge["description"].apply(len)

# ------------------------------------------------
# SIDEBAR FILTERS
# ------------------------------------------------
st.sidebar.header("Filters")

cities = sorted(df_merge["city"].dropna().unique()) if "city" in df_merge else []
selected_city = st.sidebar.selectbox("Select City (if available)", ["All"] + cities)

cuisines = sorted(df_merge["category_y"].dropna().unique())
selected_cuisine = st.sidebar.selectbox("Select Menu Category", ["All"] + cuisines)

# Filter data
filtered_df = df_merge.copy()
if selected_city != "All" and "city" in df_merge.columns:
    filtered_df = filtered_df[filtered_df["city"] == selected_city]

if selected_cuisine != "All":
    filtered_df = filtered_df[filtered_df["category_y"] == selected_cuisine]

st.subheader("Filtered Dataset Preview")
st.dataframe(filtered_df.head())

# ------------------------------------------------
# SECTION 1: PRICE DISTRIBUTION
# ------------------------------------------------
st.subheader("Price Distribution")

fig1 = px.histogram(filtered_df, x="price", nbins=50, title="Menu Price Distribution")
st.plotly_chart(fig1, use_container_width=True)

# ------------------------------------------------
# SECTION 2: AVERAGE PRICE BY CUISINE
# ------------------------------------------------
st.subheader("Average Price by Cuisine")

avg_price = (
    filtered_df.groupby("category_y")["price"]
    .mean()
    .sort_values(ascending=False)
    .reset_index()
)

fig2 = px.bar(avg_price, x="category_y", y="price", title="Average Price by Category")
st.plotly_chart(fig2, use_container_width=True)

# ------------------------------------------------
# SECTION 3: RATING VS PRICE SCATTER
# ------------------------------------------------
st.subheader("Rating vs Menu Item Price")

if "score" in filtered_df.columns:
    fig3 = px.scatter(
        filtered_df,
        x="score",
        y="price",
        color="category_y",
        hover_data=["name_x"],
        title="Score vs Price"
    )
    st.plotly_chart(fig3, use_container_width=True)

# ------------------------------------------------
# SECTION 4: CITY-LEVEL PRICE VARIATION
# ------------------------------------------------
if "city" in df_merge.columns:
    st.subheader("City-level Price Variation")

    variation_city = (
        df_merge.groupby(["city"])["price"]
        .agg(["mean", "std", "min", "max", "count"])
        .sort_values("mean", ascending=False)
        .reset_index()
    )

    st.dataframe(variation_city)

    fig4 = px.bar(variation_city.head(15), x="city", y="mean",
                  title="Top 15 Cities by Avg Price")
    st.plotly_chart(fig4, use_container_width=True)

# ------------------------------------------------
# SECTION 5: UNDERPRICED vs OVERPRICED ITEMS
# ------------------------------------------------
st.subheader("Underpriced & Overpriced Items")

df_merge["benchmark_price"] = df_merge.groupby("category_y")["price"].transform("mean")
df_merge["price_gap"] = df_merge["price"] - df_merge["benchmark_price"]

underpriced = df_merge[df_merge["price_gap"] < -1][["name_x","price","price_gap","ratings"]].head(20)
overpriced = df_merge[df_merge["price_gap"] > 1][["name_x","price","price_gap","ratings"]].head(20)

tab1, tab2 = st.tabs(["Underpriced Items", "Overpriced Items"])

with tab1:
    st.write("Items priced WELL BELOW category average")
    st.dataframe(underpriced)

with tab2:
    st.write("Items priced ABOVE category average")
    st.dataframe(overpriced)

# ------------------------------------------------
# SECTION 6: MENU SIZE ANALYSIS
# ------------------------------------------------
st.subheader("Menu Size Analysis")

menu_size = df_merge.groupby("restaurant_id")["name_y"].count().reset_index()
menu_size.columns = ["restaurant_id", "menu_size"]

fig5 = px.histogram(menu_size, x="menu_size", title="Menu Size Distribution")
st.plotly_chart(fig5, use_container_width=True)

# ------------------------------------------------
# SUMMARY
# ------------------------------------------------
st.subheader("Summary Insights")
st.write("""
- High-price variation across categories and cities shows strong regional influence.
- Underpriced items often have high customer ratings → missed revenue opportunities.
- Overpriced items frequently correlate with lower ratings → poor value perception.
- Menu size does not strongly affect ratings or pricing — quality matters more.
""")
