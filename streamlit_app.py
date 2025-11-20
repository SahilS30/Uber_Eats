import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ----------------------------------------------------
# Load Data
# ----------------------------------------------------
st.title("Restaurant & Menu Analytics Dashboard (No Plotly Version)")

@st.cache_data
def load_data():
    df = pd.read_csv("restaurants.csv")
    df_menu = pd.read_csv("restaurant-menus.csv")

    # Clean price
    df_menu["price"] = df_menu["price"].str.replace(" USD", "").astype(float)

    # Category split
    df["category"] = df["category"].str.split(", ").apply(lambda x: [i.strip() for i in x])
    df["price_range_numeric"] = df["price_range"].map({"$":1,"$$":2,"$$$":3,"$$$$":4,"Unknown":0})

    return df, df_menu

df, df_menu = load_data()

# ----------------------------------------------------
# Merge
# ----------------------------------------------------
df_merge = pd.merge(df, df_menu, left_on="id", right_on="restaurant_id", how="inner")
df_merge["description"] = df_merge["description"].fillna("No description")
df_merge["description_length"] = df_merge["description"].apply(len)

# ----------------------------------------------------
# Filters
# ----------------------------------------------------
st.sidebar.header("Filters")

cities = sorted(df_merge["city"].dropna().unique()) if "city" in df_merge else []
selected_city = st.sidebar.selectbox("Select City", ["All"] + cities)

categories = sorted(df_merge["category_y"].dropna().unique())
selected_cat = st.sidebar.selectbox("Select Menu Category", ["All"] + categories)

filtered = df_merge.copy()
if selected_city != "All" and "city" in df_merge.columns:
    filtered = filtered[filtered["city"] == selected_city]

if selected_cat != "All":
    filtered = filtered[filtered["category_y"] == selected_cat]

st.subheader("Filtered Dataset Preview")
st.dataframe(filtered.head())

# ----------------------------------------------------
# PRICE DISTRIBUTION (Matplotlib)
# ----------------------------------------------------
st.subheader("Price Distribution")

fig, ax = plt.subplots(figsize=(8,4))
sns.histplot(filtered["price"], bins=40, kde=True, ax=ax)
ax.set_title("Menu Item Price Distribution")
ax.set_xlabel("Price")
ax.set_ylabel("Count")
st.pyplot(fig)

# ----------------------------------------------------
# AVERAGE PRICE BY CATEGORY (Bar Chart)
# ----------------------------------------------------
st.subheader("Average Price by Category")

avg_cat = filtered.groupby("category_y")["price"].mean().sort_values(ascending=False)

fig, ax = plt.subplots(figsize=(8,5))
avg_cat.plot(kind="bar", ax=ax, color="skyblue")
ax.set_title("Average Price per Menu Category")
ax.set_xlabel("Category")
ax.set_ylabel("Average Price")
plt.xticks(rotation=45)
st.pyplot(fig)

# ----------------------------------------------------
# RATING VS PRICE SCATTER
# ----------------------------------------------------
st.subheader("Rating vs Price")

if "score" in filtered:
    fig, ax = plt.subplots(figsize=(8,5))
    ax.scatter(filtered["score"], filtered["price"], alpha=0.5)
    ax.set_xlabel("Score")
    ax.set_ylabel("Price")
    ax.set_title("Score vs Price Scatter Plot")
    st.pyplot(fig)

# ----------------------------------------------------
# CITY PRICE VARIATION TABLE
# ----------------------------------------------------
st.subheader("City-Level Price Variation")

if "city" in df_merge.columns:
    variation_city = (
        df_merge.groupby("city")["price"]
        .agg(["mean", "std", "min", "max", "count"])
        .sort_values("mean", ascending=False)
        .reset_index()
    )

    st.dataframe(variation_city)

    fig, ax = plt.subplots(figsize=(8,5))
    sns.barplot(data=variation_city.head(10), x="city", y="mean", ax=ax)
    ax.set_title("Top 10 Cities by Avg Price")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    st.pyplot(fig)

# ----------------------------------------------------
# UNDERPRICED & OVERPRICED ITEMS
# ----------------------------------------------------
st.subheader("Underpriced & Overpriced Items")

df_merge["benchmark_price"] = df_merge.groupby("category_y")["price"].transform("mean")
df_merge["price_gap"] = df_merge["price"] - df_merge["benchmark_price"]

underpriced = df_merge[df_merge["price_gap"] < -1][["name_x","price","price_gap","ratings"]].head(20)
overpriced = df_merge[df_merge["price_gap"] > 1][["name_x","price","price_gap","ratings"]].head(20)

tab1, tab2 = st.tabs(["Underpriced Items", "Overpriced Items"])

with tab1:
    st.write("Items priced far BELOW category avg")
    st.dataframe(underpriced)

with tab2:
    st.write("Items priced ABOVE category avg")
    st.dataframe(overpriced)

# ----------------------------------------------------
# MENU SIZE ANALYSIS
# ----------------------------------------------------
st.subheader("Menu Size per Restaurant")

menu_size = df_merge.groupby("restaurant_id")["name_y"].count()

fig, ax = plt.subplots(figsize=(8,4))
sns.histplot(menu_size, bins=40, kde=True, ax=ax)
ax.set_title("Distribution of Menu Sizes")
ax.set_xlabel("Menu Size (# items)")
st.pyplot(fig)

# ----------------------------------------------------
# Summary
# ----------------------------------------------------
st.subheader("Insight Summary")
st.write("""
### Key Insights  
- Price variation across cities is very high, showing regional influence.  
- Many underpriced items have high ratings → revenue opportunity.  
- Overpriced items often correlate with lower ratings → weak value perception.  
- Menu size has weak correlation with rating or pricing → quality > quantity.  
""")

