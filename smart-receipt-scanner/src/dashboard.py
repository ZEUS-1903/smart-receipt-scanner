"""
dashboard.py — Streamlit dashboard for the expense tracker.

Run with: streamlit run src/dashboard.py

Displays:
- Spending overview cards
- Category breakdown (pie chart)
- Daily spending trend (line chart)
- Budget progress bars
- Recent receipts table
- Receipt upload & scanning
"""

import sys
from pathlib import Path

# Add src to path so imports work when run via streamlit
sys.path.insert(0, str(Path(__file__).parent))

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from datetime import datetime
from database import (
    get_spending_totals, get_spending_by_category, get_daily_spending,
    get_monthly_spending, get_all_receipts, get_receipt_with_items,
    get_budgets, set_budget, seed_demo_data,
)


# --- Page Config ---
st.set_page_config(
    page_title="Receipt Scanner — Expense Tracker",
    page_icon="🧾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Custom Styling ---
st.markdown("""
<style>
    .stMetric {
        background: linear-gradient(135deg, #0f1724, #1a2332);
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid rgba(0, 224, 255, 0.1);
    }
    .stMetric label { color: #6b7084 !important; }
    .stMetric [data-testid="stMetricValue"] { color: #e2e4ea !important; }
    div[data-testid="stSidebar"] { background: #0a0e1a; }
    .budget-bar {
        height: 8px;
        border-radius: 4px;
        background: #1a2332;
        margin: 4px 0 12px 0;
    }
    .budget-fill {
        height: 100%;
        border-radius: 4px;
        transition: width 0.3s ease;
    }
</style>
""", unsafe_allow_html=True)


# --- Sidebar ---
with st.sidebar:
    st.title("🧾 Receipt Scanner")
    st.caption("AI-Powered Expense Tracker")
    st.divider()

    page = st.radio(
        "Navigation",
        ["📊 Dashboard", "📸 Scan Receipt", "💰 Budgets", "📋 All Receipts"],
        label_visibility="collapsed",
    )

    st.divider()

    # Demo data button
    if st.button("🎲 Load Demo Data", use_container_width=True):
        seed_demo_data()
        st.success("Demo data loaded!")
        st.rerun()

    st.divider()
    st.caption("Powered by Donut Transformer")
    st.caption("Models run 100% locally")


# === DASHBOARD PAGE ===
if page == "📊 Dashboard":
    st.title("Expense Dashboard")
    st.caption(f"Updated: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}")

    # --- Summary Cards ---
    totals = get_spending_totals()

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Today", f"${totals['total_today']:.2f}")
    with col2:
        st.metric("This Week", f"${totals['total_this_week']:.2f}")
    with col3:
        st.metric("This Month", f"${totals['total_this_month']:.2f}")
    with col4:
        st.metric("Receipts Scanned", totals['receipt_count'])

    st.divider()

    # --- Charts Row ---
    col_left, col_right = st.columns([3, 2])

    with col_left:
        st.subheader("Daily Spending Trend")
        daily = get_daily_spending(days=30)
        if daily:
            df_daily = pd.DataFrame(daily)
            df_daily["date"] = pd.to_datetime(df_daily["date"])

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df_daily["date"],
                y=df_daily["daily_total"],
                mode="lines+markers",
                fill="tozeroy",
                line=dict(color="#00e0ff", width=2),
                fillcolor="rgba(0, 224, 255, 0.1)",
                marker=dict(size=4),
            ))
            fig.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=0, r=0, t=10, b=0),
                height=300,
                xaxis=dict(showgrid=False),
                yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)",
                          tickprefix="$"),
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No spending data yet. Scan some receipts or load demo data!")

    with col_right:
        st.subheader("By Category")
        by_cat = get_spending_by_category(days=30)
        if by_cat:
            df_cat = pd.DataFrame(by_cat)

            colors = ["#00e0ff", "#8b5cf6", "#ff2d78", "#fbbf24",
                     "#34d399", "#f97316", "#ec4899", "#6366f1"]

            fig = px.pie(
                df_cat,
                values="total_spent",
                names="category",
                color_discrete_sequence=colors,
                hole=0.5,
            )
            fig.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=0, r=0, t=10, b=0),
                height=300,
                showlegend=True,
                legend=dict(font=dict(size=10)),
            )
            fig.update_traces(textinfo="percent+label", textfont_size=10)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No category data yet.")

    st.divider()

    # --- Budget Progress ---
    st.subheader("Budget Status")
    budgets = get_budgets()
    if budgets:
        cols = st.columns(min(len(budgets), 4))
        for i, budget in enumerate(budgets):
            with cols[i % len(cols)]:
                pct = (budget["spent"] / budget["monthly_limit"] * 100
                      if budget["monthly_limit"] > 0 else 0)
                color = "#34d399" if pct < 75 else "#fbbf24" if pct < 90 else "#ff2d78"

                st.markdown(f"**{budget['category'].title()}**")
                st.markdown(
                    f"${budget['spent']:.0f} / ${budget['monthly_limit']:.0f}"
                )
                st.progress(min(pct / 100, 1.0))
                if pct >= 90:
                    st.warning(f"{pct:.0f}% used!")
    else:
        st.info("No budgets set. Go to Budgets page to set them up.")

    st.divider()

    # --- Recent Receipts ---
    st.subheader("Recent Receipts")
    receipts = get_all_receipts(limit=10)
    if receipts:
        df = pd.DataFrame(receipts)
        df["total"] = df["total"].apply(lambda x: f"${x:.2f}")
        df["is_anomaly"] = df["is_anomaly"].apply(lambda x: "⚠️" if x else "")
        st.dataframe(
            df[["date", "store_name", "total", "category", "is_anomaly"]],
            column_config={
                "date": "Date",
                "store_name": "Store",
                "total": "Total",
                "category": "Category",
                "is_anomaly": "Alert",
            },
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("No receipts yet. Start scanning!")


# === SCAN RECEIPT PAGE ===
elif page == "📸 Scan Receipt":
    st.title("Scan a Receipt")
    st.caption("Upload a receipt image and let the AI extract the data.")

    uploaded = st.file_uploader(
        "Upload receipt image",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
        help="Take a clear photo of your receipt. The AI works best with "
             "well-lit, non-crumpled receipts.",
    )

    if uploaded:
        col1, col2 = st.columns([1, 1])

        with col1:
            st.image(uploaded, caption="Uploaded Receipt", use_container_width=True)

        with col2:
            st.subheader("Extracted Data")

            # Save uploaded file temporarily
            temp_path = Path(__file__).parent.parent / "data" / f"temp_{uploaded.name}"
            temp_path.parent.mkdir(parents=True, exist_ok=True)
            temp_path.write_bytes(uploaded.getvalue())

            with st.spinner("🔍 Running Donut model inference..."):
                try:
                    from agent import ReceiptAgent
                    agent = ReceiptAgent()
                    result = agent.process_receipt(str(temp_path))

                    if "error" in result:
                        st.error(result["error"])
                    else:
                        st.success(f"Receipt processed in {result['processing_time']:.1f}s!")

                        st.markdown(f"**Store:** {result['store_name']}")
                        st.markdown(f"**Date:** {result['date']}")
                        st.markdown(f"**Category:** {result['category']} "
                                   f"({result['category_confidence']:.0%})")
                        st.markdown(f"**Total:** ${result['total']:.2f}")

                        if result["is_anomaly"]:
                            st.warning(f"⚠️ Anomaly: {result['anomaly_reason']}")

                        if result.get("budget_status"):
                            bs = result["budget_status"]
                            st.info(f"Budget: ${bs['spent_this_month']:.0f} / "
                                   f"${bs['monthly_limit']:.0f} "
                                   f"({bs['percent_used']:.0f}% used)")

                        if result["items"]:
                            st.markdown("**Items:**")
                            for item in result["items"]:
                                st.markdown(
                                    f"- {item['name']} x{item['quantity']} — "
                                    f"${item['price']:.2f}"
                                )
                except ImportError:
                    st.error(
                        "Could not load the Donut model. Make sure you've installed "
                        "all requirements: `pip install -r requirements.txt`"
                    )
                except Exception as e:
                    st.error(f"Error processing receipt: {str(e)}")
                finally:
                    # Cleanup temp file
                    if temp_path.exists():
                        temp_path.unlink()

    else:
        st.markdown("""
        ### How it works:
        1. **Upload** a receipt photo (JPG or PNG)
        2. **Donut Transformer** extracts store, items, and prices — no OCR needed
        3. **ML Categorizer** classifies the purchase type
        4. **Anomaly Detector** flags unusual spending
        5. Everything is saved to your local database

        > 💡 **Tip:** For best results, use well-lit photos with the full receipt visible.
        """)


# === BUDGETS PAGE ===
elif page == "💰 Budgets":
    st.title("Budget Management")
    st.caption("Set monthly spending limits by category.")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Set a Budget")
        categories = ["groceries", "dining", "transport", "healthcare",
                      "entertainment", "shopping", "utilities", "subscriptions"]

        cat = st.selectbox("Category", categories)
        limit = st.number_input("Monthly Limit ($)", min_value=0, value=300, step=50)

        if st.button("Save Budget", type="primary"):
            set_budget(cat, limit)
            st.success(f"Budget set: {cat} → ${limit}/month")
            st.rerun()

    with col2:
        st.subheader("Current Budgets")
        budgets = get_budgets()
        if budgets:
            for b in budgets:
                pct = (b["spent"] / b["monthly_limit"] * 100
                      if b["monthly_limit"] > 0 else 0)
                remaining = b["monthly_limit"] - b["spent"]
                emoji = "✅" if pct < 75 else "⚠️" if pct < 100 else "🚨"

                st.markdown(
                    f"{emoji} **{b['category'].title()}**: "
                    f"${b['spent']:.0f} / ${b['monthly_limit']:.0f} "
                    f"(${remaining:.0f} remaining)"
                )
                st.progress(min(pct / 100, 1.0))
        else:
            st.info("No budgets set yet. Add one on the left!")


# === ALL RECEIPTS PAGE ===
elif page == "📋 All Receipts":
    st.title("All Receipts")

    receipts = get_all_receipts(limit=100)
    if receipts:
        df = pd.DataFrame(receipts)

        # Filters
        col1, col2 = st.columns(2)
        with col1:
            cats = ["All"] + sorted(df["category"].unique().tolist())
            selected_cat = st.selectbox("Filter by Category", cats)
        with col2:
            show_anomalies = st.checkbox("Show anomalies only")

        if selected_cat != "All":
            df = df[df["category"] == selected_cat]
        if show_anomalies:
            df = df[df["is_anomaly"] == 1]

        st.markdown(f"**{len(df)} receipts found**")

        df_display = df.copy()
        df_display["total"] = df_display["total"].apply(lambda x: f"${x:.2f}")
        df_display["is_anomaly"] = df_display["is_anomaly"].apply(
            lambda x: "⚠️ Anomaly" if x else ""
        )

        st.dataframe(
            df_display[["date", "store_name", "total", "category", "is_anomaly"]],
            column_config={
                "date": "Date",
                "store_name": "Store",
                "total": "Total",
                "category": "Category",
                "is_anomaly": "Status",
            },
            use_container_width=True,
            hide_index=True,
        )

        # Summary stats
        totals_df = pd.DataFrame(receipts)
        st.divider()
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Spent", f"${totals_df['total'].sum():.2f}")
        with col2:
            st.metric("Average Receipt", f"${totals_df['total'].mean():.2f}")
        with col3:
            st.metric("Highest Receipt", f"${totals_df['total'].max():.2f}")
    else:
        st.info("No receipts yet. Scan one or load demo data from the sidebar!")
