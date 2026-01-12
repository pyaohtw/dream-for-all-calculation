import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="DFA vs Self Down Payment — Cashflow View", layout="wide")

# =========================
# Core idea (kept simple)
# =========================
# In this simplified comparison (ignoring mortgage amortization/interest and selling costs),
# both scenarios own the same house and have the same first mortgage payoff at sale.
# The ONLY differences are:
#   (1) DFA repayment at sale: DFA principal + share of positive appreciation
#   (2) Investment growth on the cash DFA replaces (your cash not put into down payment)
#
# Therefore:
#   DFA advantage = (InvestmentFuture) - (DFA Repayment)
# House sale price cancels out in the comparison; we show it only for intuition.

# -----------------------------
# Helper functions
# -----------------------------
def compute_future_sale_price(cost: float, home_rate: float, years: int) -> float:
    return cost * ((1.0 + home_rate) ** years)

def compute_dfa_repayment(cost: float, years: int, home_rate: float, dfa_amount: float,
                         share_factor: float, cap_to_sale_price: bool) -> dict:
    sale_price = compute_future_sale_price(cost, home_rate, years)
    appreciation = max(sale_price - cost, 0.0)
    dfa_share_pct = (dfa_amount / cost) if cost > 0 else 0.0
    appreciation_share = share_factor * dfa_share_pct * appreciation
    repayment = dfa_amount + appreciation_share
    if cap_to_sale_price:
        repayment = min(repayment, sale_price)
    return {
        "sale_price": sale_price,
        "appreciation": appreciation,
        "dfa_share_pct": dfa_share_pct,
        "appreciation_share": appreciation_share,
        "repayment": repayment
    }

def compute_investment_future_value(
    principal: float,
    years: int,
    stock_rate: float,
    use_margin: bool,
    leverage_multiple: float,
    margin_rate: float,
) -> dict:
    """
    No margin:
        future = principal*(1+stock_rate)^years
    Margin with leverage_multiple L:
        invest_amount = principal*L
        borrowed = invest_amount - principal
        future = invest_amount*(1+stock_rate)^years - borrowed*(1+margin_rate)^years

    Note: This can go negative (owing more than portfolio). This model does NOT simulate margin calls.
    """
    if principal <= 0:
        return {"invest_amount": 0.0, "borrowed": 0.0, "future": 0.0, "interest_cost": 0.0}

    if not use_margin or leverage_multiple <= 1.0:
        future = principal * ((1.0 + stock_rate) ** years)
        return {"invest_amount": principal, "borrowed": 0.0, "future": future, "interest_cost": 0.0}

    L = leverage_multiple
    invest_amount = principal * L
    borrowed = invest_amount - principal

    future_portfolio = invest_amount * ((1.0 + stock_rate) ** years)
    future_debt = borrowed * ((1.0 + margin_rate) ** years)
    future = future_portfolio - future_debt

    interest_cost = future_debt - borrowed
    return {"invest_amount": invest_amount, "borrowed": borrowed, "future": future, "interest_cost": interest_cost}

def compute_advantage(investment_future: float, dfa_repayment: float) -> float:
    # In this simplified model, all other house cashflows cancel out.
    return investment_future - dfa_repayment

def break_even_stock_return_bisect(
    cost: float, years: int, home_rate: float,
    dfa_amount: float, share_factor: float, cap_to_sale_price: bool,
    cash_invested: float,
    use_margin: bool, leverage_multiple: float, margin_rate: float,
    lo: float = -0.95, hi: float = 1.50, max_iter: int = 90
):
    if cash_invested <= 0:
        return None

    dfa = compute_dfa_repayment(cost, years, home_rate, dfa_amount, share_factor, cap_to_sale_price)
    target = dfa["repayment"]

    def f(s):
        inv = compute_investment_future_value(cash_invested, years, s, use_margin, leverage_multiple, margin_rate)["future"]
        return inv - target

    f_lo, f_hi = f(lo), f(hi)
    if f_lo == 0:
        return lo
    if f_hi == 0:
        return hi
    if f_lo * f_hi > 0:
        return None

    a, b = lo, hi
    for _ in range(max_iter):
        m = (a + b) / 2
        fm = f(m)
        if abs(fm) < 1e-6:
            return m
        if f_lo * fm <= 0:
            b, f_hi = m, fm
        else:
            a, f_lo = m, fm
    return (a + b) / 2

def break_even_home_return_bisect(
    cost: float, years: int, stock_rate: float,
    dfa_amount: float, share_factor: float, cap_to_sale_price: bool,
    cash_invested: float,
    use_margin: bool, leverage_multiple: float, margin_rate: float,
    lo: float = -0.50, hi: float = 0.25, max_iter: int = 90
):
    dfa_amount = float(dfa_amount)

    def g(h):
        dfa = compute_dfa_repayment(cost, years, h, dfa_amount, share_factor, cap_to_sale_price)
        inv = compute_investment_future_value(cash_invested, years, stock_rate, use_margin, leverage_multiple, margin_rate)["future"]
        return inv - dfa["repayment"]

    g_lo, g_hi = g(lo), g(hi)
    if g_lo == 0:
        return lo
    if g_hi == 0:
        return hi
    if g_lo * g_hi > 0:
        return None

    a, b = lo, hi
    for _ in range(max_iter):
        m = (a + b) / 2
        gm = g(m)
        if abs(gm) < 1e-6:
            return m
        if g_lo * gm <= 0:
            b, g_hi = m, gm
        else:
            a, g_lo = m, gm
    return (a + b) / 2

# -----------------------------
# UI
# -----------------------------
st.title("DFA vs Self Down Payment — Clean Cashflow Comparison")
st.caption(
    "This tool avoids the confusing idea that DFA is 'owning X% of your home'. "
    "Instead, it treats DFA as a **repayment at exit**: principal + a share of positive appreciation."
)

with st.sidebar:
    st.header("Inputs")

    purchase_cost = st.number_input(
        "All-in home cost ($)",
        min_value=1_000.0, value=1_000_000.0, step=10_000.0, format="%.0f"
    )

    st.subheader("Market assumptions (annual, decimals)")
    home_app_rate = st.number_input(
        "Home appreciation rate h (0.04 = 4%)",
        min_value=-0.20, max_value=0.25, value=0.04, step=0.005, format="%.3f"
    )
    stock_rate = st.number_input(
        "Stock return rate s (0.07 = 7%)",
        min_value=-0.95, max_value=1.50, value=0.07, step=0.005, format="%.3f"
    )
    years = st.slider("Years until sale", min_value=1, max_value=40, value=7, step=1)

    st.subheader("Down payment plan")
    dp_target_pct = st.slider(
        "Target down payment (%)",
        min_value=0.0, max_value=50.0, value=20.0, step=0.5
    ) / 100.0

    st.subheader("DFA parameters (simplified)")
    dfa_max_pct = st.slider(
        "DFA max % of home cost",
        min_value=0.0, max_value=30.0, value=20.0, step=0.5
    ) / 100.0
    dfa_cap_amount = st.number_input(
        "DFA cap ($)",
        min_value=0.0, value=150_000.0, step=5_000.0, format="%.0f"
    )
    share_factor = st.slider(
        "Share factor (1.0 standard; <1 reduced share)",
        min_value=0.0, max_value=1.5, value=1.0, step=0.05
    )

    st.subheader("Investing leverage option")
    use_margin = st.checkbox("Use margin leverage on invested cash", value=False)
    leverage_multiple = st.slider(
        "Leverage multiple (1.0 to 2.0)",
        min_value=1.0, max_value=2.0, value=1.0, step=0.05, disabled=not use_margin
    )
    margin_rate = st.number_input(
        "Margin borrowing rate (annual, decimal)",
        min_value=0.0, max_value=0.50, value=0.09, step=0.005, format="%.3f",
        disabled=not use_margin
    )

    st.subheader("Optional")
    cap_repayment_to_sale_price = st.checkbox(
        "Cap DFA repayment to sale price (avoids impossible negative proceeds)",
        value=True
    )

# -----------------------------
# Compute baseline quantities
# -----------------------------
starting_equity = purchase_cost * dp_target_pct
first_mortgage = purchase_cost - starting_equity  # simplifying assumption: same in both scenarios

dfa_amount = min(purchase_cost * dfa_max_pct, dfa_cap_amount)

cash_needed_with_dfa = max(starting_equity - dfa_amount, 0.0)
cash_invested = min(starting_equity, dfa_amount)

dfa = compute_dfa_repayment(purchase_cost, years, home_app_rate, dfa_amount, share_factor, cap_repayment_to_sale_price)
inv = compute_investment_future_value(cash_invested, years, stock_rate, use_margin, leverage_multiple, margin_rate)

dfa_adv = compute_advantage(inv["future"], dfa["repayment"])

# =========================
# 1) Rewrite UI language: no "% equity" framing
# =========================
st.subheader("Inputs summary (what you’re actually doing)")
row1, row2, row3, row4 = st.columns(4)
row1.metric("Starting equity (your down payment plan)", f"${starting_equity:,.0f}")
row2.metric("First mortgage (assumed constant)", f"${first_mortgage:,.0f}")
row3.metric("DFA amount used", f"${dfa_amount:,.0f}")
row4.metric("Years to sale", f"{years}")

row5, row6, row7 = st.columns(3)
row5.metric("Your cash into down payment (with DFA)", f"${cash_needed_with_dfa:,.0f}")
row6.metric("Cash DFA replaces (invested)", f"${cash_invested:,.0f}")
row7.metric("DFA share % (approx = DFA/home)", f"{dfa['dfa_share_pct']*100:.1f}%")

if use_margin:
    st.warning(
        "Margin leverage is risky. This model includes borrowing cost but does NOT model margin calls/forced liquidation."
    )

st.divider()

# =========================
# 2) Refactor to a single clean equation + show cash timeline
# =========================
st.subheader("Clean cashflow equation (this is the whole comparison)")

st.markdown(
    """
### DFA advantage at sale
Because both scenarios own the same house and have the same first mortgage in this simplified model, the house cashflows cancel out.

**So the only comparison is:**

**DFA advantage = (Investment value at sale) − (DFA repayment at sale)**

- If this number is **positive**, using DFA + investing the replaced cash wins.
- If **negative**, paying the down payment yourself wins.
"""
)

eq1, eq2, eq3 = st.columns(3)
eq1.metric("Investment value at sale", f"${inv['future']:,.0f}")
eq2.metric("DFA repayment at sale", f"${dfa['repayment']:,.0f}")
eq3.metric("DFA advantage", f"${dfa_adv:,.0f}")

if dfa_adv > 0:
    st.success("Result: DFA + investing wins under these assumptions.")
elif dfa_adv < 0:
    st.warning("Result: Paying the down payment yourself wins under these assumptions.")
else:
    st.info("Result: Break-even under these assumptions.")

with st.expander("Cash timeline view (makes it intuitive)"):
    # Timeline is shown in $ terms, not ownership terms.
    st.markdown("#### Time 0 (purchase)")
    st.write(f"- You have **${starting_equity:,.0f}** of down payment capital.")
    st.write(f"- With DFA, you put **${cash_needed_with_dfa:,.0f}** into the house and invest **${cash_invested:,.0f}**.")

    st.markdown(f"#### Time {years} (sale)")
    st.write(f"- Home sells for **${dfa['sale_price']:,.0f}** (modelled).")
    st.write(f"- You pay off first mortgage: **${first_mortgage:,.0f}** (assumed unchanged).")
    st.write(
        f"- DFA repayment: **${dfa_amount:,.0f} principal + ${dfa['appreciation_share']:,.0f} appreciation share = "
        f"${dfa['repayment']:,.0f}**"
    )
    if use_margin and inv["borrowed"] > 0:
        st.write(
            f"- Investment uses margin: invest ${inv['invest_amount']:,.0f} with ${inv['borrowed']:,.0f} borrowed; "
            f"interest cost over time ≈ ${inv['interest_cost']:,.0f} (compounded)."
        )
    st.write(f"- Investment value at sale: **${inv['future']:,.0f}**")

st.divider()

# =========================
# Tipping points (not just "equal rates")
# =========================
st.subheader("Tipping points (solve for the break-even rate)")

be_stock = break_even_stock_return_bisect(
    cost=purchase_cost, years=years, home_rate=home_app_rate,
    dfa_amount=dfa_amount, share_factor=share_factor, cap_to_sale_price=cap_repayment_to_sale_price,
    cash_invested=cash_invested,
    use_margin=use_margin, leverage_multiple=leverage_multiple, margin_rate=margin_rate
)
be_home = break_even_home_return_bisect(
    cost=purchase_cost, years=years, stock_rate=stock_rate,
    dfa_amount=dfa_amount, share_factor=share_factor, cap_to_sale_price=cap_repayment_to_sale_price,
    cash_invested=cash_invested,
    use_margin=use_margin, leverage_multiple=leverage_multiple, margin_rate=margin_rate
)

tp1, tp2 = st.columns(2)
with tp1:
    st.markdown("#### Break-even stock return (given home return)")
    if cash_invested <= 0:
        st.write("Not applicable: DFA is not replacing any of your down payment (cash invested = $0).")
    elif be_stock is None:
        st.write("Could not find a break-even stock return in the search range. Try different inputs.")
    else:
        st.write(f"Break-even stock return ≈ **{be_stock*100:.2f}% / year**.")
        st.write("DFA tends to be better if the **stock return is above** this (holding home return fixed).")

with tp2:
    st.markdown("#### Break-even home appreciation (given stock return)")
    if be_home is None:
        st.write("Could not find a break-even home appreciation rate in the search range. Try different inputs.")
    else:
        st.write(f"Break-even home appreciation ≈ **{be_home*100:.2f}% / year**.")
        st.write("DFA tends to be better if the **home appreciation is below** this (holding stock return fixed).")

# =========================
# Sensitivity table (years)
# =========================
st.divider()
st.subheader("Sensitivity table (selected years)")

T = np.arange(1, 41)

sale_prices = purchase_cost * ((1.0 + home_app_rate) ** T)
appreciations = np.maximum(sale_prices - purchase_cost, 0.0)

dfa_share_pct = (dfa_amount / purchase_cost) if purchase_cost > 0 else 0.0
repayments = dfa_amount + share_factor * dfa_share_pct * appreciations
if cap_repayment_to_sale_price:
    repayments = np.minimum(repayments, sale_prices)

# Investment series
inv_values = np.array([
    compute_investment_future_value(cash_invested, int(t), stock_rate, use_margin, leverage_multiple, margin_rate)["future"]
    for t in T
])

advantages = inv_values - repayments  # clean identity

df = pd.DataFrame({
    "Year": T,
    "Home Sale Price": sale_prices,
    "DFA Repayment": repayments,
    "Investment Value": inv_values,
    "DFA Advantage (Investment - Repayment)": advantages
})

st.dataframe(
    df[df["Year"].isin([1, 3, 5, 7, 10, 15, 20, 30, 40])].round(0),
    use_container_width=True
)

# =========================
# 3) Plot at the end (as requested)
# =========================
st.subheader("Plot: DFA advantage vs time to sale")

fig = plt.figure()
plt.plot(df["Year"], df["DFA Advantage (Investment - Repayment)"])
plt.axhline(0)
plt.xlabel("Years to sale")
plt.ylabel("DFA Advantage ($)")
st.pyplot(fig)

st.caption(
    "Interpretation: Above 0 means DFA + investing wins; below 0 means self down payment wins. "
    "This model does not include mortgage amortization/interest, selling costs, taxes, PMI differences, or refinance timing."
)
