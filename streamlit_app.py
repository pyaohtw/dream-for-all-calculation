import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="DFA vs Self Down Payment", layout="wide")

st.title("DFA vs Paying Down Payment Yourself — Take-home Equity at Sale")
st.caption(
    "Simplified comparison: (A) pay down payment yourself vs (B) use DFA and invest the replaced cash. "
    "Mortgage, taxes, insurance, maintenance, PMI, and selling costs are excluded (assumed equal or ignored). "
    "Optional: model margin leverage on the invested cash (adds borrowing cost and amplifies losses)."
)

# -----------------------------
# Helpers
# -----------------------------
def future_home_value(cost: float, h: float, t: int) -> float:
    return cost * ((1 + h) ** t)

def dfa_repayment(cost: float, t: int, h: float, dfa_assist: float, share_factor: float, cap_to_proceeds: bool) -> float:
    fv = future_home_value(cost, h, t)
    appreciation = max(fv - cost, 0.0)
    dfa_share_pct = (dfa_assist / cost) if cost > 0 else 0.0
    repay = dfa_assist + share_factor * dfa_share_pct * appreciation
    if cap_to_proceeds:
        repay = min(repay, fv)
    return repay

def investment_future_value(
    principal: float,
    t: int,
    stock_r: float,
    use_margin: bool,
    leverage_multiple: float,
    margin_rate: float
) -> float:
    """
    If no margin:
        principal*(1+stock_r)^t
    If margin with leverage_multiple L:
        invest_amount = principal * L
        borrowed = invest_amount - principal
        end_value = invest_amount*(1+stock_r)^t - borrowed*(1+margin_rate)^t
    Note: can be negative (margin losses can exceed equity).
    """
    if principal <= 0:
        return 0.0

    if not use_margin or leverage_multiple <= 1.0:
        return principal * ((1 + stock_r) ** t)

    L = leverage_multiple
    invest_amount = principal * L
    borrowed = invest_amount - principal

    return invest_amount * ((1 + stock_r) ** t) - borrowed * ((1 + margin_rate) ** t)

def advantage(
    cost: float,
    t: int,
    h: float,
    s: float,
    dfa_assist: float,
    share_factor: float,
    cap_to_proceeds: bool,
    cash_invested: float,
    use_margin: bool,
    leverage_multiple: float,
    margin_rate: float
) -> float:
    """
    DFA advantage in this simplified model is:
        InvestmentFuture - DFA_repayment
    because the house sale price itself cancels out between scenarios.
    """
    repay = dfa_repayment(cost, t, h, dfa_assist, share_factor, cap_to_proceeds)
    inv = investment_future_value(cash_invested, t, s, use_margin, leverage_multiple, margin_rate)
    return inv - repay

def solve_break_even_stock(
    cost: float, t: int, h: float,
    dfa_assist: float, share_factor: float, cap_to_proceeds: bool,
    cash_invested: float,
    use_margin: bool, leverage_multiple: float, margin_rate: float,
    lo: float = -0.95, hi: float = 1.50, max_iter: int = 80
):
    """
    Find stock return s such that advantage(...) == 0 using bisection.
    Monotonic in s for typical ranges.
    Returns None if it cannot bracket a root.
    """
    if cash_invested <= 0:
        return None

    f_lo = advantage(cost, t, h, lo, dfa_assist, share_factor, cap_to_proceeds,
                     cash_invested, use_margin, leverage_multiple, margin_rate)
    f_hi = advantage(cost, t, h, hi, dfa_assist, share_factor, cap_to_proceeds,
                     cash_invested, use_margin, leverage_multiple, margin_rate)

    # Need opposite signs to bracket
    if f_lo == 0:
        return lo
    if f_hi == 0:
        return hi
    if f_lo * f_hi > 0:
        return None

    a, b = lo, hi
    fa, fb = f_lo, f_hi
    for _ in range(max_iter):
        m = (a + b) / 2
        fm = advantage(cost, t, h, m, dfa_assist, share_factor, cap_to_proceeds,
                       cash_invested, use_margin, leverage_multiple, margin_rate)
        if abs(fm) < 1e-6:
            return m
        # keep the bracket
        if fa * fm <= 0:
            b, fb = m, fm
        else:
            a, fa = m, fm
    return (a + b) / 2

def solve_break_even_home(
    cost: float, t: int, s: float,
    dfa_assist: float, share_factor: float, cap_to_proceeds: bool,
    cash_invested: float,
    use_margin: bool, leverage_multiple: float, margin_rate: float,
    lo: float = -0.50, hi: float = 0.25, max_iter: int = 80
):
    """
    Find home appreciation h such that advantage(...) == 0 using bisection.
    Advantage decreases as h increases (because repayment increases with appreciation).
    Returns None if it cannot bracket a root.
    """
    f_lo = advantage(cost, t, lo, s, dfa_assist, share_factor, cap_to_proceeds,
                     cash_invested, use_margin, leverage_multiple, margin_rate)
    f_hi = advantage(cost, t, hi, s, dfa_assist, share_factor, cap_to_proceeds,
                     cash_invested, use_margin, leverage_multiple, margin_rate)

    if f_lo == 0:
        return lo
    if f_hi == 0:
        return hi
    if f_lo * f_hi > 0:
        return None

    a, b = lo, hi
    fa, fb = f_lo, f_hi
    for _ in range(max_iter):
        m = (a + b) / 2
        fm = advantage(cost, t, m, s, dfa_assist, share_factor, cap_to_proceeds,
                       cash_invested, use_margin, leverage_multiple, margin_rate)
        if abs(fm) < 1e-6:
            return m
        if fa * fm <= 0:
            b, fb = m, fm
        else:
            a, fa = m, fm
    return (a + b) / 2

# -----------------------------
# Sidebar inputs (order updated)
# -----------------------------
DEFAULT_HOME_APPRECIATION = 0.04
DEFAULT_STOCK_RETURN = 0.07
DEFAULT_MARGIN_RATE = 0.09

with st.sidebar:
    st.header("Inputs")

    purchase_cost = st.number_input(
        "All-in home cost (purchase price + any acquisition costs you include)",
        min_value=1_000.0, value=1_000_000.0, step=10_000.0, format="%.0f"
    )

    st.subheader("Market assumptions (annual rates as decimals)")
    home_app_rate = st.number_input(
        "Home appreciation rate h (e.g., 0.04 = 4%)",
        min_value=-0.20, max_value=0.25, value=DEFAULT_HOME_APPRECIATION, step=0.005, format="%.3f"
    )
    stock_rate = st.number_input(
        "Stock return rate s (e.g., 0.07 = 7%)",
        min_value=-0.95, max_value=1.50, value=DEFAULT_STOCK_RETURN, step=0.005, format="%.3f"
    )

    years = st.slider("Years until sale", min_value=1, max_value=40, value=7, step=1)

    st.subheader("Your planned down payment capital")
    dp_target_pct = st.slider(
        "Target down payment (%)",
        min_value=0.0, max_value=50.0, value=20.0, step=0.5
    ) / 100.0

    st.subheader("DFA parameters (simplified)")
    dfa_max_pct = st.slider(
        "DFA max % of cost",
        min_value=0.0, max_value=30.0, value=20.0, step=0.5
    ) / 100.0

    dfa_cap_amount = st.number_input(
        "DFA cap ($)",
        min_value=0.0, value=150_000.0, step=5_000.0, format="%.0f"
    )

    share_factor = st.slider(
        "Appreciation share factor (1.0 = standard; <1.0 = reduced-share)",
        min_value=0.0, max_value=1.5, value=1.0, step=0.05
    )

    st.subheader("Leverage option: margin on invested cash")
    use_margin = st.checkbox(
        "Use margin leverage for the invested cash (higher risk)",
        value=False
    )
    leverage_multiple = st.slider(
        "Leverage multiple (1.0 = none, 2.0 ≈ 50% initial margin)",
        min_value=1.0, max_value=2.0, value=1.0, step=0.05, disabled=not use_margin
    )
    margin_rate = st.number_input(
        "Margin borrowing rate (annual, decimal)",
        min_value=0.00, max_value=0.50, value=DEFAULT_MARGIN_RATE, step=0.005, format="%.3f",
        disabled=not use_margin
    )

    st.subheader("Optional realism toggle")
    cap_repayment_to_sale_proceeds = st.checkbox(
        "Cap DFA repayment to sale proceeds",
        value=True
    )

# -----------------------------
# Core scenario calculations
# -----------------------------
starting_equity = purchase_cost * dp_target_pct

dfa_assist = min(purchase_cost * dfa_max_pct, dfa_cap_amount)
cash_to_house_with_dfa = max(starting_equity - dfa_assist, 0.0)
cash_invested = min(starting_equity, dfa_assist)

fv = future_home_value(purchase_cost, home_app_rate, years)
repay = dfa_repayment(purchase_cost, years, home_app_rate, dfa_assist, share_factor, cap_repayment_to_sale_proceeds)
sale_proceeds_self = fv
sale_proceeds_dfa = fv - repay

inv_future = investment_future_value(cash_invested, years, stock_rate, use_margin, leverage_multiple, margin_rate)

take_home_self = sale_proceeds_self
take_home_dfa = sale_proceeds_dfa + inv_future

profit_self = take_home_self - starting_equity
profit_dfa = take_home_dfa - starting_equity

dfa_advantage = profit_dfa - profit_self  # equals inv_future - repay

# -----------------------------
# Display: starting equity
# -----------------------------
st.subheader("Starting equity")
st.metric("Starting equity (your down payment capital)", f"${starting_equity:,.0f}")

c1, c2, c3 = st.columns(3)
c1.metric("DFA assistance used", f"${dfa_assist:,.0f}")
c2.metric("Cash you still put into house with DFA", f"${cash_to_house_with_dfa:,.0f}")
c3.metric("Cash invested (replaced by DFA)", f"${cash_invested:,.0f}")

if use_margin:
    st.warning(
        "Margin leverage can amplify losses and can lead to outcomes where investment value is negative "
        "(owing more than the portfolio). This simplified model includes borrowing cost but does not model margin calls."
    )

st.divider()

# -----------------------------
# Display: at sale framing
# -----------------------------
st.subheader(f"At sale after {years} years (take-home framing)")

a, b = st.columns(2)
with a:
    st.markdown("### Pay down payment yourself")
    st.metric("Sale proceeds (from house)", f"${sale_proceeds_self:,.0f}")
    st.metric("Profit vs starting equity", f"${profit_self:,.0f}")

with b:
    st.markdown("### Use DFA + invest replaced cash")
    st.metric("Sale proceeds after DFA repayment", f"${sale_proceeds_dfa:,.0f}")
    st.metric("Investment value at sale", f"${inv_future:,.0f}")
    st.metric("Total take-home at sale", f"${take_home_dfa:,.0f}")
    st.metric("Profit vs starting equity", f"${profit_dfa:,.0f}")

st.divider()

st.subheader("Decision")
st.metric("DFA advantage (Profit DFA − Profit Self)", f"${dfa_advantage:,.0f}")

if dfa_advantage > 0:
    st.success("Under these assumptions, DFA + investing the replaced cash is better.")
elif dfa_advantage < 0:
    st.warning("Under these assumptions, paying the down payment yourself is better.")
else:
    st.info("Under these assumptions, they are break-even.")

with st.expander("Why it’s NOT just 'home return = stock return'"):
    st.write(
        """
The break-even condition compares **dollars**, not just rates:

**Break-even:** InvestmentFuture = DFA Repayment

- DFA repayment depends on **DFA principal + share of *positive* home appreciation**, which depends on the home return AND how big the DFA assistance is (cap matters).
- InvestmentFuture depends on **how much cash DFA replaces**, the stock return, years held, and (optionally) margin borrowing cost.

So the tipping point is generally **not** when home return equals stock return.
        """
    )

# -----------------------------
# Tipping points
# -----------------------------
st.subheader("Tipping points (when DFA becomes better)")

be_stock = solve_break_even_stock(
    cost=purchase_cost, t=years, h=home_app_rate,
    dfa_assist=dfa_assist, share_factor=share_factor, cap_to_proceeds=cap_repayment_to_sale_proceeds,
    cash_invested=cash_invested,
    use_margin=use_margin, leverage_multiple=leverage_multiple, margin_rate=margin_rate
)

be_home = solve_break_even_home(
    cost=purchase_cost, t=years, s=stock_rate,
    dfa_assist=dfa_assist, share_factor=share_factor, cap_to_proceeds=cap_repayment_to_sale_proceeds,
    cash_invested=cash_invested,
    use_margin=use_margin, leverage_multiple=leverage_multiple, margin_rate=margin_rate
)

tp1, tp2 = st.columns(2)

with tp1:
    st.markdown("#### Required stock return (given home return)")
    if cash_invested <= 0:
        st.write("Not applicable: cash invested is $0 (DFA is not replacing any of your down payment).")
    elif be_stock is None:
        st.write("Could not find a break-even stock return in the search range. Try different inputs.")
    else:
        st.write(
            f"Break-even stock return ≈ **{be_stock*100:.2f}%/yr**. "
            f"DFA tends to be better if your stock return is **above** this (holding home return fixed)."
        )

with tp2:
    st.markdown("#### Required home appreciation (given stock return)")
    if be_home is None:
        st.write("Could not find a break-even home appreciation rate in the search range. Try different inputs.")
    else:
        st.write(
            f"Break-even home appreciation ≈ **{be_home*100:.2f}%/yr**. "
            f"DFA tends to be better if home appreciation is **below** this (holding stock return fixed)."
        )

# -----------------------------
# Sensitivity table (selected years)
# -----------------------------
st.divider()
st.subheader("Sensitivity table (selected years)")

T = np.arange(1, 41)

fv_series = purchase_cost * ((1 + home_app_rate) ** T)
app_series = np.maximum(fv_series - purchase_cost, 0.0)

dfa_share_pct = (dfa_assist / purchase_cost) if purchase_cost > 0 else 0.0
repay_series = dfa_assist + share_factor * dfa_share_pct * app_series
if cap_repayment_to_sale_proceeds:
    repay_series = np.minimum(repay_series, fv_series)

sale_dfa_series = fv_series - repay_series

# Investment series (with or without margin)
inv_series = []
for t in T:
    inv_series.append(
        investment_future_value(cash_invested, int(t), stock_rate, use_margin, leverage_multiple, margin_rate)
    )
inv_series = np.array(inv_series)

take_self_series = fv_series
take_dfa_series = sale_dfa_series + inv_series

profit_self_series = take_self_series - starting_equity
profit_dfa_series = take_dfa_series - starting_equity
adv_series = profit_dfa_series - profit_self_series  # = inv_series - repay_series

df = pd.DataFrame({
    "Year": T,
    "Self: Sale Proceeds": take_self_series,
    "DFA: Sale Proceeds After Repay": sale_dfa_series,
    "DFA: Investment Value": inv_series,
    "Self: Profit vs Start": profit_self_series,
    "DFA: Profit vs Start": profit_dfa_series,
    "DFA Advantage": adv_series
})

st.dataframe(
    df[df["Year"].isin([1, 3, 5, 7, 10, 15, 20, 30, 40])].round(0),
    use_container_width=True
)

# -----------------------------
# Plot at the very end (as requested)
# -----------------------------
st.subheader("Plot: DFA advantage vs time to sale")

fig = plt.figure()
plt.plot(df["Year"], df["DFA Advantage"])
plt.axhline(0)
plt.xlabel("Years to sale")
plt.ylabel("DFA Advantage ($)")
st.pyplot(fig)

st.caption(
    "Notes: Margin investing is risky and can trigger margin calls; this model does not simulate margin calls. "
    "To be more decision-grade, add mortgage paydown differences, selling costs, taxes, and refinance timing."
)
