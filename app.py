# -*- coding: utf-8 -*-
import streamlit as st
import numpy as np
import time
import matplotlib.pyplot as plt

# ============================================================
# إعداد واجهة Streamlit (الديكور)
# ============================================================
st.set_page_config(
    page_title="AHRH Solver",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

page_bg = """
<style>
[data-testid="stAppViewContainer"] {
    background-image: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
    background-attachment: fixed;
    color: white;
}
[data-testid="stHeader"] {
    background: rgba(0,0,0,0);
}
[data-testid="stSidebar"] {
    background: rgba(255,255,255,0.1);
}
h1, h2, h3, h4 {
    color: #00e6e6;
    text-shadow: 0px 0px 10px rgba(0,230,230,0.5);
}
.stButton>button {
    background-color: #00e6e6;
    color: black;
    border-radius: 10px;
    font-weight: bold;
    transition: 0.3s;
}
.stButton>button:hover {
    background-color: #00b3b3;
    color: white;
}
</style>
"""
st.markdown(page_bg, unsafe_allow_html=True)

st.markdown("<h1 style='text-align:center;'>⚙️ خوارزمية AHRH</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align:center;'>بحث محلي متقدم بأنماط تبادل متعددة</h3>", unsafe_allow_html=True)
st.markdown("---")

# ============================================================
# البحث المحلي المتقدم (1-1, 2-1, 1-2, 2-2)
# ============================================================
def solve_lp_fixed_y_uflp(y_int, f, c):
    open_fac = np.where(y_int > 0.5)[0]
    if len(open_fac) == 0:
        return float('inf')
    total = np.sum(f[open_fac])
    for j in range(c.shape[1]):
        total += min(c[i, j] for i in open_fac)
    return total

def local_search_advanced_uflp(y, best_cost, f, c, max_iter=10, verbose=True):
    n = len(f)
    improved = True
    iteration = 0
    best_y = y.copy()
    best = best_cost
    start_time = time.time()
    log = []

    while improved and iteration < max_iter:
        improved = False
        open_fac = np.where(best_y > 0.5)[0].tolist()
        closed_fac = np.where(best_y < 0.5)[0].tolist()

        # 1-1 exchange
        for i in open_fac:
            for j in closed_fac:
                y_new = best_y.copy()
                y_new[i] = 0
                y_new[j] = 1
                cost = solve_lp_fixed_y_uflp(y_new, f, c)
                if cost < best:
                    best = cost
                    best_y = y_new
                    improved = True
                    break
            if improved:
                break

        # 2-1 exchange
        if not improved and len(open_fac) >= 2 and len(closed_fac) >= 1:
            for i1 in range(len(open_fac)):
                for i2 in range(i1+1, len(open_fac)):
                    for j in closed_fac:
                        y_new = best_y.copy()
                        y_new[open_fac[i1]] = 0
                        y_new[open_fac[i2]] = 0
                        y_new[j] = 1
                        cost = solve_lp_fixed_y_uflp(y_new, f, c)
                        if cost < best:
                            best = cost
                            best_y = y_new
                            improved = True
                            break
                    if improved: break
                if improved: break

        # 1-2 exchange
        if not improved and len(open_fac) >= 1 and len(closed_fac) >= 2:
            for i in open_fac:
                for j1 in range(len(closed_fac)):
                    for j2 in range(j1+1, len(closed_fac)):
                        y_new = best_y.copy()
                        y_new[i] = 0
                        y_new[closed_fac[j1]] = 1
                        y_new[closed_fac[j2]] = 1
                        cost = solve_lp_fixed_y_uflp(y_new, f, c)
                        if cost < best:
                            best = cost
                            best_y = y_new
                            improved = True
                            break
                    if improved: break
                if improved: break

        # 2-2 exchange
        if not improved and len(open_fac) >= 2 and len(closed_fac) >= 2:
            for i1 in range(len(open_fac)):
                for i2 in range(i1+1, len(open_fac)):
                    for j1 in range(len(closed_fac)):
                        for j2 in range(j1+1, len(closed_fac)):
                            y_new = best_y.copy()
                            y_new[open_fac[i1]] = 0
                            y_new[open_fac[i2]] = 0
                            y_new[closed_fac[j1]] = 1
                            y_new[closed_fac[j2]] = 1
                            cost = solve_lp_fixed_y_uflp(y_new, f, c)
                            if cost < best:
                                best = cost
                                best_y = y_new
                                improved = True
                                break
                        if improved: break
                    if improved: break
                if improved: break

        iteration += 1
        log.append((iteration, best))
        if verbose:
            st.write(f"🔁 Iteration {iteration}: Best cost = {best:,.2f}")

    runtime = time.time() - start_time
    return best, best_y, log, runtime

# ============================================================
# واجهة المستخدم
# ============================================================
st.sidebar.title("إعدادات البحث المحلي")
max_iter = st.sidebar.slider("عدد التكرارات القصوى", 1, 50, 10)
n_fac = st.sidebar.slider("عدد المرافق", 3, 20, 6)
n_clients = st.sidebar.slider("عدد العملاء", 3, 20, 6)

st.markdown("### 📂 بيانات عشوائية لتجربة البحث المحلي")
if st.button("🚀 تشغيل البحث المحلي"):
    np.random.seed(42)
    f = np.random.randint(10, 50, size=n_fac)
    c = np.random.randint(5, 30, size=(n_fac, n_clients))
    y0 = np.random.randint(0, 2, size=n_fac)
    best_cost = solve_lp_fixed_y_uflp(y0, f, c)

    st.write("🔹 **التكلفة الابتدائية:**", best_cost)
    st.write("🔹 **المرافق المفتوحة:**", np.where(y0 > 0.5)[0].tolist())

    best, best_y, log, runtime = local_search_advanced_uflp(y0, best_cost, f, c, max_iter=max_iter)

    st.success(f"✅ تم الانتهاء خلال {runtime:.2f} ثانية")
    st.write("🔹 **أفضل تكلفة نهائية:**", best)
    st.write("🔹 **المرافق المفتوحة النهائية:**", np.where(best_y > 0.5)[0].tolist())

    # رسم تطور التكلفة
    if log:
        iters, costs = zip(*log)
        fig, ax = plt.subplots(figsize=(8,4))
        ax.plot(iters, costs, 'o-', color='#00e6e6')
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Best Cost")
        ax.set_title("تطور التكلفة أثناء البحث المحلي")
        ax.grid(True)
        st.pyplot(fig)

# ============================================================
# مساحة التعليقات والرسائل
# ============================================================
st.markdown("---")
st.markdown("### 💬 مساحة التعليقات (GitHub Issue: `#swaps`)")
comment = st.text_area("أضف ملاحظاتك أو نتائجك هنا:", height=150)
if st.button("💾 حفظ التعليق"):
    st.info("تم حفظ التعليق محليًا (يمكن نسخه إلى GitHub Issue #swaps).")
    st.code(comment)