# -*- coding: utf-8 -*-
"""
تطبيق Streamlit لخوارزمية AHRH مع الحفاظ على التوازي (ThreadPoolExecutor)
يمكنك:
- توليد مسائل عشوائية بأبعاد تختارها
- رفع ملفات مسائل حقيقية (بصيغة Körkel-Ghosh)
- تعديل معاملات الخوارزمية (عدد الدورات، حجم المجموعة الخشنة، الصبر)
- رؤية النتائج وتطور الفجوة بشكل تفاعلي
"""

import streamlit as st
import numpy as np
import pulp
import time
import matplotlib.pyplot as plt
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
warnings.filterwarnings("ignore")

# ------------------- إعدادات التوازي -------------------
NUM_WORKERS = 4  # يمكن تعديله حسب الحاجة

# ===================== دوال الخوارزمية الأساسية =====================

def solve_lp_fixed_y_uflp(y_int, f, c):
    """حل مشكلة التخصيص لـ UFLP بمتجه y محدد (المرافق المفتوحة)"""
    open_fac = np.where(y_int > 0.5)[0]
    if len(open_fac) == 0:
        return float('inf')
    total = np.sum(f[open_fac])
    for j in range(c.shape[1]):
        total += min(c[i, j] for i in open_fac)
    return total

def lp_relaxation_uflp(f, c):
    n, m = len(f), c.shape[1]
    lp = pulp.LpProblem("UFLP_LP_Relax", pulp.LpMinimize)
    y = [pulp.LpVariable(f"y_{i}", 0, 1) for i in range(n)]
    x = {(i, j): pulp.LpVariable(f"x_{i}_{j}", 0) for i in range(n) for j in range(m)}
    lp += pulp.lpSum(f[i] * y[i] for i in range(n)) + pulp.lpSum(c[i, j] * x[(i, j)] for i in range(n) for j in range(m))
    for j in range(m):
        lp += pulp.lpSum(x[(i, j)] for i in range(n)) == 1
    for i in range(n):
        for j in range(m):
            lp += x[(i, j)] <= y[i]
    solver = pulp.PULP_CBC_CMD(msg=False)
    lp.solve(solver)
    if lp.status == 1:
        y_val = np.array([pulp.value(y[i]) for i in range(n)])
        obj = pulp.value(lp.objective)
        return y_val, obj
    return None, None

def get_fractional_indices(y, eps=0.01):
    return np.where((y > eps) & (y < 1 - eps))[0]

def compute_R(y):
    return np.max(np.minimum(np.abs(y), np.abs(1 - y)))

def generate_biased_directions(y_lp, frac_idx, count, alpha, bias_strength=0.5):
    n_free = len(frac_idx)
    if y_lp is None or n_free == 0:
        dirs = np.random.randn(count, max(1, n_free))
        dirs = dirs / (np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-12)
        return dirs
    y_frac_target = np.clip(y_lp[frac_idx], 0, 1)
    base_dir = y_frac_target - 0.5
    if np.linalg.norm(base_dir) > 0:
        base_dir = base_dir / np.linalg.norm(base_dir)
    else:
        base_dir = np.zeros(n_free)
    dirs = []
    for _ in range(count):
        rand = np.random.randn(n_free)
        rand = rand / (np.linalg.norm(rand) + 1e-12)
        u = bias_strength * base_dir + (1 - bias_strength) * rand
        u = u / (np.linalg.norm(u) + 1e-12)
        dirs.append(u)
    return np.array(dirs)

def hierarchical_radial_scan_parallel(y_center, R, n_free, f, c, best_cost, best_y,
                                      n_layers=2, dirs_per_layer=10, alpha_schedule='adaptive',
                                      y_lp=None, gap_threshold=5.0):
    frac_idx = get_fractional_indices(y_center)
    if len(frac_idx) == 0:
        return best_cost, best_y
    y_frac = y_center[frac_idx].copy()
    n_free_actual = len(frac_idx)
    local_best = best_cost
    local_best_y = best_y
    base_alpha = R / (np.sqrt(n_free) + 1e-12) if n_free > 0 else 0.1

    for layer in range(1, n_layers + 1):
        if alpha_schedule == 'adaptive':
            alpha_k = base_alpha * np.exp(- (layer - 1) / n_layers)
        else:
            alpha_k = (layer / n_layers) * base_alpha
        dirs_count = max(3, dirs_per_layer // 2)
        dirs = generate_biased_directions(y_lp, frac_idx, dirs_count, alpha_k, bias_strength=0.5)

        def evaluate_direction(u):
            best_loc = local_best
            best_loc_y = local_best_y
            for sign in [1, -1]:
                y_cand = y_frac + sign * alpha_k * u
                y_cand_int = (y_cand > 0.5).astype(int)
                y_full = y_center.copy()
                y_full[frac_idx] = y_cand_int
                y_int = np.round(y_full).astype(int)
                cost = solve_lp_fixed_y_uflp(y_int, f, c)
                if cost < best_loc:
                    best_loc = cost
                    best_loc_y = y_int.copy()
            return best_loc, best_loc_y

        with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
            futures = [executor.submit(evaluate_direction, dirs[i]) for i in range(dirs_count)]
            for future in as_completed(futures):
                cost, y_cand = future.result()
                if cost < local_best:
                    local_best = cost
                    local_best_y = y_cand
        y_center = local_best_y.copy()
        y_frac = y_center[frac_idx].copy()
    return local_best, local_best_y

def smooth(y, f, c, y_lp=None, iters=1, gap_threshold=5.0):
    best = solve_lp_fixed_y_uflp(y, f, c)
    best_y = y.copy()
    for _ in range(iters):
        R_val = compute_R(y)
        n_free = len(get_fractional_indices(y))
        new_cost, new_y = hierarchical_radial_scan_parallel(
            y, R_val, n_free, f, c, best, best_y,
            n_layers=2, dirs_per_layer=8, alpha_schedule='adaptive',
            y_lp=y_lp, gap_threshold=gap_threshold
        )
        if new_cost < best:
            best = new_cost
            best_y = new_y
        else:
            break
    return best, best_y

def vcycle(y, f, c, coarse, y_lp=None, gap_threshold=5.0):
    cost1, y1 = smooth(y, f, c, y_lp=y_lp, iters=1, gap_threshold=gap_threshold)
    if not coarse:
        return cost1, y1
    best = cost1
    best_y = y1
    n_coarse = len(coarse)

    def evaluate_bits(bits):
        yc = np.array([(bits >> i) & 1 for i in range(n_coarse)])
        y_full = y1.copy()
        for idx, val in zip(coarse, yc):
            y_full[idx] = val
        cost = solve_lp_fixed_y_uflp(y_full, f, c)
        return cost, y_full

    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = [executor.submit(evaluate_bits, bits) for bits in range(1 << n_coarse)]
        for future in as_completed(futures):
            cost, y_cand = future.result()
            if cost < best:
                best = cost
                best_y = y_cand
    cost2, y2 = smooth(best_y, f, c, y_lp=y_lp, iters=1, gap_threshold=gap_threshold)
    return cost2, y2

def read_koerkel_ghosh_from_text(text):
    """قراءة ملف Körkel-Ghosh (تجاهل سطور FILE:)"""
    lines = text.strip().splitlines()
    lines = [l.strip() for l in lines if l.strip()]
    start_idx = -1
    n, m = None, None
    for idx, line in enumerate(lines):
        if line.upper().startswith('FILE:'):
            continue
        parts = line.split()
        if len(parts) >= 2:
            try:
                n_cand = int(parts[0])
                m_cand = int(parts[1])
                if n_cand > 0 and m_cand > 0:
                    n, m = n_cand, m_cand
                    start_idx = idx
                    break
            except:
                continue
    if n is None or m is None:
        raise ValueError("لم يتم العثور على سطر صالح يحتوي على n و m.")
    f = np.zeros(n, dtype=float)
    c = np.zeros((n, m), dtype=float)
    for i in range(n):
        line = lines[start_idx + 1 + i]
        parts = line.split()
        idx = int(parts[0]) - 1
        f[idx] = float(parts[1])
        for j in range(m):
            c[idx, j] = float(parts[2 + j])
    return f, c, n, m

def generate_random_instance(n, m):
    """توليد مسألة UFLP عشوائية"""
    f = np.random.uniform(1000, 20000, n)
    c = np.random.uniform(100, 500, (n, m))
    return f, c

def solve_ahrh(f, c, max_cycles, k_coarse, patience):
    """الدالة الرئيسية لحل المسألة مع إمكانية تعديل المعاملات"""
    n, m = len(f), c.shape[1]
    y_lp, lp_val = lp_relaxation_uflp(f, c)
    if lp_val is None:
        lp_val = float('inf')
    y = np.ones(n, dtype=int)
    best = solve_lp_fixed_y_uflp(y, f, c)
    gap_history = []
    cycles_done = 0
    no_improve = 0
    start_time = time.time()
    for cycle in range(max_cycles):
        if y_lp is not None:
            open_now = np.where(y > 0.5)[0].tolist()
            top_lp = np.argsort(-y_lp)[:k_coarse].tolist()
            coarse = list(set(open_now + top_lp))
            if len(coarse) > 10:
                importance = [(i, y_lp[i]) for i in coarse]
                importance.sort(key=lambda x: x[1], reverse=True)
                coarse = [i for i, _ in importance[:10]]
        else:
            coarse = []
        new_cost, new_y = vcycle(y, f, c, coarse, y_lp=y_lp, gap_threshold=3.0)
        gap = (new_cost - lp_val) / lp_val * 100 if lp_val != float('inf') else 0
        gap_history.append(gap)
        if new_cost < best - 1e-6:
            best = new_cost
            y = new_y
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                cycles_done = cycle + 1
                break
    if cycles_done == 0:
        cycles_done = max_cycles
    total_time = time.time() - start_time
    return {
        'best_cost': best,
        'lp_val': lp_val,
        'gap': (best - lp_val) / lp_val * 100 if lp_val != float('inf') else 0,
        'open_fac': len(np.where(y > 0.5)[0]),
        'cycles_done': cycles_done,
        'gap_history': gap_history,
        'total_time': total_time
    }

# ===================== واجهة Streamlit =====================
st.set_page_config(page_title="AHRH Solver", layout="wide")
st.title("🧠 AHRH: خوارزمية هرمية انكماشية لحل مسائل UFLP")
st.markdown("---")

# إدخال المعاملات
with st.sidebar:
    st.header("⚙️ معاملات الخوارزمية")
    max_cycles = st.slider("عدد الدورات الأقصى", 5, 50, 15, 5)
    k_coarse = st.slider("حجم المجموعة الخشنة (k)", 3, 10, 5)
    patience = st.slider("الصبر (عدد الدورات بدون تحسن)", 2, 10, 3)
    st.markdown("---")
    st.write("عدد العمال المستخدم في التوازي:", NUM_WORKERS)

col1, col2 = st.columns([1, 2])

with col1:
    st.header("📂 مصدر المسألة")
    option = st.radio("اختر:", ("توليد عشوائي", "رفع ملف"))

    if option == "توليد عشوائي":
        n = st.number_input("عدد المواقع (n)", min_value=5, max_value=200, value=50, step=5)
        m = st.number_input("عدد العملاء (m)", min_value=5, max_value=200, value=50, step=5)
        if st.button("🚀 توليد وحل"):
            with st.spinner("جاري توليد المسألة وتشغيل الخوارزمية..."):
                f, c = generate_random_instance(int(n), int(m))
                result = solve_ahrh(f, c, max_cycles, k_coarse, patience)
                st.session_state['result'] = result
                st.session_state['n'] = n
                st.session_state['m'] = m
    else:
        uploaded_file = st.file_uploader("ارفع ملف المسألة (txt)", type=["txt"])
        if uploaded_file is not None:
            with st.spinner("جاري قراءة الملف وتشغيل الخوارزمية..."):
                text = uploaded_file.getvalue().decode("utf-8")
                try:
                    f, c, n, m = read_koerkel_ghosh_from_text(text)
                    result = solve_ahrh(f, c, max_cycles, k_coarse, patience)
                    st.session_state['result'] = result
                    st.session_state['n'] = n
                    st.session_state['m'] = m
                except Exception as e:
                    st.error(f"خطأ في قراءة الملف: {e}")

with col2:
    st.header("📊 النتائج")
    if 'result' in st.session_state:
        res = st.session_state['result']
        colA, colB, colC, colD = st.columns(4)
        colA.metric("أفضل تكلفة", f"{res['best_cost']:,.0f}")
        colB.metric("قيمة LP", f"{res['lp_val']:,.0f}")
        colC.metric("الفجوة", f"{res['gap']:.4f}%")
        colD.metric("المرافق المفتوحة", res['open_fac'])

        colE, colF, colG = st.columns(3)
        colE.metric("عدد الدورات", res['cycles_done'])
        colF.metric("الزمن (ث)", f"{res['total_time']:.2f}")
        colG.metric("حجم المسألة", f"{st.session_state['n']}×{st.session_state['m']}")

        if res['gap_history']:
            st.subheader("📈 تطور الفجوة خلال الدورات")
            fig, ax = plt.subplots(figsize=(10, 5))
            cycles = list(range(1, len(res['gap_history'])+1))
            ax.plot(cycles, res['gap_history'], marker='o', linestyle='-', color='b')
            ax.set_xlabel("الدورة")
            ax.set_ylabel("الفجوة (%)")
            ax.set_title("تطور الفجوة")
            ax.grid(True)
            st.pyplot(fig)

            df = pd.DataFrame({"Cycle": cycles, "Gap (%)": res['gap_history']})
            csv = df.to_csv(index=False)
            st.download_button(
                label="📥 تحميل تطور الفجوة (CSV)",
                data=csv,
                file_name="gap_evolution.csv",
                mime="text/csv"
            )
    else:
        st.info("👈 اختر مصدر المسألة واضغط على زر التشغيل لعرض النتائج.")

st.markdown("---")
st.caption("تم التطوير بواسطة [اسمك] - خوارزمية AHRH محمية ببراءة اختراع.")
