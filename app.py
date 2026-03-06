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
NUM_WORKERS = 4

# ===================== دوال الخوارزمية الأساسية =====================

def solve_lp_fixed_y_uflp(y_int, f, c):
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

def local_search_advanced(y, best_cost, f, c, max_iter=10):
    n = len(f)
    improved = True
    iteration = 0
    best_y = y.copy()
    best = best_cost

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
                    open_fac = np.where(best_y > 0.5)[0].tolist()
                    closed_fac = np.where(best_y < 0.5)[0].tolist()
                    break
            if improved:
                break

        if not improved and len(open_fac) >= 2 and len(closed_fac) >= 1:
            # 2-1 exchange
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

        if not improved and len(open_fac) >= 1 and len(closed_fac) >= 2:
            # 1-2 exchange
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

        if not improved and len(open_fac) >= 2 and len(closed_fac) >= 2:
            # 2-2 exchange
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
    cost3, y3 = local_search_advanced(y2, cost2, f, c, max_iter=5)
    return cost3, y3

def read_instance_from_text(text):
    lines = text.strip().splitlines()
    clean_lines = []
    for line in lines:
        line = line.strip()
        if line and not line.startswith('#') and not line.startswith('!') and not line.upper().startswith('FILE:'):
            clean_lines.append(line)
    if len(clean_lines) < 2:
        raise ValueError("الملف لا يحتوي على بيانات كافية.")
    n, m = None, None
    data_start = -1
    for idx, line in enumerate(clean_lines):
        parts = line.split()
        if len(parts) >= 2:
            try:
                n_cand = int(parts[0])
                m_cand = int(parts[1])
                if n_cand > 0 and m_cand > 0:
                    n, m = n_cand, m_cand
                    data_start = idx
                    break
            except:
                continue
    if n is None or m is None:
        raise ValueError("لم يتم العثور على سطر صالح يحتوي على n و m.")
    f = np.zeros(n, dtype=float)
    c = np.zeros((n, m), dtype=float)
    for i in range(n):
        if data_start + 1 + i >= len(clean_lines):
            raise ValueError(f"الملف لا يحتوي على {n} سطراً من البيانات.")
        line = clean_lines[data_start + 1 + i]
        parts = line.split()
        if len(parts) == 1 + m:
            f[i] = float(parts[0])
            for j in range(m):
                c[i, j] = float(parts[1 + j])
        elif len(parts) >= 2 + m:
            idx = int(parts[0]) - 1
            f[idx] = float(parts[1])
            for j in range(m):
                c[idx, j] = float(parts[2 + j])
        else:
            raise ValueError(f"السطر {data_start+1+i+1} لا يحتوي على العدد المناسب من القيم.")
    return f, c, n, m

def generate_random_instance(n, m):
    f = np.random.uniform(1000, 20000, n)
    c = np.random.uniform(100, 500, (n, m))
    return f, c

def solve_ahrh_with_log(f, c, max_cycles, k_coarse, patience, stop_criteria):
    n, m = len(f), c.shape[1]
    y_lp, lp_val = lp_relaxation_uflp(f, c)
    if lp_val is None:
        lp_val = float('inf')
    y = np.ones(n, dtype=int)
    best = solve_lp_fixed_y_uflp(y, f, c)
    cycles_log = []
    gap_history = []
    R_history = []
    cost_history = []
    no_improve = 0
    cycles_done = 0
    start_time = time.time()
    stop_reason = ""

    # تكرارات التوقف
    cost_repeat_count = 0
    gap_repeat_count = 0
    last_cost = None
    last_gap = None

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
        R_val = compute_R(new_y)

        improved = new_cost < best - 1e-6
        if improved:
            best = new_cost
            y = new_y
            no_improve = 0
        else:
            no_improve += 1

        cost_history.append(new_cost)
        gap_history.append(gap)
        R_history.append(R_val)
        cycles_log.append({
            'cycle': cycle+1,
            'cost': new_cost,
            'gap': gap,
            'R': R_val,
            'improved': improved,
            'best_so_far': best
        })

        # ========== شروط التوقف ==========
        stop_now = False

        # شرط عدم التحسن (patience)
        if no_improve >= patience:
            stop_reason = f"لم يتحسن لمدة {patience} دورات متتالية"
            stop_now = True

        # شرط R
        if not stop_now and stop_criteria.get('use_R', False):
            if R_val < stop_criteria['R_tol']:
                stop_reason = f"R أصغر من العتبة ({R_val:.2e} < {stop_criteria['R_tol']:.0e})"
                stop_now = True

        # شرط تكرار التكلفة
        if not stop_now and stop_criteria.get('use_cost_repeat', False):
            if last_cost is not None and abs(new_cost - last_cost) < 1e-6:
                cost_repeat_count += 1
                if cost_repeat_count >= stop_criteria['cost_repeat_times']:
                    stop_reason = f"تكرار التكلفة ({stop_criteria['cost_repeat_times']} مرات)"
                    stop_now = True
            else:
                cost_repeat_count = 0
            last_cost = new_cost

        # شرط تكرار الفجوة
        if not stop_now and stop_criteria.get('use_gap_repeat', False):
            if last_gap is not None and abs(gap - last_gap) < 1e-6:
                gap_repeat_count += 1
                if gap_repeat_count >= stop_criteria['gap_repeat_times']:
                    stop_reason = f"تكرار الفجوة ({stop_criteria['gap_repeat_times']} مرات)"
                    stop_now = True
            else:
                gap_repeat_count = 0
            last_gap = gap

        if stop_now:
            cycles_done = cycle + 1
            break

    if cycles_done == 0:
        cycles_done = max_cycles
        stop_reason = "الحد الأقصى للدورات"

    total_time = time.time() - start_time

    return {
        'best_cost': best,
        'lp_val': lp_val,
        'gap': (best - lp_val) / lp_val * 100 if lp_val != float('inf') else 0,
        'open_fac': len(np.where(y > 0.5)[0]),
        'cycles_done': cycles_done,
        'gap_history': gap_history,
        'R_history': R_history,
        'cost_history': cost_history,
        'total_time': total_time,
        'cycles_log': cycles_log,
        'stop_reason': stop_reason
    }

# ===================== واجهة Streamlit =====================
st.set_page_config(page_title="AHRH Solver", layout="wide")
st.title("🧠 AHRH: خوارزمية هرمية انكماشية متطورة")
st.markdown("""
هذا التطبيق يطبق خوارزمية AHRH المتقدمة التي تجمع بين:
- المسح الشعاعي الهرمي مع اتجاهات موجهة
- الرفع الهرمي للاتجاهات
- إزاحة الاسترخاء الديناميكية
- بحث محلي متقدم بأنماط تبادل متعددة (1-1، 2-1، 1-2، 2-2)
- توازي الحسابات لتسريع الأداء
- **معايير توقف متعددة قابلة للاختيار**
""")

# الشريط الجانبي
with st.sidebar:
    st.header("⚙️ معاملات الخوارزمية")
    max_cycles = st.slider("عدد الدورات الأقصى", 5, 50, 15, 5)
    k_coarse = st.slider("حجم المجموعة الخشنة (k)", 3, 10, 5)
    patience = st.slider("الصبر (عدد الدورات بدون تحسن)", 2, 10, 3)

    st.header("⏹️ معايير التوقف")
    st.markdown("اختر أي مجموعة من الشروط (عند تحقق أي منها تتوقف الخوارزمية):")

    use_R = st.checkbox("استخدام عتبة R", value=True)
    R_tol = 1e-6
    if use_R:
        R_tol = st.number_input("قيمة R الصغرى (ε)", value=1e-6, format="%.0e", step=1e-6)

    use_cost_repeat = st.checkbox("استخدام تكرار التكلفة", value=False)
    cost_repeat_times = 2
    if use_cost_repeat:
        cost_repeat_times = st.number_input("عدد مرات التكرار", min_value=2, max_value=10, value=2)

    use_gap_repeat = st.checkbox("استخدام تكرار الفجوة", value=False)
    gap_repeat_times = 2
    if use_gap_repeat:
        gap_repeat_times = st.number_input("عدد مرات التكرار للفجوة", min_value=2, max_value=10, value=2)

    st.markdown("---")
    st.write(f"عدد العمال (للتوازي): {NUM_WORKERS}")

    # تجميع معايير التوقف في قاموس
    stop_criteria = {
        'use_R': use_R,
        'R_tol': R_tol,
        'use_cost_repeat': use_cost_repeat,
        'cost_repeat_times': cost_repeat_times,
        'use_gap_repeat': use_gap_repeat,
        'gap_repeat_times': gap_repeat_times
    }

# تبويبات
tab1, tab2, tab3 = st.tabs(["📂 رفع ملف", "🎲 توليد عشوائي", "✍️ إدخال يدوي"])

with tab1:
    st.header("رفع ملف المسألة")
    st.info("يدعم أي ملف نصي (txt, dat, bub, opt, ...). يتم تجاهل الأسطر التي تبدأ بـ # أو ! أو FILE:")
    uploaded_file = st.file_uploader("اختر ملف المسألة", type=None)
    if uploaded_file is not None:
        with st.spinner("جاري قراءة الملف وتشغيل الخوارزمية..."):
            try:
                text = uploaded_file.getvalue().decode("utf-8")
            except:
                text = uploaded_file.getvalue().decode("latin-1")
            try:
                f, c, n, m = read_instance_from_text(text)
                st.success(f"تم قراءة الملف بنجاح: {n} موقع، {m} عميل")
                result = solve_ahrh_with_log(f, c, max_cycles, k_coarse, patience, stop_criteria)
                st.session_state['result'] = result
                st.session_state['n'] = n
                st.session_state['m'] = m
            except Exception as e:
                st.error(f"خطأ في قراءة الملف: {e}")

with tab2:
    st.header("توليد مسألة عشوائية")
    col1, col2 = st.columns(2)
    with col1:
        n_rand = st.number_input("عدد المواقع (n)", min_value=5, max_value=200, value=50, step=5, key="n_rand")
    with col2:
        m_rand = st.number_input("عدد العملاء (m)", min_value=5, max_value=200, value=50, step=5, key="m_rand")
    if st.button("🎲 توليد وحل", key="gen_rand"):
        with st.spinner("جاري توليد المسألة وتشغيل الخوارزمية..."):
            f, c = generate_random_instance(int(n_rand), int(m_rand))
            result = solve_ahrh_with_log(f, c, max_cycles, k_coarse, patience, stop_criteria)
            st.session_state['result'] = result
            st.session_state['n'] = n_rand
            st.session_state['m'] = m_rand
            st.success("تم التوليد والحل بنجاح!")

with tab3:
    st.header("إدخال بيانات المسألة يدويًا")
    st.warning("للمسائل الصغيرة فقط (n ≤ 10, m ≤ 10)")
    col1, col2 = st.columns(2)
    with col1:
        n_man = st.number_input("عدد المواقع (n)", min_value=1, max_value=10, value=3, step=1, key="n_man")
    with col2:
        m_man = st.number_input("عدد العملاء (m)", min_value=1, max_value=10, value=3, step=1, key="m_man")

    if 'f_man' not in st.session_state or st.session_state.get('n_man_prev') != n_man:
        st.session_state['f_man'] = np.zeros(n_man)
        st.session_state['n_man_prev'] = n_man
    if 'c_man' not in st.session_state or st.session_state.get('n_man_prev') != n_man or st.session_state.get('m_man_prev') != m_man:
        st.session_state['c_man'] = np.zeros((n_man, m_man))
        st.session_state['m_man_prev'] = m_man

    st.subheader("تكاليف فتح المرافق f[i]")
    f_vals = []
    cols = st.columns(min(5, n_man))
    for i in range(n_man):
        with cols[i % 5]:
            val = st.number_input(f"f[{i}]", value=float(st.session_state['f_man'][i]), key=f"f_man_{i}")
            f_vals.append(val)
    st.session_state['f_man'] = np.array(f_vals)

    st.subheader("تكاليف النقل c[i][j]")
    c_vals = np.zeros((n_man, m_man))
    for i in range(n_man):
        st.write(f"**الموقع {i}:**")
        cols = st.columns(min(5, m_man))
        for j in range(m_man):
            with cols[j % 5]:
                val = st.number_input(f"c[{i}][{j}]", value=float(st.session_state['c_man'][i, j]), key=f"c_man_{i}_{j}")
                c_vals[i, j] = val
    st.session_state['c_man'] = c_vals

    if st.button("🚀 حل المسألة المدخلة", key="solve_manual"):
        with st.spinner("جاري تشغيل الخوارزمية..."):
            result = solve_ahrh_with_log(st.session_state['f_man'], st.session_state['c_man'], max_cycles, k_coarse, patience, stop_criteria)
            st.session_state['result'] = result
            st.session_state['n'] = n_man
            st.session_state['m'] = m_man
            st.success("تم الحل بنجاح!")

# ------------------- عرض النتائج -------------------
st.markdown("---")
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

    st.info(f"سبب التوقف: {res['stop_reason']}")

    if res['gap_history']:
        st.subheader("📈 تطور الفجوة و R خلال الدورات")
        fig, ax1 = plt.subplots(figsize=(10, 5))
        cycles = list(range(1, len(res['gap_history'])+1))
        ax1.plot(cycles, res['gap_history'], 'b-o', label='Gap (%)')
        ax1.set_xlabel("الدورة")
        ax1.set_ylabel("الفجوة (%)", color='b')
        ax1.tick_params(axis='y', labelcolor='b')
        ax2 = ax1.twinx()
        ax2.plot(cycles, res['R_history'], 'r-s', label='R')
        ax2.set_ylabel("R", color='r')
        ax2.tick_params(axis='y', labelcolor='r')
        plt.title("تطور الفجوة و R")
        fig.tight_layout()
        st.pyplot(fig)

        st.subheader("📋 سجل الدورات")
        df_cycles = pd.DataFrame(res['cycles_log'])
        df_cycles = df_cycles.rename(columns={
            'cycle': 'دورة',
            'cost': 'التكلفة',
            'gap': 'الفجوة %',
            'R': 'R',
            'improved': 'تحسن',
            'best_so_far': 'أفضل حتى الآن'
        })
        st.dataframe(df_cycles, use_container_width=True)

        # تحميل البيانات
        df = pd.DataFrame({
            "Cycle": cycles,
            "Gap (%)": res['gap_history'],
            "R": res['R_history']
        })
        csv = df.to_csv(index=False)
        st.download_button(
            label="📥 تحميل التطور (CSV)",
            data=csv,
            file_name="evolution.csv",
            mime="text/csv"
        )
else:
    st.info("👈 اختر مصدر المسألة من التبويبات أعلاه واضغط على زر التشغيل.")

st.markdown("---")
st.caption("تم التطوير بواسطة [Zakarya Benregreg] - خوارزمية AHRH محمية ببراءة اختراع.")    
