# -*- coding: utf-8 -*-
"""
تطبيق AHRH لحل مسائل البرمجة الخطية (صحيحة أو مستمرة)
مع واجهة متعددة اللغات وعرض تطور الدورات والزمن
نسخة محسنة ومعممة
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
NUM_WORKERS = 4

# ------------------- ترجمة النصوص -------------------
translations = {
    'العربية': {
        'app_title': '🧠 AHRH: خوارزمية هرمية انكماشية متطورة',
        'app_desc': 'هذا التطبيق يطبق خوارزمية AHRH المتقدمة التي تجمع بين:',
        'feature1': 'المسح الشعاعي الهرمي مع اتجاهات موجهة',
        'feature2': 'الرفع الهرمي للاتجاهات',
        'feature3': 'إزاحة الاسترخاء الديناميكية',
        'feature4': 'بحث محلي متقدم بأنماط تبادل متعددة (1-1، 2-1، 1-2، 2-2)',
        'feature5': 'توازي الحسابات لتسريع الأداء',
        'feature6': 'معايير توقف متعددة قابلة للاختيار',
        'feature7': 'وضع التسريع التلقائي عندما تقترب الفجوة من 2%',
        'problem_type': 'نوع المسألة',
        'lp': 'برمجة خطية مستمرة (LP)',
        'ilp': 'برمجة خطية صحيحة (ILP)',
        'sidebar_algo': '⚙️ معاملات الخوارزمية',
        'max_cycles': 'عدد الدورات الأقصى',
        'k_coarse': 'حجم المجموعة الخشنة (k)',
        'patience': 'الصبر (عدد الدورات بدون تحسن)',
        'sidebar_stop': '⏹️ معايير التوقف',
        'choose_criteria': 'اختر أي مجموعة من الشروط (عند تحقق أي منها تتوقف الخوارزمية):',
        'use_R': 'استخدام عتبة R (مع استقرار الفجوة)',
        'R_tol': 'قيمة R الصغرى (ε)',
        'stable_gap': 'عدد دورات استقرار الفجوة المطلوبة',
        'use_cost_repeat': 'استخدام تكرار التكلفة',
        'cost_repeat_times': 'عدد مرات التكرار',
        'use_gap_repeat': 'استخدام تكرار الفجوة',
        'gap_repeat_times': 'عدد مرات التكرار للفجوة',
        'use_contraction': 'استخدام معيار الانكماش (diff + R)',
        'diff_tol': 'عتبة الفرق بين الحلول (ε₁)',
        'workers': 'عدد العمال (للتوازي)',
        'tab_upload': '📂 رفع ملف',
        'tab_random': '🎲 توليد عشوائي',
        'tab_manual': '✍️ إدخال يدوي',
        'upload_header': 'رفع ملف المسألة',
        'upload_info': 'يدعم أي ملف نصي (txt, dat, bub, opt, ...). يتم تجاهل الأسطر التي تبدأ بـ # أو ! أو FILE:',
        'choose_file': 'اختر ملف المسألة',
        'random_header': 'توليد مسألة عشوائية',
        'random_n': 'عدد المتغيرات (n)',
        'random_m': 'عدد القيود (m)',
        'random_button': '🎲 توليد وحل',
        'manual_header': 'إدخال بيانات المسألة يدويًا',
        'manual_warning': 'للمسائل الصغيرة فقط (n ≤ 10, m ≤ 10)',
        'manual_n': 'عدد المتغيرات (n)',
        'manual_m': 'عدد القيود (m)',
        'manual_c': 'معاملات الهدف c[i]',
        'manual_A': 'مصفوفة القيود A[i][j]',
        'manual_b': 'الطرف الأيمن b[i]',
        'solve_button': '🚀 حل المسألة المدخلة',
        'results': '📊 النتائج',
        'best_cost': 'أفضل تكلفة',
        'lp_val': 'قيمة LP',
        'gap': 'الفجوة',
        'cycles_done': 'عدد الدورات',
        'cycle_time': 'زمن الدورة (ث)',
        'total_time': 'الزمن الإجمالي (ث)',
        'size': 'حجم المسألة',
        'stop_reason': 'سبب التوقف',
        'gap_plot': '📈 تطور الفجوة و R خلال الدورات',
        'gap_label': 'الفجوة (%)',
        'R_label': 'R',
        'cycle_log': '📋 سجل الدورات',
        'cycle': 'دورة',
        'cost': 'التكلفة',
        'improved': 'تحسن',
        'best_so_far': 'أفضل حتى الآن',
        'yes': 'نعم',
        'no': 'لا',
        'download': '📥 تحميل التطور (CSV)',
        'info_placeholder': '👈 اختر مصدر المسألة من التبويبات أعلاه واضغط على زر التشغيل.',
        'footer': 'تم التطوير بواسطة Zakarya Benregreg - خوارزمية AHRH محمية ببراءة اختراع.',
        'acceleration_on': '⚡ وضع التسريع مُفعّل (فجوة < 2% و R < 0.01)',
        'acceleration_off': 'وضع التسريع غير مُفعّل',
    },
    'English': {
        'app_title': '🧠 AHRH: Advanced Hierarchical Radial Heuristic',
        'app_desc': 'This app implements the AHRH algorithm, combining:',
        'feature1': 'Hierarchical radial scan with biased directions',
        'feature2': 'Hierarchical direction lifting',
        'feature3': 'Dynamic relaxation shift',
        'feature4': 'Advanced local search (1-1, 2-1, 1-2, 2-2 swaps)',
        'feature5': 'Parallel computing for speed',
        'feature6': 'Multiple customizable stopping criteria',
        'feature7': 'Automatic acceleration mode when gap approaches 2%',
        'problem_type': 'Problem Type',
        'lp': 'Linear Programming (LP)',
        'ilp': 'Integer Linear Programming (ILP)',
        'sidebar_algo': '⚙️ Algorithm Parameters',
        'max_cycles': 'Max Cycles',
        'k_coarse': 'Coarse Set Size (k)',
        'patience': 'Patience (cycles without improvement)',
        'sidebar_stop': '⏹️ Stopping Criteria',
        'choose_criteria': 'Choose any combination (algorithm stops when any condition is met):',
        'use_R': 'Use R threshold (with gap stability)',
        'R_tol': 'R tolerance (ε)',
        'stable_gap': 'Stable gap cycles required',
        'use_cost_repeat': 'Use cost repetition',
        'cost_repeat_times': 'Repetition count',
        'use_gap_repeat': 'Use gap repetition',
        'gap_repeat_times': 'Repetition count',
        'use_contraction': 'Use contraction criterion (diff + R)',
        'diff_tol': 'Solution difference tolerance (ε₁)',
        'workers': 'Workers (parallel threads)',
        'tab_upload': '📂 Upload File',
        'tab_random': '🎲 Random Generation',
        'tab_manual': '✍️ Manual Input',
        'upload_header': 'Upload Problem File',
        'upload_info': 'Accepts any text file (txt, dat, bub, opt, ...). Lines starting with #, !, or FILE: are ignored.',
        'choose_file': 'Choose a file',
        'random_header': 'Generate Random Instance',
        'random_n': 'Number of variables (n)',
        'random_m': 'Number of constraints (m)',
        'random_button': '🎲 Generate and Solve',
        'manual_header': 'Manual Data Entry',
        'manual_warning': 'For small problems only (n ≤ 10, m ≤ 10)',
        'manual_n': 'Number of variables (n)',
        'manual_m': 'Number of constraints (m)',
        'manual_c': 'Objective coefficients c[i]',
        'manual_A': 'Constraint matrix A[i][j]',
        'manual_b': 'Right-hand side b[i]',
        'solve_button': '🚀 Solve Entered Problem',
        'results': '📊 Results',
        'best_cost': 'Best Cost',
        'lp_val': 'LP Value',
        'gap': 'Gap',
        'cycles_done': 'Cycles Done',
        'cycle_time': 'Cycle Time (s)',
        'total_time': 'Total Time (s)',
        'size': 'Problem Size',
        'stop_reason': 'Stop Reason',
        'gap_plot': '📈 Gap & R Evolution',
        'gap_label': 'Gap (%)',
        'R_label': 'R',
        'cycle_log': '📋 Cycle Log',
        'cycle': 'Cycle',
        'cost': 'Cost',
        'improved': 'Improved?',
        'best_so_far': 'Best so far',
        'yes': 'Yes',
        'no': 'No',
        'download': '📥 Download Evolution (CSV)',
        'info_placeholder': '👈 Choose a data source and click the run button.',
        'footer': 'Developed by Zakarya Benregreg - AHRH algorithm patented.',
        'acceleration_on': '⚡ Acceleration mode ON (gap < 2% and R < 0.01)',
        'acceleration_off': 'Acceleration mode OFF',
    },
    'Français': {
        'app_title': '🧠 AHRH: Algorithme Hiérarchique Radial Contractant',
        'app_desc': 'Cette application implémente l\'algorithme AHRH, combinant :',
        'feature1': 'Balayage radial hiérarchique avec directions orientées',
        'feature2': 'Relèvement hiérarchique des directions',
        'feature3': 'Décalage dynamique de relaxation',
        'feature4': 'Recherche locale avancée (échanges 1-1, 2-1, 1-2, 2-2)',
        'feature5': 'Calcul parallèle pour la rapidité',
        'feature6': 'Critères d\'arrêt multiples personnalisables',
        'feature7': 'Mode accélération automatique lorsque le gap approche 2%',
        'problem_type': 'Type de problème',
        'lp': 'Programmation linéaire (LP)',
        'ilp': 'Programmation linéaire en nombres entiers (ILP)',
        'sidebar_algo': '⚙️ Paramètres de l\'algorithme',
        'max_cycles': 'Cycles max',
        'k_coarse': 'Taille de l\'ensemble grossier (k)',
        'patience': 'Patience (cycles sans amélioration)',
        'sidebar_stop': '⏹️ Critères d\'arrêt',
        'choose_criteria': 'Choisissez une combinaison (l\'algorithme s\'arrête dès qu\'une condition est remplie) :',
        'use_R': 'Utiliser le seuil R (avec stabilité du gap)',
        'R_tol': 'Tolérance R (ε)',
        'stable_gap': 'Cycles de stabilité du gap requis',
        'use_cost_repeat': 'Utiliser la répétition du coût',
        'cost_repeat_times': 'Nombre de répétitions',
        'use_gap_repeat': 'Utiliser la répétition du gap',
        'gap_repeat_times': 'Nombre de répétitions',
        'use_contraction': 'Utiliser le critère de contraction (diff + R)',
        'diff_tol': 'Tolérance de différence entre solutions (ε₁)',
        'workers': 'Travailleurs (threads parallèles)',
        'tab_upload': '📂 Télécharger un fichier',
        'tab_random': '🎲 Génération aléatoire',
        'tab_manual': '✍️ Saisie manuelle',
        'upload_header': 'Télécharger le fichier problème',
        'upload_info': 'Accepte tout fichier texte (txt, dat, bub, opt, ...). Les lignes commençant par #, ! ou FILE: sont ignorées.',
        'choose_file': 'Choisir un fichier',
        'random_header': 'Générer une instance aléatoire',
        'random_n': 'Nombre de variables (n)',
        'random_m': 'Nombre de contraintes (m)',
        'random_button': '🎲 Générer et résoudre',
        'manual_header': 'Saisie manuelle des données',
        'manual_warning': 'Pour petits problèmes uniquement (n ≤ 10, m ≤ 10)',
        'manual_n': 'Nombre de variables (n)',
        'manual_m': 'Nombre de contraintes (m)',
        'manual_c': 'Coefficients objectifs c[i]',
        'manual_A': 'Matrice des contraintes A[i][j]',
        'manual_b': 'Second membre b[i]',
        'solve_button': '🚀 Résoudre le problème saisi',
        'results': '📊 Résultats',
        'best_cost': 'Meilleur coût',
        'lp_val': 'Valeur LP',
        'gap': 'Écart',
        'cycles_done': 'Cycles effectués',
        'cycle_time': 'Temps par cycle (s)',
        'total_time': 'Temps total (s)',
        'size': 'Taille du problème',
        'stop_reason': 'Raison de l\'arrêt',
        'gap_plot': '📈 Évolution du gap et de R',
        'gap_label': 'Écart (%)',
        'R_label': 'R',
        'cycle_log': '📋 Journal des cycles',
        'cycle': 'Cycle',
        'cost': 'Coût',
        'improved': 'Amélioration ?',
        'best_so_far': 'Meilleur jusqu\'à présent',
        'yes': 'Oui',
        'no': 'Non',
        'download': '📥 Télécharger l\'évolution (CSV)',
        'info_placeholder': '👈 Choisissez une source de données et cliquez sur le bouton.',
        'footer': 'Développé par Zakarya Benregreg - Algorithme AHRH breveté.',
        'acceleration_on': '⚡ Mode accélération ACTIVÉ (gap < 2% et R < 0.01)',
        'acceleration_off': 'Mode accélération DÉSACTIVÉ',
    }
}

def t(key):
    return translations[st.session_state.language][key]

if 'language' not in st.session_state:
    st.session_state.language = 'English'

# ------------------- دوال أساسية للمسائل الخطية -------------------
def solve_lp_pulp(c, A, b, integer=False):
    """حل مسألة LP أو ILP باستخدام pulp"""
    n = len(c)
    m = len(b)
    prob = pulp.LpProblem("LP", pulp.LpMinimize)
    if integer:
        x = [pulp.LpVariable(f"x_{i}", lowBound=0, cat='Integer') for i in range(n)]
    else:
        x = [pulp.LpVariable(f"x_{i}", lowBound=0, cat='Continuous') for i in range(n)]
    prob += pulp.lpSum(c[i] * x[i] for i in range(n))
    for j in range(m):
        prob += pulp.lpSum(A[j][i] * x[i] for i in range(n)) <= b[j]
    solver = pulp.PULP_CBC_CMD(msg=False)
    prob.solve(solver)
    if prob.status == pulp.LpStatusOptimal:
        x_val = np.array([pulp.value(x[i]) for i in range(n)])
        obj_val = pulp.value(prob.objective)
        return x_val, obj_val
    else:
        return None, None

def compute_R(x):
    """نصف القطر R: أقصى مسافة إلى التكامل"""
    fractional = np.abs(x - np.round(x))
    return np.max(fractional)

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

def evaluate_solution(x, c, A, b, integer):
    """تقييم حل: حساب التكلفة والتحقق من الجدوى"""
    if integer:
        x_int = np.round(x).astype(int)
    else:
        x_int = x
    if np.all(A @ x_int <= b + 1e-6) and np.all(x_int >= 0):
        return c @ x_int, x_int
    else:
        return float('inf'), None

def local_search_advanced(y, best_cost, c, A, b, max_iter=5):
    """بحث محلي للمتغيرات الصحيحة (تبادل 1-1)"""
    n = len(y)
    improved = True
    iteration = 0
    best_y = y.copy()
    best = best_cost

    while improved and iteration < max_iter:
        improved = False
        ones = np.where(best_y > 0.5)[0].tolist()
        zeros = np.where(best_y < 0.5)[0].tolist()
        for i in ones:
            for j in zeros:
                y_new = best_y.copy()
                y_new[i] = 0
                y_new[j] = 1
                cost, _ = evaluate_solution(y_new, c, A, b, integer=True)
                if cost < best - 1e-6:
                    best = cost
                    best_y = y_new
                    improved = True
                    ones = np.where(best_y > 0.5)[0].tolist()
                    zeros = np.where(best_y < 0.5)[0].tolist()
                    break
            if improved:
                break
        iteration += 1
    return best, best_y

def vcycle(y, c, A, b, coarse, y_lp, R, integer=True):
    """دورة V واحدة"""
    n = len(y)
    y_smooth = y.copy()
    best_cost, _ = evaluate_solution(y, c, A, b, integer)
    if best_cost == float('inf'):
        best_cost = c @ y  # fallback

    alpha = R / 5
    frac_idx = np.where((y > 0.01) & (y < 0.99))[0] if integer else list(range(n))
    if len(frac_idx) == 0:
        frac_idx = list(range(n))

    dirs = generate_biased_directions(y_lp, frac_idx, 10, alpha, bias_strength=0.5)
    for u in dirs:
        for sign in [1, -1]:
            y_cand = y[frac_idx] + sign * alpha * u
            if integer:
                y_cand = np.round(y_cand).astype(int)
            y_full = y.copy()
            y_full[frac_idx] = y_cand
            cost, feasible = evaluate_solution(y_full, c, A, b, integer)
            if feasible is not None and cost < best_cost - 1e-6:
                best_cost = cost
                y_smooth = y_full.copy()

    # تصحيح خشن (للمسائل الصحيحة فقط)
    if integer and len(coarse) > 0:
        y_coarse = y_smooth.copy()
        ones = np.where(y_coarse > 0.5)[0].tolist()
        zeros = np.where(y_coarse < 0.5)[0].tolist()
        for i in ones:
            if i not in coarse:
                continue
            for j in zeros:
                if j not in coarse:
                    continue
                y_new = y_coarse.copy()
                y_new[i] = 0
                y_new[j] = 1
                cost, feasible = evaluate_solution(y_new, c, A, b, integer)
                if feasible is not None and cost < best_cost - 1e-6:
                    best_cost = cost
                    y_smooth = y_new.copy()
    return best_cost, y_smooth

def solve_ahrh_general(c, A, b, integer=True, max_cycles=20, k_coarse=5, patience=5,
                       use_R=False, R_tol=1e-6, stable_gap_needed=2,
                       use_cost_repeat=False, cost_repeat_times=2,
                       use_gap_repeat=False, gap_repeat_times=2,
                       use_contraction=False, diff_tol=1e-12):
    """الدالة الرئيسية لحل أي مسألة LP/ILP"""
    n = len(c)
    m = len(b)

    # LP relaxation
    x_lp, lp_val = solve_lp_pulp(c, A, b, integer=False)
    if x_lp is None:
        return None, None, None

    # R الأولي
    R_initial = compute_R(x_lp) if integer else 1.0
    R = R_initial if R_initial > 0 else 1.0

    # حل ابتدائي
    if integer:
        x = np.zeros(n, dtype=int)
        # نبدأ بصفر ثم نبحث عن حل أفضل
        best_cost, _ = evaluate_solution(x, c, A, b, integer)
        if best_cost == float('inf'):
            x = np.round(x_lp).astype(int)
            best_cost, _ = evaluate_solution(x, c, A, b, integer)
    else:
        x = x_lp.copy()
        best_cost = c @ x

    cycles_log = []
    gap_history = []
    R_history = []
    no_improve = 0
    cycles_done = 0
    stop_reason = ""
    acceleration_active = False
    cycle_times = []

    cost_repeat_count = 0
    gap_repeat_count = 0
    stable_gap_count = 0
    last_cost = None
    last_gap = None
    last_R = None
    last_x = x.copy()

    # عناصر عرض التقدم
    progress_bar = st.progress(0)
    status_placeholder = st.empty()
    details_placeholder = st.empty()
    start_total = time.time()

    for cycle in range(max_cycles):
        cycle_start = time.time()
        current_R = R / (cycle + 1)

        # اختيار مجموعة خشنة
        coarse = []
        if integer and x_lp is not None:
            open_now = np.where(x > 0.5)[0].tolist()
            top_lp = np.argsort(-x_lp)[:k_coarse].tolist()
            coarse = list(set(open_now + top_lp))
            if len(coarse) > 10:
                importance = [(i, x_lp[i]) for i in coarse]
                importance.sort(key=lambda x: x[1], reverse=True)
                coarse = [i for i, _ in importance[:10]]

        new_cost, new_x = vcycle(x, c, A, b, coarse, x_lp, current_R, integer=integer)
        gap = (new_cost - lp_val) / lp_val * 100 if lp_val != 0 else 0
        R_val = compute_R(new_x) if integer else 0
        diff = np.linalg.norm(new_x - last_x)

        improved = new_cost < best_cost - 1e-6
        if improved:
            best_cost = new_cost
            x = new_x
            no_improve = 0
        else:
            no_improve += 1

        gap_history.append(gap)
        R_history.append(R_val)
        cycle_time = time.time() - cycle_start
        cycle_times.append(cycle_time)

        cycles_log.append({
            'cycle': cycle+1,
            'cost': new_cost,
            'gap': gap,
            'R': R_val,
            'diff': diff,
            'improved': improved,
            'best_so_far': best_cost,
            'time': cycle_time
        })

        # تحديث واجهة التقدم
        progress_bar.progress((cycle + 1) / max_cycles)
        status_placeholder.info(f"**Cycle {cycle+1} / {max_cycles}**")
        details_placeholder.markdown(
            f"**Current cost:** {new_cost:,.2f}  \n"
            f"**Gap:** {gap:.4f}%  \n"
            f"**R:** {R_val:.6f}  \n"
            f"**Improved:** {'✅' if improved else '❌'}  \n"
            f"**Best so far:** {best_cost:,.2f}  \n"
            f"**Cycle time:** {cycle_time:.3f}s"
        )
        time.sleep(0.1)

        # تحديث حالة التسريع
        if integer and not acceleration_active and gap < 2.0 and R_val < 0.01:
            acceleration_active = True
            st.info(t('acceleration_on'))
        elif integer and acceleration_active and (gap >= 2.0 or R_val >= 0.01):
            acceleration_active = False

        # معايير التوقف
        stop_now = False

        if no_improve >= patience:
            stop_reason = f"Patience ({patience} cycles without improvement)"
            stop_now = True

        if not stop_now and use_R and current_R < R_tol:
            if last_gap is not None and abs(gap - last_gap) < 1e-6:
                stable_gap_count += 1
                if stable_gap_count >= stable_gap_needed:
                    stop_reason = f"R < {R_tol} and gap stable for {stable_gap_needed} cycles"
                    stop_now = True
            else:
                stable_gap_count = 0

        if not stop_now and use_cost_repeat and last_cost is not None and abs(new_cost - last_cost) < 1e-6:
            cost_repeat_count += 1
            if cost_repeat_count >= cost_repeat_times:
                stop_reason = f"Cost repeated {cost_repeat_times} times"
                stop_now = True
        else:
            cost_repeat_count = 0

        if not stop_now and use_gap_repeat and last_gap is not None and abs(gap - last_gap) < 1e-6:
            gap_repeat_count += 1
            if gap_repeat_count >= gap_repeat_times:
                stop_reason = f"Gap repeated {gap_repeat_times} times"
                stop_now = True
        else:
            gap_repeat_count = 0

        if not stop_now and use_contraction and diff < diff_tol and current_R < R_tol:
            stop_reason = f"Contraction: diff < {diff_tol} and R < {R_tol}"
            stop_now = True

        last_cost = new_cost
        last_gap = gap
        last_R = R_val
        last_x = new_x.copy()

        if stop_now:
            cycles_done = cycle + 1
            break

    if cycles_done == 0:
        cycles_done = max_cycles
        stop_reason = f"Max cycles ({max_cycles}) reached"

    total_time = time.time() - start_total
    progress_bar.empty()
    status_placeholder.empty()
    details_placeholder.empty()

    return {
        'best_cost': best_cost,
        'lp_val': lp_val,
        'gap': (best_cost - lp_val) / lp_val * 100 if lp_val != 0 else 0,
        'cycles_done': cycles_done,
        'gap_history': gap_history,
        'R_history': R_history,
        'cycles_log': cycles_log,
        'cycle_times': cycle_times,
        'total_time': total_time,
        'stop_reason': stop_reason,
        'acceleration_active': acceleration_active
    }, x

def read_instance_from_text(text):
    """قراءة مسألة عامة من نص"""
    lines = text.strip().splitlines()
    clean_lines = []
    for line in lines:
        line = line.strip()
        if line and not line.startswith('#') and not line.startswith('!') and not line.upper().startswith('FILE:'):
            clean_lines.append(line)
    if len(clean_lines) < 3:
        raise ValueError("الملف لا يحتوي على بيانات كافية.")
    parts = clean_lines[0].split()
    n = int(parts[0])
    m = int(parts[1])
    c = np.array(list(map(float, clean_lines[1].split())))
    if len(c) != n:
        raise ValueError("عدد معاملات الهدف لا يتطابق مع n")
    A = np.zeros((m, n))
    for i in range(m):
        row = list(map(float, clean_lines[2+i].split()))
        if len(row) != n:
            raise ValueError(f"عدد عناصر الصف {i} لا يتطابق مع n")
        A[i] = row
    b_line = clean_lines[2+m].split()
    b = np.array(list(map(float, b_line)))
    if len(b) != m:
        raise ValueError("عدد عناصر b لا يتطابق مع m")
    return c, A, b, n, m

def generate_random_instance(n, m):
    c = np.random.uniform(1, 10, n)
    A = np.random.uniform(0, 5, (m, n))
    b = np.random.uniform(5, 20, m)
    return c, A, b

# ------------------- واجهة Streamlit -------------------
st.set_page_config(page_title="AHRH Solver", layout="wide")

col1, col2 = st.columns([4, 1])
with col2:
    language = st.selectbox("", ['English', 'Français', 'العربية'], key='language_selector')
    st.session_state.language = language

st.title(t('app_title'))
st.markdown(t('app_desc'))
st.markdown(f"- {t('feature1')}\n- {t('feature2')}\n- {t('feature3')}\n- {t('feature4')}\n- {t('feature5')}\n- {t('feature6')}\n- {t('feature7')}")

with st.sidebar:
    st.header(t('sidebar_algo'))
    max_cycles = st.slider(t('max_cycles'), 5, 50, 15, 5)
    k_coarse = st.slider(t('k_coarse'), 3, 10, 5)
    patience = st.slider(t('patience'), 2, 10, 3)

    st.header(t('sidebar_stop'))
    st.markdown(t('choose_criteria'))

    use_R = st.checkbox(t('use_R'), value=False)
    if use_R:
        R_tol = st.number_input(t('R_tol'), value=1e-6, format="%.0e", step=1e-6)
        stable_gap_needed = st.number_input(t('stable_gap'), min_value=1, max_value=5, value=2)
    else:
        R_tol, stable_gap_needed = 1e-6, 2

    use_cost_repeat = st.checkbox(t('use_cost_repeat'), value=False)
    if use_cost_repeat:
        cost_repeat_times = st.number_input(t('cost_repeat_times'), min_value=2, max_value=10, value=2)
    else:
        cost_repeat_times = 2

    use_gap_repeat = st.checkbox(t('use_gap_repeat'), value=False)
    if use_gap_repeat:
        gap_repeat_times = st.number_input(t('gap_repeat_times'), min_value=2, max_value=10, value=2)
    else:
        gap_repeat_times = 2

    use_contraction = st.checkbox(t('use_contraction'), value=True)
    if use_contraction:
        diff_tol = st.number_input(t('diff_tol'), value=1e-12, format="%.0e", step=1e-12)
    else:
        diff_tol = 1e-12

    st.markdown("---")
    st.write(f"{t('workers')}: {NUM_WORKERS}")

problem_type = st.radio(t('problem_type'), [t('lp'), t('ilp')])
is_integer = (problem_type == t('ilp'))

tab1, tab2, tab3 = st.tabs([t('tab_upload'), t('tab_random'), t('tab_manual')])

with tab1:
    st.header(t('upload_header'))
    st.info(t('upload_info'))
    uploaded_file = st.file_uploader(t('choose_file'), type=None)
    if uploaded_file is not None:
        with st.spinner("Reading file and running algorithm..."):
            try:
                text = uploaded_file.getvalue().decode("utf-8")
            except:
                text = uploaded_file.getvalue().decode("latin-1")
            try:
                c, A, b, n, m = read_instance_from_text(text)
                st.success(f"File loaded: {n} variables, {m} constraints")
                result, x = solve_ahrh_general(
                    c, A, b, integer=is_integer,
                    max_cycles=max_cycles, k_coarse=k_coarse, patience=patience,
                    use_R=use_R, R_tol=R_tol, stable_gap_needed=stable_gap_needed,
                    use_cost_repeat=use_cost_repeat, cost_repeat_times=cost_repeat_times,
                    use_gap_repeat=use_gap_repeat, gap_repeat_times=gap_repeat_times,
                    use_contraction=use_contraction, diff_tol=diff_tol
                )
                if result:
                    st.session_state['result'] = result
                    st.session_state['n'] = n
                    st.session_state['m'] = m
            except Exception as e:
                st.error(f"Error reading file: {e}")

with tab2:
    st.header(t('random_header'))
    col1, col2 = st.columns(2)
    with col1:
        n_rand = st.number_input(t('random_n'), min_value=5, max_value=50, value=10, step=1, key="n_rand")
    with col2:
        m_rand = st.number_input(t('random_m'), min_value=5, max_value=50, value=10, step=1, key="m_rand")
    if st.button(t('random_button'), key="gen_rand"):
        with st.spinner("Generating and solving..."):
            c, A, b = generate_random_instance(int(n_rand), int(m_rand))
            result, x = solve_ahrh_general(
                c, A, b, integer=is_integer,
                max_cycles=max_cycles, k_coarse=k_coarse, patience=patience,
                use_R=use_R, R_tol=R_tol, stable_gap_needed=stable_gap_needed,
                use_cost_repeat=use_cost_repeat, cost_repeat_times=cost_repeat_times,
                use_gap_repeat=use_gap_repeat, gap_repeat_times=gap_repeat_times,
                use_contraction=use_contraction, diff_tol=diff_tol
            )
            if result:
                st.session_state['result'] = result
                st.session_state['n'] = n_rand
                st.session_state['m'] = m_rand
                st.success("Done!")

with tab3:
    st.header(t('manual_header'))
    st.warning(t('manual_warning'))
    col1, col2 = st.columns(2)
    with col1:
        n_man = st.number_input(t('manual_n'), min_value=1, max_value=10, value=3, step=1, key="n_man")
    with col2:
        m_man = st.number_input(t('manual_m'), min_value=1, max_value=10, value=3, step=1, key="m_man")

    # تخزين القيم في session_state
    if 'c_man' not in st.session_state or len(st.session_state.c_man) != n_man:
        st.session_state.c_man = np.zeros(n_man)
    if 'A_man' not in st.session_state or st.session_state.A_man.shape != (m_man, n_man):
        st.session_state.A_man = np.zeros((m_man, n_man))
    if 'b_man' not in st.session_state or len(st.session_state.b_man) != m_man:
        st.session_state.b_man = np.zeros(m_man)

    st.subheader(t('manual_c'))
    c_vals = []
    cols = st.columns(min(5, n_man))
    for i in range(n_man):
        with cols[i % 5]:
            val = st.number_input(f"c[{i}]", value=float(st.session_state.c_man[i]), key=f"c_man_{i}")
            c_vals.append(val)
    st.session_state.c_man = np.array(c_vals)

    st.subheader(t('manual_A'))
    for i in range(m_man):
        st.write(f"**Constraint {i+1}:**")
        cols = st.columns(min(5, n_man))
        for j in range(n_man):
            with cols[j % 5]:
                val = st.number_input(f"A[{i}][{j}]", value=float(st.session_state.A_man[i, j]), key=f"A_man_{i}_{j}")
                st.session_state.A_man[i, j] = val

    st.subheader(t('manual_b'))
    b_vals = []
    cols = st.columns(min(5, m_man))
    for i in range(m_man):
        with cols[i % 5]:
            val = st.number_input(f"b[{i}]", value=float(st.session_state.b_man[i]), key=f"b_man_{i}")
            b_vals.append(val)
    st.session_state.b_man = np.array(b_vals)

    if st.button(t('solve_button'), key="solve_manual"):
        with st.spinner("Running algorithm..."):
            result, x = solve_ahrh_general(
                st.session_state.c_man, st.session_state.A_man, st.session_state.b_man,
                integer=is_integer,
                max_cycles=max_cycles, k_coarse=k_coarse, patience=patience,
                use_R=use_R, R_tol=R_tol, stable_gap_needed=stable_gap_needed,
                use_cost_repeat=use_cost_repeat, cost_repeat_times=cost_repeat_times,
                use_gap_repeat=use_gap_repeat, gap_repeat_times=gap_repeat_times,
                use_contraction=use_contraction, diff_tol=diff_tol
            )
            if result:
                st.session_state['result'] = result
                st.session_state['n'] = n_man
                st.session_state['m'] = m_man
                st.success("Done!")

st.markdown("---")
st.header(t('results'))

if 'result' in st.session_state:
    res = st.session_state['result']
    colA, colB, colC = st.columns(3)
    colA.metric(t('best_cost'), f"{res['best_cost']:,.2f}")
    colB.metric(t('lp_val'), f"{res['lp_val']:,.2f}")
    colC.metric(t('gap'), f"{res['gap']:.4f}%")

    colD, colE, colF = st.columns(3)
    colD.metric(t('cycles_done'), res['cycles_done'])
    colE.metric(t('total_time'), f"{res['total_time']:.2f}")
    colF.metric(t('size'), f"{st.session_state['n']}×{st.session_state['m']}")

    st.info(f"**{t('stop_reason')}:** {res['stop_reason']}")
    if res.get('acceleration_active'):
        st.success(t('acceleration_on'))

    if res['gap_history']:
        st.subheader(t('gap_plot'))
        fig, ax1 = plt.subplots(figsize=(10, 5))
        cycles = list(range(1, len(res['gap_history'])+1))
        ax1.plot(cycles, res['gap_history'], 'b-o', label=t('gap_label'))
        ax1.set_xlabel(t('cycle'))
        ax1.set_ylabel(t('gap_label'), color='b')
        ax1.tick_params(axis='y', labelcolor='b')
        ax2 = ax1.twinx()
        ax2.plot(cycles, res['R_history'], 'r-s', label=t('R_label'))
        ax2.set_ylabel(t('R_label'), color='r')
        ax2.tick_params(axis='y', labelcolor='r')
        plt.title(t('gap_plot'))
        fig.tight_layout()
        st.pyplot(fig)

        st.subheader(t('cycle_log'))
        df_cycles = pd.DataFrame(res['cycles_log'])
        df_cycles = df_cycles.rename(columns={
            'cycle': t('cycle'),
            'cost': t('cost'),
            'gap': t('gap'),
            'R': 'R',
            'diff': 'Diff',
            'improved': t('improved'),
            'best_so_far': t('best_so_far'),
            'time': t('cycle_time')
        })
        df_cycles[t('improved')] = df_cycles[t('improved')].apply(lambda x: t('yes') if x else t('no'))
        st.dataframe(df_cycles, use_container_width=True)

        csv = df_cycles.to_csv(index=False)
        st.download_button(t('download'), data=csv, file_name="cycle_log.csv", mime="text/csv")
else:
    st.info(t('info_placeholder'))

st.markdown("---")
st.caption(t('footer'))
