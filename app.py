```python
# -*- coding: utf-8 -*-
"""
AHRH Integer Programming Solver Application
"""
import streamlit as st
import numpy as np
import pulp
import time
import matplotlib.pyplot as plt
import pandas as pd
import json
import os
import traceback
import warnings
from datetime import datetime
warnings.filterwarnings("ignore")

# ------------------- Configuration -------------------
NUM_WORKERS = 4
STATE_FILE = "ahrh_state.json"

# ------------------- Translations -------------------
translations = {
    'العربية': {
        'app_title': '🧠 AHRH: حلول البرمجة الصحيحة',
        'app_desc': 'تطبيق لحل مسائل البرمجة الصحيحة (ILP) باستخدام خوارزمية AHRH.',
        'note_uflp': 'ملاحظة: يمكن رفع ملفات UFLP، ويتم تحويلها تلقائياً إلى ILP.',
        'sidebar_algo': '⚙️ معاملات الخوارزمية',
        'max_cycles': 'عدد الدورات الأقصى',
        'k_coarse': 'حجم المجموعة الخشنة (k)',
        'patience': 'الصبر (عدد الدورات بدون تحسن)',
        'sidebar_stop': '⏹️ معايير التوقف',
        'choose_criteria': 'اختر مجموعة من الشروط (عند تحقق أي منها تتوقف):',
        'use_R': 'استخدام عتبة R (مع استقرار الفجوة)',
        'R_tol': 'قيمة R الصغرى (ε)',
        'stable_gap': 'عدد دورات استقرار الفجوة',
        'use_cost_repeat': 'استخدام تكرار التكلفة',
        'cost_repeat_times': 'عدد مرات التكرار',
        'use_gap_repeat': 'استخدام تكرار الفجوة',
        'gap_repeat_times': 'عدد مرات التكرار',
        'use_contraction': 'استخدام معيار الانكماش (diff + R)',
        'diff_tol': 'عتبة الفرق بين الحلول (ε₁)',
        'workers': 'عدد العمال (للتوازي)',
        'tab_upload': '📂 رفع ملف',
        'tab_manual': '✍️ إدخال يدوي',
        'upload_header': 'رفع ملف المسألة',
        'upload_info': 'يدعم ملفات ILP العامة وملفات UFLP (يتم تحويلها).',
        'choose_file': 'اختر ملف المسألة',
        'ilp_format_help': 'التنسيق: السطر الأول: n m، ثم سطر c، ثم m سطر A، ثم سطر b.',
        'uflp_format_help': 'التنسيق: السطر الأول: n m 0، ثم n سطر: f_i + m قيمة نقل.',
        'manual_header': 'إدخال بيانات ILP يدوياً',
        'manual_warning': 'للمسائل الصغيرة فقط (n ≤ 10, m ≤ 10).',
        'manual_n': 'عدد المتغيرات (n)',
        'manual_m': 'عدد القيود (m)',
        'manual_c': 'معاملات الهدف c[i]',
        'manual_A': 'مصفوفة القيود A[i][j]',
        'manual_b': 'الطرف الأيمن b[i]',
        'solve_button': '🚀 حل المسألة',
        'results': '📊 النتائج',
        'best_cost': 'أفضل تكلفة',
        'lp_val': 'قيمة LP',
        'gap': 'الفجوة (%)',
        'open_fac': 'المرافق المفتوحة (إن وُجدت)',
        'cycles_done': 'عدد الدورات',
        'time': 'الزمن (ث)',
        'size': 'حجم المسألة',
        'stop_reason': 'سبب التوقف',
        'gap_plot': '📈 تطور الفجوة و R',
        'gap_label': 'الفجوة',
        'R_label': 'R',
        'cycle_log': '📋 سجل الدورات',
        'cycle': 'دورة',
        'cost': 'التكلفة',
        'improved': 'تحسن',
        'best_so_far': 'أفضل حتى الآن',
        'yes': 'نعم',
        'no': 'لا',
        'download': '📥 تحميل التطور (CSV)',
        'info_placeholder': '👈 اختر مصدر المسألة من التبويبات أعلاه.',
        'footer': 'تم التطوير بواسطة Zakarya Benregreg - خوارزمية AHRH محمية ببراءة اختراع.',
        'acceleration_on': '⚡ وضع التسريع مُفعّل',
        'acceleration_off': 'وضع التسريع غير مُفعّل',
        'save_state': 'تم حفظ الحالة. يمكنك الاستئناف لاحقاً.',
        'resume_state': 'تم العثور على حالة محفوظة. استئناف التشغيل...',
        'no_state': 'لم يتم العثور على حالة محفوظة. بدء جديد.',
        'pause': '⏸️ إيقاف مؤقت',
        'resume': '▶️ استئناف',
        'paused': 'تم الإيقاف المؤقت. اضغط استئناف للمتابعة.',
        'elapsed_time': 'الوقت المنقضي',
        'estimated_remaining': 'الوقت المتبقي المقدر',
        'gap_by_cycles': 'الفجوة حسب عدد الدورات',
        'gap_by_R': 'الفجوة بدلالة R',
        'convergence_plot': 'مخطط التقارب',
        'cycles_vs_gap': 'عدد الدورات مقابل الفجوة',
        'R_vs_gap': 'R مقابل الفجوة',
    },
    'English': {
        'app_title': '🧠 AHRH: Integer Programming Solver',
        'app_desc': 'Solve Integer Linear Programming (ILP) problems using AHRH algorithm.',
        'note_uflp': 'Note: UFLP files are automatically converted to ILP.',
        'sidebar_algo': '⚙️ Algorithm Parameters',
        'max_cycles': 'Max Cycles',
        'k_coarse': 'Coarse Set Size (k)',
        'patience': 'Patience (cycles without improvement)',
        'sidebar_stop': '⏹️ Stopping Criteria',
        'choose_criteria': 'Choose any combination (algorithm stops when any condition is met):',
        'use_R': 'Use R threshold (with gap stability)',
        'R_tol': 'R tolerance (ε)',
        'stable_gap': 'Stable gap cycles',
        'use_cost_repeat': 'Use cost repetition',
        'cost_repeat_times': 'Repetition count',
        'use_gap_repeat': 'Use gap repetition',
        'gap_repeat_times': 'Repetition count',
        'use_contraction': 'Use contraction criterion (diff + R)',
        'diff_tol': 'Solution difference tolerance (ε₁)',
        'workers': 'Workers (parallel threads)',
        'tab_upload': '📂 Upload File',
        'tab_manual': '✍️ Manual Input',
        'upload_header': 'Upload Problem File',
        'upload_info': 'Supports general ILP files and UFLP files (converted).',
        'choose_file': 'Choose a file',
        'ilp_format_help': 'Format: first line: n m, then c line, then m lines of A, then b line.',
        'uflp_format_help': 'Format: first line: n m 0, then n lines: f_i + m transport costs.',
        'manual_header': 'Manual ILP Data Entry',
        'manual_warning': 'For small problems only (n ≤ 10, m ≤ 10).',
        'manual_n': 'Number of variables (n)',
        'manual_m': 'Number of constraints (m)',
        'manual_c': 'Objective coefficients c[i]',
        'manual_A': 'Constraint matrix A[i][j]',
        'manual_b': 'Right-hand side b[i]',
        'solve_button': '🚀 Solve Problem',
        'results': '📊 Results',
        'best_cost': 'Best Cost',
        'lp_val': 'LP Value',
        'gap': 'Gap (%)',
        'open_fac': 'Open facilities (if any)',
        'cycles_done': 'Cycles Done',
        'time': 'Time (s)',
        'size': 'Problem Size',
        'stop_reason': 'Stop Reason',
        'gap_plot': '📈 Gap & R Evolution',
        'gap_label': 'Gap',
        'R_label': 'R',
        'cycle_log': '📋 Cycle Log',
        'cycle': 'Cycle',
        'cost': 'Cost',
        'improved': 'Improved?',
        'best_so_far': 'Best so far',
        'yes': 'Yes',
        'no': 'No',
        'download': '📥 Download Evolution (CSV)',
        'info_placeholder': '👈 Choose a data source from the tabs above.',
        'footer': 'Developed by Zakarya Benregreg - AHRH algorithm patented.',
        'acceleration_on': '⚡ Acceleration mode ON',
        'acceleration_off': 'Acceleration mode OFF',
        'save_state': 'State saved. You can resume later.',
        'resume_state': 'Found saved state. Resuming execution...',
        'no_state': 'No saved state found. Starting fresh.',
        'pause': '⏸️ Pause',
        'resume': '▶️ Resume',
        'paused': 'Paused. Click Resume to continue.',
        'elapsed_time': 'Elapsed Time',
        'estimated_remaining': 'Estimated Remaining',
        'gap_by_cycles': 'Gap by Cycles',
        'gap_by_R': 'Gap by R',
        'convergence_plot': 'Convergence Plot',
        'cycles_vs_gap': 'Cycles vs Gap',
        'R_vs_gap': 'R vs Gap',
    },
    'Français': {
        'app_title': '🧠 AHRH: Solveur de programmation en nombres entiers',
        'app_desc': 'Résoudre des problèmes de programmation linéaire en nombres entiers (ILP) avec l\'algorithme AHRH.',
        'note_uflp': 'Note: Les fichiers UFLP sont automatiquement convertis en ILP.',
        'sidebar_algo': '⚙️ Paramètres de l\'algorithme',
        'max_cycles': 'Cycles max',
        'k_coarse': 'Taille de l\'ensemble grossier (k)',
        'patience': 'Patience (cycles sans amélioration)',
        'sidebar_stop': '⏹️ Critères d\'arrêt',
        'choose_criteria': 'Choisissez une combinaison (l\'algorithme s\'arrête dès qu\'une condition est remplie) :',
        'use_R': 'Utiliser le seuil R (avec stabilité du gap)',
        'R_tol': 'Tolérance R (ε)',
        'stable_gap': 'Cycles de stabilité du gap',
        'use_cost_repeat': 'Utiliser la répétition du coût',
        'cost_repeat_times': 'Nombre de répétitions',
        'use_gap_repeat': 'Utiliser la répétition du gap',
        'gap_repeat_times': 'Nombre de répétitions',
        'use_contraction': 'Utiliser le critère de contraction (diff + R)',
        'diff_tol': 'Tolérance de différence (ε₁)',
        'workers': 'Travailleurs (threads parallèles)',
        'tab_upload': '📂 Télécharger un fichier',
        'tab_manual': '✍️ Saisie manuelle',
        'upload_header': 'Télécharger le fichier problème',
        'upload_info': 'Accepte les fichiers ILP généraux et UFLP (convertis).',
        'choose_file': 'Choisir un fichier',
        'ilp_format_help': 'Format: première ligne: n m, puis ligne c, puis m lignes A, puis ligne b.',
        'uflp_format_help': 'Format: première ligne: n m 0, puis n lignes: f_i + m coûts de transport.',
        'manual_header': 'Saisie manuelle des données ILP',
        'manual_warning': 'Pour petits problèmes uniquement (n ≤ 10, m ≤ 10).',
        'manual_n': 'Nombre de variables (n)',
        'manual_m': 'Nombre de contraintes (m)',
        'manual_c': 'Coefficients objectifs c[i]',
        'manual_A': 'Matrice des contraintes A[i][j]',
        'manual_b': 'Second membre b[i]',
        'solve_button': '🚀 Résoudre le problème',
        'results': '📊 Résultats',
        'best_cost': 'Meilleur coût',
        'lp_val': 'Valeur LP',
        'gap': 'Écart (%)',
        'open_fac': 'Sites ouverts (le cas échéant)',
        'cycles_done': 'Cycles effectués',
        'time': 'Temps (s)',
        'size': 'Taille du problème',
        'stop_reason': 'Raison de l\'arrêt',
        'gap_plot': '📈 Évolution du gap et de R',
        'gap_label': 'Écart',
        'R_label': 'R',
        'cycle_log': '📋 Journal des cycles',
        'cycle': 'Cycle',
        'cost': 'Coût',
        'improved': 'Amélioration ?',
        'best_so_far': 'Meilleur jusqu\'à présent',
        'yes': 'Oui',
        'no': 'Non',
        'download': '📥 Télécharger l\'évolution (CSV)',
        'info_placeholder': '👈 Choisissez une source de données dans les onglets ci-dessus.',
        'footer': 'Développé par Zakarya Benregreg - Algorithme AHRH breveté.',
        'acceleration_on': '⚡ Mode accélération ACTIVÉ',
        'acceleration_off': 'Mode accélération DÉSACTIVÉ',
        'save_state': 'État sauvegardé. Vous pouvez reprendre plus tard.',
        'resume_state': 'État sauvegardé trouvé. Reprise de l\'exécution...',
        'no_state': 'Aucun état sauvegardé trouvé. Nouveau départ.',
        'pause': '⏸️ Pause',
        'resume': '▶️ Reprendre',
        'paused': 'En pause. Cliquez sur Reprendre pour continuer.',
        'elapsed_time': 'Temps écoulé',
        'estimated_remaining': 'Temps restant estimé',
        'gap_by_cycles': 'Écart par cycles',
        'gap_by_R': 'Écart par R',
        'convergence_plot': 'Graphique de convergence',
        'cycles_vs_gap': 'Cycles vs Écart',
        'R_vs_gap': 'R vs Écart',
    }
}

def t(key):
    return translations[st.session_state.language][key]

if 'language' not in st.session_state:
    st.session_state.language = 'English'
if 'paused' not in st.session_state:
    st.session_state.paused = False
if 'stop_requested' not in st.session_state:
    st.session_state.stop_requested = False

# ------------------- UFLP to ILP Conversion -------------------
def uflp_to_ilp(f, c):
    n = len(f)
    m = c.shape[1]
    n_vars = n + n * m
    obj = np.zeros(n_vars)
    obj[:n] = f
    obj[n:] = c.flatten()
    n_constraints = m + n * m
    A = np.zeros((n_constraints, n_vars))
    b = np.zeros(n_constraints)
    # Coverage constraints
    for j in range(m):
        for i in range(n):
            A[j, n + i*m + j] = 1
        b[j] = 1
    # Linking constraints x <= y
    for i in range(n):
        for j in range(m):
            idx = m + i*m + j
            A[idx, n + i*m + j] = 1
            A[idx, i] = -1
            b[idx] = 0
    return obj, A, b, n_vars, n_constraints, n

# ------------------- Solution Evaluation -------------------
def evaluate_solution(x, obj, A, b, sense, uflp_info=None):
    if uflp_info is not None:
        n_y = uflp_info['n_y']
        f = uflp_info['f']
        c = uflp_info['c']
        y = np.round(x[:n_y]).astype(int)
        open_fac = np.where(y > 0.5)[0]
        if len(open_fac) == 0:
            return float('inf')
        total = np.sum(f[open_fac])
        for j in range(c.shape[1]):
            total += np.min(c[open_fac, j])
        return total
    else:
        x_int = np.round(x).astype(int)
        x_int = np.maximum(x_int, 0)
        if np.all(A @ x_int <= b + 1e-6):
            if sense == 1:
                return obj @ x_int
            else:
                return -(obj @ x_int)
        else:
            return float('inf')

def lp_relaxation(obj, A, b, sense=1):
    n = len(obj)
    m = len(b)
    prob = pulp.LpProblem("LP_Relax", pulp.LpMinimize if sense == 1 else pulp.LpMaximize)
    x = [pulp.LpVariable(f"x_{i}", lowBound=0, cat='Continuous') for i in range(n)]
    if sense == 1:
        prob += pulp.lpSum(obj[i] * x[i] for i in range(n))
    else:
        prob += pulp.lpSum(-obj[i] * x[i] for i in range(n))
    for j in range(m):
        prob += pulp.lpSum(A[j][i] * x[i] for i in range(n)) <= b[j]
    solver = pulp.PULP_CBC_CMD(msg=False)
    prob.solve(solver)
    if prob.status == pulp.LpStatusOptimal:
        x_val = np.array([pulp.value(x[i]) for i in range(n)])
        obj_val = pulp.value(prob.objective)
        if sense == 1:
            return x_val, obj_val
        else:
            return x_val, -obj_val
    else:
        return None, None

# ------------------- Algorithm Core Functions -------------------
def compute_R(y):
    return np.max(np.minimum(y, 1 - y))

def get_fractional_indices(y, eps=0.01):
    return np.where((y > eps) & (y < 1 - eps))[0]

def generate_biased_directions(x_lp, frac_idx, count, alpha, bias_strength=0.5):
    n_free = len(frac_idx)
    if x_lp is None or n_free == 0:
        dirs = np.random.randn(count, max(1, n_free))
        dirs = dirs / (np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-12)
        return dirs
    x_frac_target = np.clip(x_lp[frac_idx], 0, 1)
    base_dir = x_frac_target - 0.5
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

def hierarchical_radial_scan(x_center, R_val, frac_idx, obj, A, b, best_cost, best_x, x_lp=None, sense=1, uflp_info=None):
    if len(frac_idx) == 0:
        return best_cost, best_x
    x_frac = x_center[frac_idx].copy()
    n_free = len(frac_idx)
    local_best = best_cost
    local_best_x = best_x
    alpha = R_val / (np.sqrt(n_free) + 1e-12)
    dirs = generate_biased_directions(x_lp, frac_idx, 5, alpha, bias_strength=0.5)
    for u in dirs:
        for sign in [1, -1]:
            x_cand = x_frac + sign * alpha * u
            x_cand_int = np.round(x_cand).astype(int)
            x_cand_int = np.maximum(x_cand_int, 0)
            x_full = x_center.copy()
            x_full[frac_idx] = x_cand_int
            cost = evaluate_solution(x_full, obj, A, b, sense, uflp_info)
            if cost < local_best:
                local_best = cost
                local_best_x = x_full.copy()
    return local_best, local_best_x

def local_search(x, best_cost, obj, A, b, sense=1, uflp_info=None):
    n = len(x)
    best_x = x.copy()
    best = best_cost
    for i in range(n):
        x_new = best_x.copy()
        x_new[i] = 1 - x_new[i]
        cost = evaluate_solution(x_new, obj, A, b, sense, uflp_info)
        if cost < best:
            best = cost
            best_x = x_new
    return best, best_x

def vcycle(x, obj, A, b, coarse, x_lp, best_cost, sense=1, uflp_info=None):
    frac_idx = get_fractional_indices(x)
    R_val = compute_R(x)
    new_cost, new_x = hierarchical_radial_scan(x, R_val, frac_idx, obj, A, b, best_cost, x, x_lp, sense, uflp_info)
    if new_cost < best_cost:
        best_cost = new_cost
        x = new_x
    if len(coarse) <= 10 and len(coarse) > 0:
        best_coarse = best_cost
        best_x_coarse = x
        for bits in range(1 << len(coarse)):
            xc = np.array([(bits >> i) & 1 for i in range(len(coarse))])
            x_full = x.copy()
            for idx, val in zip(coarse, xc):
                x_full[idx] = val
            cost = evaluate_solution(x_full, obj, A, b, sense, uflp_info)
            if cost < best_coarse:
                best_coarse = cost
                best_x_coarse = x_full
        if best_coarse < best_cost:
            best_cost = best_coarse
            x = best_x_coarse
    best_cost, x = local_search(x, best_cost, obj, A, b, sense, uflp_info)
    return best_cost, x

# ------------------- File Reading Functions -------------------
def read_uflp_file(text):
    lines = text.strip().splitlines()
    clean_lines = []
    for line in lines:
        line = line.strip()
        if line and not line.startswith(('#', '!', '//', 'FILE:')):
            clean_lines.append(line)
    if len(clean_lines) < 2:
        raise ValueError("File does not contain enough data.")
    parts = clean_lines[0].split()
    if len(parts) == 3 and parts[2] == '0':
        n = int(parts[0])
        m = int(parts[1])
    elif len(parts) == 2:
        n = int(parts[0])
        m = int(parts[1])
    else:
        raise ValueError("First line format incorrect")
    f = np.zeros(n, dtype=float)
    c = np.zeros((n, m), dtype=float)
    for i in range(n):
        if i+1 >= len(clean_lines):
            raise ValueError(f"File does not contain {n} lines of data.")
        line = clean_lines[i+1]
        parts_line = line.split()
        if len(parts_line) == m+1:
            f[i] = float(parts_line[0])
            for j in range(m):
                c[i, j] = float(parts_line[1+j])
        elif len(parts_line) == m+2:
            f[i] = float(parts_line[1])
            for j in range(m):
                c[i, j] = float(parts_line[2+j])
        else:
            raise ValueError(f"Line {i+2} does not contain the correct number of values.")
    return f, c, n, m

def read_ilp_file(text):
    lines = text.strip().splitlines()
    clean_lines = []
    for line in lines:
        line = line.strip()
        if line and not line.startswith(('#', '!', '//', 'FILE:')):
            clean_lines.append(line)
    if len(clean_lines) < 3:
        raise ValueError("File does not contain enough data.")
    parts = clean_lines[0].split()
    n = int(parts[0])
    m = int(parts[1])
    c = np.array(list(map(float, clean_lines[1].split())))
    if len(c) != n:
        raise ValueError("Number of objective coefficients does not match n")
    A = np.zeros((m, n))
    for i in range(m):
        row = list(map(float, clean_lines[2+i].split()))
        if len(row) != n:
            raise ValueError(f"Number of elements in row {i} does not match n")
        A[i] = row
    b = np.array(list(map(float, clean_lines[2+m].split())))
    if len(b) != m:
        raise ValueError("Number of elements in b does not match m")
    return c, A, b, n, m

# ------------------- State Management Functions -------------------
def save_state(filename, cycle, best_cost, best_x, history, lp_val, total_time, problem_data):
    state = {
        'cycle': cycle,
        'best_cost': best_cost,
        'best_x': best_x.tolist(),
        'history': history,
        'lp_val': lp_val,
        'total_time': total_time,
        'problem_data': problem_data,
        'timestamp': datetime.now().isoformat()
    }
    with open(filename, 'w') as f:
        json.dump(state, f)

def load_state(filename):
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            state = json.load(f)
        state['best_x'] = np.array(state['best_x'])
        return state
    return None

# ------------------- Main Solving Function -------------------
def solve_ahrh(obj, A, b, uflp_info, sense, max_cycles, k_coarse, patience,
               use_R, R_tol, stable_gap_needed,
               use_cost_repeat, cost_repeat_times,
               use_gap_repeat, gap_repeat_times,
               use_contraction, diff_tol, resume=False):
    
    n = len(obj)
    x_lp, lp_val = lp_relaxation(obj, A, b, sense)
    if x_lp is None:
        lp_val = float('inf')
    
    if not resume:
        # Fresh start
        x = np.round(x_lp).astype(int) if x_lp is not None else np.zeros(n, dtype=int)
        x = np.maximum(x, 0)
        best_cost = evaluate_solution(x, obj, A, b, sense, uflp_info)
        if best_cost == float('inf'):
            x = np.zeros(n, dtype=int)
            best_cost = evaluate_solution(x, obj, A, b, sense, uflp_info)
        history = []
        total_time = 0.0
        start_cycle = 1
    else:
        # Resume from saved state
        state = load_state(STATE_FILE)
        if state is None:
            st.warning(t('no_state'))
            x = np.round(x_lp).astype(int) if x_lp is not None else np.zeros(n, dtype=int)
            x = np.maximum(x, 0)
            best_cost = evaluate_solution(x, obj, A, b, sense, uflp_info)
            if best_cost == float('inf'):
                x = np.zeros(n, dtype=int)
                best_cost = evaluate_solution(x, obj, A, b, sense, uflp_info)
            history = []
            total_time = 0.0
            start_cycle = 1
        else:
            st.info(t('resume_state'))
            start_cycle = state['cycle'] + 1
            best_cost = state['best_cost']
            x = state['best_x']
            history = state['history']
            total_time = state['total_time']
            lp_val = state['lp_val']

    cycles_log = []
    gap_history = []
    R_history = []
    diff_history = []
    no_improve = 0
    cycles_done = 0
    stop_reason = ""
    acceleration_active = False
    cost_repeat_count = 0
    gap_repeat_count = 0
    stable_gap_count = 0
    last_cost = None
    last_gap = None
    last_R = None
    last_x = x.copy()
    start_time = time.time()
    last_save_time = time.time()

    # Rebuild history from saved state
    if resume and len(history) > 0:
        cycles_log = history
        gap_history = [h['gap'] for h in history]
        R_history = [h['R'] for h in history]
        diff_history = [h['diff'] for h in history]
        last_cost = history[-1]['cost']
        last_gap = history[-1]['gap']
        last_R = history[-1]['R']
        last_x = np.array(history[-1].get('best_so_far', x))
        no_improve = 0
        acceleration_active = history[-1].get('acceleration', False)

    progress_bar = st.progress(0)
    status_placeholder = st.empty()
    details_placeholder = st.empty()
    time_placeholder = st.empty()
    pause_button_placeholder = st.empty()

    for cycle in range(start_cycle, max_cycles+1):
        # Check for pause
        while st.session_state.paused:
            time.sleep(0.5)
            if st.session_state.stop_requested:
                st.session_state.stop_requested = False
                st.session_state.paused = False
                break
        
        if x_lp is not None:
            open_now = np.where(x > 0.5)[0].tolist()
            top_lp = np.argsort(-x_lp)[:k_coarse].tolist()
            coarse = list(set(open_now + top_lp))
            if len(coarse) > 10:
                importance = [(i, x_lp[i]) for i in coarse]
                importance.sort(key=lambda x: x[1], reverse=True)
                coarse = [i for i, _ in importance[:10]]
        else:
            coarse = []
        
        if acceleration_active:
            st.session_state.acceleration = True
        else:
            st.session_state.acceleration = False
        
        new_cost, new_x = vcycle(x, obj, A, b, coarse, x_lp, best_cost, sense, uflp_info)
        gap = (new_cost - lp_val) / lp_val * 100 if lp_val not in [0, float('inf')] else 0
        R_val = compute_R(new_x)
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
        diff_history.append(diff)
        
        entry = {
            'cycle': cycle,
            'cost': new_cost,
            'gap': gap,
            'R': R_val,
            'diff': diff,
            'improved': improved,
            'best_so_far': best_cost,
            'acceleration': acceleration_active
        }
        cycles_log.append(entry)
        
        # Update progress
        progress = (cycle - start_cycle + 1) / (max_cycles - start_cycle + 1)
        progress_bar.progress(progress)
        
        # Time estimates
        elapsed = total_time + (time.time() - start_time)
        if cycle > start_cycle:
            avg_time_per_cycle = elapsed / (cycle - start_cycle + 1)
            remaining_cycles = max_cycles - cycle
            estimated_remaining = avg_time_per_cycle * remaining_cycles
        else:
            estimated_remaining = 0
        
        status_placeholder.info(f"**Cycle {cycle} / {max_cycles}**")
        details_placeholder.markdown(
            f"**Current cost:** {new_cost:,.2f}  \n"
            f"**Gap:** {gap:.4f}%  \n"
            f"**R:** {R_val:.6f}  \n"
            f"**Improved:** {'✅' if improved else '❌'}  \n"
            f"**Best so far:** {best_cost:,.2f}"
        )
        time_placeholder.markdown(
            f"**{t('elapsed_time')}:** {elapsed:.1f}s  \n"
            f"**{t('estimated_remaining')}:** {estimated_remaining:.1f}s"
        )
        
        # Pause button
        if pause_button_placeholder.button(t('pause' if not st.session_state.paused else 'resume'), key=f"pause_{cycle}"):
            st.session_state.paused = not st.session_state.paused
            if st.session_state.paused:
                st.info(t('paused'))
        
        # Update acceleration status
        if not acceleration_active and gap < 2.0 and R_val < 0.01:
            acceleration_active = True
            st.session_state.acceleration = True
            st.info(t('acceleration_on'))
        elif acceleration_active and (gap >= 2.0 or R_val >= 0.01):
            acceleration_active = False
            st.session_state.acceleration = False
        
        # Stopping criteria
        stop_now = False
        if no_improve >= patience:
            stop_reason = f"Patience ({patience} cycles without improvement)"
            stop_now = True
        
        if not stop_now and use_R and R_val < R_tol:
            if last_gap is not None and abs(gap - last_gap) < 1e-6:
                stable_gap_count += 1
                if stable_gap_count >= stable_gap_needed:
                    stop_reason = f"R < {R_tol} and gap stable for {stable_gap_needed} cycles"
                    stop_now = True
            else:
                stable_gap_count = 0
        
        if not stop_now and use_cost_repeat:
            if last_cost is not None and abs(new_cost - last_cost) < 1e-6:
                cost_repeat_count += 1
                if cost_repeat_count >= cost_repeat_times:
                    stop_reason = f"Cost repeated {cost_repeat_times} times"
                    stop_now = True
            else:
                cost_repeat_count = 0
        
        if not stop_now and use_gap_repeat:
            if last_gap is not None and abs(gap - last_gap) < 1e-6:
                gap_repeat_count += 1
                if gap_repeat_count >= gap_repeat_times:
                    stop_reason = f"Gap repeated {gap_repeat_times} times"
                    stop_now = True
            else:
                gap_repeat_count = 0
        
        if not stop_now and use_contraction:
            if diff < diff_tol and R_val < R_tol:
                stop_reason = f"Contraction: diff < {diff_tol} and R < {R_tol}"
                stop_now = True
        
        last_cost = new_cost
        last_gap = gap
        last_R = R_val
        last_x = new_x.copy()
        
        # Save state periodically
        current_time = time.time()
        if current_time - last_save_time > 5:  # Save every 5 seconds
            problem_data = {
                'type': 'UFLP' if uflp_info else 'ILP',
                'obj': obj.tolist(),
                'A': A.tolist(),
                'b': b.tolist(),
                'n': n,
                'sense': sense,
                'uflp_info': uflp_info
            }
            total_time_elapsed = total_time + (time.time() - start_time)
            save_state(STATE_FILE, cycle, best_cost, x, cycles_log, lp_val, total_time_elapsed, problem_data)
            last_save_time = current_time
        
        if stop_now:
            cycles_done = cycle
            break
    
    if cycles_done == 0:
        cycles_done = max_cycles
        stop_reason = f"Max cycles ({max_cycles}) reached"
    
    total_time = total_time + (time.time() - start_time)
    
    # Clear placeholders
    progress_bar.empty()
    status_placeholder.empty()
    details_placeholder.empty()
    time_placeholder.empty()
    pause_button_placeholder.empty()
    
    if acceleration_active:
        st.info(t('acceleration_on'))
    
    return {
        'best_cost': best_cost,
        'lp_val': lp_val,
        'gap': (best_cost - lp_val) / lp_val * 100 if lp_val not in [0, float('inf')] else 0,
        'cycles_done': cycles_done,
        'gap_history': gap_history,
        'R_history': R_history,
        'diff_history': diff_history,
        'cycles_log': cycles_log,
        'stop_reason': stop_reason,
        'acceleration_active': acceleration_active,
        'total_time': total_time,
        'open_fac': len(np.where(x > 0.5)[0]) if uflp_info else None
    }

# ------------------- Streamlit Interface -------------------
st.set_page_config(page_title="AHRH ILP Solver", layout="wide")

# Language selector
col1, col2 = st.columns([4, 1])
with col2:
    language = st.selectbox(
        "Language",
        options=['English', 'Français', 'العربية'],
        key='language_selector',
        label_visibility="collapsed"
    )
    st.session_state.language = language

# Title and description
st.title(t('app_title'))
st.markdown(t('app_desc'))
st.caption(t('note_uflp'))

# Sidebar parameters
with st.sidebar:
    st.header(t('sidebar_algo'))
    max_cycles = st.slider(t('max_cycles'), 5, 200, 50, 5)
    k_coarse = st.slider(t('k_coarse'), 3, 20, 10)
    patience = st.slider(t('patience'), 2, 50, 10)
    
    st.header(t('sidebar_stop'))
    st.markdown(t('choose_criteria'))
    
    use_R = st.checkbox(t('use_R'), value=True)
    if use_R:
        R_tol = st.number_input(t('R_tol'), value=1e-4, format="%.0e", step=1e-4)
        stable_gap_needed = st.number_input(t('stable_gap'), min_value=1, max_value=10, value=3)
    else:
        R_tol, stable_gap_needed = 1e-4, 3
    
    use_cost_repeat = st.checkbox(t('use_cost_repeat'), value=False)
    if use_cost_repeat:
        cost_repeat_times = st.number_input(t('cost_repeat_times'), min_value=2, max_value=20, value=5)
    else:
        cost_repeat_times = 5
    
    use_gap_repeat = st.checkbox(t('use_gap_repeat'), value=False)
    if use_gap_repeat:
        gap_repeat_times = st.number_input(t('gap_repeat_times'), min_value=2, max_value=20, value=5)
    else:
        gap_repeat_times = 5
    
    use_contraction = st.checkbox(t('use_contraction'), value=True)
    if use_contraction:
        diff_tol = st.number_input(t('diff_tol'), value=1e-6, format="%.0e", step=1e-6)
    else:
        diff_tol = 1e-6
    
    st.markdown("---")
    st.write(f"{t('workers')}: {NUM_WORKERS}")

# Main tabs
tab1, tab2 = st.tabs([t('tab_upload'), t('tab_manual')])

with tab1:
    st.header(t('upload_header'))
    st.info(t('upload_info'))
    
    with st.expander("📄 File Format Help"):
        st.markdown("**ILP:**\n" + t('ilp_format_help'))
        st.markdown("**UFLP:**\n" + t('uflp_format_help'))
    
    uploaded_file = st.file_uploader(t('choose_file'), type=['txt', 'uflp', 'ilp', 'mps'])
    
    if uploaded_file is not None:
        with st.spinner("Reading file and running algorithm..."):
            try:
                text = uploaded_file.getvalue().decode("utf-8", errors='ignore')
            except:
                text = uploaded_file.getvalue().decode("latin-1", errors='ignore')
            
            try:
                lines = [line.strip() for line in text.splitlines() if line.strip() and not line.startswith(('#', '!', '//', 'FILE:'))]
                first_parts = lines[0].split()
                
                if len(first_parts) == 3 and first_parts[2] == '0':
                    # UFLP
                    f, c, n, m = read_uflp_file(text)
                    st.success(f"UFLP file loaded: {n} facilities, {m} customers")
                    obj, A, b, n_vars, n_constraints, n_y = uflp_to_ilp(f, c)
                    uflp_info = {'n_y': n_y, 'f': f, 'c': c}
                    sense = 1
                else:
                    # ILP
                    c_vec, A, b, n, m = read_ilp_file(text)
                    st.success(f"ILP file loaded: {n} variables, {m} constraints")
                    obj = c_vec
                    uflp_info = None
                    sense = 1
                
                # Check for saved state
                if os.path.exists(STATE_FILE):
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button(t('resume')):
                            result = solve_ahrh(obj, A, b, uflp_info, sense,
                                                max_cycles, k_coarse, patience,
                                                use_R, R_tol, stable_gap_needed,
                                                use_cost_repeat, cost_repeat_times,
                                                use_gap_repeat, gap_repeat_times,
                                                use_contraction, diff_tol,
                                                resume=True)
                            st.session_state['result'] = result
                            st.session_state['n'] = n
                            st.session_state['m'] = m
                            st.session_state['is_uflp'] = (uflp_info is not None)
                    with col2:
                        if st.button(t('solve_button')):
                            if os.path.exists(STATE_FILE):
                                os.remove(STATE_FILE)
                            result = solve_ahrh(obj, A, b, uflp_info, sense,
                                                max_cycles, k_coarse, patience,
                                                use_R, R_tol, stable_gap_needed,
                                                use_cost_repeat, cost_repeat_times,
                                                use_gap_repeat, gap_repeat_times,
                                                use_contraction, diff_tol,
                                                resume=False)
                            st.session_state['result'] = result
                            st.session_state['n'] = n
                            st.session_state['m'] = m
                            st.session_state['is_uflp'] = (uflp_info is not None)
                else:
                    if st.button(t('solve_button')):
                        result = solve_ahrh(obj, A, b, uflp_info, sense,
                                            max_cycles, k_coarse, patience,
                                            use_R, R_tol, stable_gap_needed,
                                            use_cost_repeat, cost_repeat_times,
                                            use_gap_repeat, gap_repeat_times,
                                            use_contraction, diff_tol,
                                            resume=False)
                        st.session_state['result'] = result
                        st.session_state['n'] = n
                        st.session_state['m'] = m
                        st.session_state['is_uflp'] = (uflp_info is not None)
            
            except Exception as e:
                st.error(f"Error reading file: {e}")
                st.code(traceback.format_exc())

with tab2:
    st.header(t('manual_header'))
    st.warning(t('manual_warning'))
    
    col1, col2 = st.columns(2)
    with col1:
        n_man = st.number_input(t('manual_n'), min_value=1, max_value=10, value=3, step=1, key="n_man")
    with col2:
        m_man = st.number_input(t('manual_m'), min_value=1, max_value=10, value=3, step=1, key="m_man")
    
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
            try:
                c_vec = st.session_state.c_man
                A = st.session_state.A_man
                b = st.session_state.b_man
                obj = c_vec
                uflp_info = None
                sense = 1
                
                if os.path.exists(STATE_FILE):
                    if st.button(t('resume'), key="resume_manual"):
                        result = solve_ahrh(obj, A, b, uflp_info, sense,
                                            max_cycles, k_coarse, patience,
                                            use_R, R_tol, stable_gap_needed,
                                            use_cost_repeat, cost_repeat_times,
                                            use_gap_repeat, gap_repeat_times,
                                            use_contraction, diff_tol,
                                            resume=True)
                    else:
                        result = solve_ahrh(obj, A, b, uflp_info, sense,
                                            max_cycles, k_coarse, patience,
                                            use_R, R_tol, stable_gap_needed,
                                            use_cost_repeat, cost_repeat_times,
                                            use_gap_repeat, gap_repeat_times,
                                            use_contraction, diff_tol,
                                            resume=False)
                else:
                    result = solve_ahrh(obj, A, b, uflp_info, sense,
                                        max_cycles, k_coarse, patience,
                                        use_R, R_tol, stable_gap_needed,
                                        use_cost_repeat, cost_repeat_times,
                                        use_gap_repeat, gap_repeat_times,
                                        use_contraction, diff_tol,
                                        resume=False)
                
                st.session_state['result'] = result
                st.session_state['n'] = n_man
                st.session_state['m'] = m_man
                st.session_state['is_uflp'] = False
                st.success("Done!")
            
            except Exception as e:
                st.error(f"Error: {e}")
                st.code(traceback.format_exc())

# ------------------- Results Display -------------------
st.markdown("---")
st.header(t('results'))

if 'result' in st.session_state:
    res = st.session_state['result']
    
    # Metrics
    colA, colB, colC, colD = st.columns(4)
    colA.metric(t('best_cost'), f"{res['best_cost']:,.2f}")
    colB.metric(t('lp_val'), f"{res['lp_val']:,.2f}")
    colC.metric(t('gap'), f"{res['gap']:.4f}%")
    
    if st.session_state.get('is_uflp', False) and res['open_fac'] is not None:
        colD.metric(t('open_fac'), res['open_fac'])
    else:
        colD.metric(t('size'), f"{st.session_state['n']}×{st.session_state['m']}")
    
    colE, colF, colG = st.columns(3)
    colE.metric(t('cycles_done'), res['cycles_done'])
    colF.metric(t('time'), f"{res['total_time']:.2f}")
    colG.metric(t('size'), f"{st.session_state['n']}×{st.session_state['m']}")
    
    st.info(f"**{t('stop_reason')}:** {res['stop_reason']}")
    
    if res.get('acceleration_active'):
        st.success(t('acceleration_on'))
    
    # Plots
    if res['gap_history']:
        st.subheader(t('gap_plot'))
        
        # Create tabs for different plot views
        plot_tab1, plot_tab2 = st.tabs([t('gap_by_cycles'), t('gap_by_R')])
        
        with plot_tab1:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
            cycles = list(range(1, len(res['gap_history'])+1))
            
            # Gap vs Cycles with shaded area
            ax1.plot(cycles, res['gap_history'], 'b-', linewidth=2, label=t('gap_label'))
            ax1.fill_between(cycles, 0, res['gap_history'], alpha=0.3, color='blue')
            ax1.set_xlabel(t('cycle'))
            ax1.set_ylabel(t('gap_label'))
            ax1.set_title(t('cycles_vs_gap'))
            ax1.grid(True, alpha=0.3)
            ax1.legend()
            
            # R vs Cycles with shaded area
            ax2.plot(cycles, res['R_history'], 'r-', linewidth=2, label=t('R_label'))
            ax2.fill_between(cycles, 0, res['R_history'], alpha=0.3, color='red')
            ax2.set_xlabel(t('cycle'))
            ax2.set_ylabel(t('R_label'))
            ax2.set_title('R ' + t('vs') + ' ' + t('cycle'))
            ax2.grid(True, alpha=0.3)
            ax2.legend()
            
            plt.tight_layout()
            st.pyplot(fig)
        
        with plot_tab2:
            fig, ax = plt.subplots(figsize=(10, 5))
            
            # Gap vs R scatter with color gradient by cycle
            scatter = ax.scatter(res['R_history'], res['gap_history'], 
                               c=cycles, cmap='viridis', s=50, alpha=0.6)
            ax.set_xlabel(t('R_label'))
            ax.set_ylabel(t('gap_label'))
            ax.set_title(t('R_vs_gap'))
            ax.grid(True, alpha=0.3)
            plt.colorbar(scatter, ax=ax, label=t('cycle'))
            
            # Add trend line
            if len(res['R_history']) > 1:
                z = np.polyfit(res['R_history'], res['gap_history'], 1)
                p = np.poly1d(z)
                ax.plot(res['R_history'], p(res['R_history']), "r--", alpha=0.8, label='Trend')
                ax.legend()
            
            plt.tight_layout()
            st.pyplot(fig)
        
        # Convergence plot
        st.subheader(t('convergence_plot'))
        fig, ax = plt.subplots(figsize=(10, 5))
        
        # Plot gap evolution with logarithmic scale
        ax.semilogy(cycles, res['gap_history'], 'b-', linewidth=2, label=t('gap_label'))
        ax.semilogy(cycles, res['R_history'], 'r-', linewidth=2, label=t('R_label'))
        ax.set_xlabel(t('cycle'))
        ax.set_ylabel('Value (log scale)')
        ax.set_title(t('convergence_plot'))
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Cycle log table
        st.subheader(t('cycle_log'))
        df_cycles = pd.DataFrame(res['cycles_log'])
        df_cycles = df_cycles.rename(columns={
            'cycle': t('cycle'),
            'cost': t('cost'),
            'gap': t('gap'),
            'R': 'R',
            'diff': 'Diff',
            'improved': t('improved'),
            'best_so_far': t('best_so_far')
        })
        df_cycles[t('improved')] = df_cycles[t('improved')].apply(lambda x: t('yes') if x else t('no'))
        st.dataframe(df_cycles, use_container_width=True)
        
        # Download button
        df = pd.DataFrame({
            t('cycle'): cycles,
            t('gap_label'): res['gap_history'],
            'R': res['R_history']
        })
        csv = df.to_csv(index=False)
        st.download_button(
            label=t('download'),
            data=csv,
            file_name="evolution.csv",
            mime="text/csv"
        )
else:
    st.info(t('info_placeholder'))

st.markdown("---")
st.caption(t('footer'))

# Clean up on exit
if os.path.exists(STATE_FILE) and 'result' in st.session_state:
    # Optionally keep state file for resuming later
    pass
```
