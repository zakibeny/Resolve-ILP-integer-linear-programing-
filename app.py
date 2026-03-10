# -*- coding: utf-8 -*-
"""
AHRH: حل مسائل البرمجة الصحيحة (ILP/MILP) ومسائل مواقع المرافق (UFLP)
مع واجهة متعددة اللغات، عرض التقدم، معايير توقف متعددة، وحفظ الحالة
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
from concurrent.futures import ThreadPoolExecutor, as_completed
warnings.filterwarnings("ignore")

# ------------------- إعدادات -------------------
NUM_WORKERS = 4
STATE_FILE = "ahrh_state.json"

# ------------------- ترجمة النصوص -------------------
translations = {
    'العربية': {
        'app_title': '🧠 AHRH: حلول البرمجة الصحيحة (ILP/MILP) و UFLP',
        'app_desc': 'تطبيق لحل مسائل البرمجة الصحيحة (ILP) والبرمجة الصحيحة المختلطة (MILP) ومسائل مواقع المرافق (UFLP) باستخدام خوارزمية AHRH.',
        'note_uflp': 'ملاحظة: يتم تحويل مسائل UFLP تلقائياً إلى ILP.',
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
        'upload_info': 'يدعم جميع الصيغ: ILP, MILP, UFLP, MPS, LP, KoerkelGhosh, gs250a-1',
        'choose_file': 'اختر ملف المسألة',
        'supported_formats': 'الصيغ المدعومة: txt, uflp, ilp, mps, lp, dat, gms, mod, gs250a-1',
        'ilp_format_help': 'ILP: السطر الأول: n m، ثم سطر c، ثم m سطر A، ثم سطر b',
        'uflp_format_help': 'UFLP: السطر الأول: n m 0، ثم n سطر: f_i + m تكلفة نقل',
        'mps_format_help': 'MPS: صيغة MPS القياسية',
        'koerkel_help': 'KoerkelGhosh: يتم الكشف تلقائياً',
        'manual_header': 'إدخال بيانات ILP يدوياً',
        'manual_warning': 'للمسائل الصغيرة فقط (n ≤ 10, m ≤ 10)',
        'manual_n': 'عدد المتغيرات (n)',
        'manual_m': 'عدد القيود (m)',
        'manual_c': 'معاملات الهدف c[i]',
        'manual_A': 'مصفوفة القيود A[i][j]',
        'manual_b': 'الطرف الأيمن b[i]',
        'manual_f': 'تكاليف الفتح f[i]',
        'manual_costs': 'تكاليف النقل c[i][j]',
        'solve_button': '🚀 حل المسألة',
        'results': '📊 النتائج',
        'best_cost': 'أفضل تكلفة',
        'lp_val': 'قيمة LP',
        'gap': 'الفجوة (%)',
        'open_fac': 'المرافق المفتوحة',
        'cycles_done': 'عدد الدورات',
        'time': 'الزمن (ث)',
        'size': 'الحجم',
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
        'download': '📥 تحميل التطور',
        'info_placeholder': '👈 اختر مصدر المسألة',
        'footer': 'تم التطوير بواسطة Zakarya Benregreg - AHRH خوارزمية محمية ببراءة اختراع',
        'acceleration_on': '⚡ وضع التسريع مُفعّل',
        'acceleration_off': 'وضع التسريع غير مُفعّل',
        'save_state': '💾 تم حفظ الحالة',
        'resume_state': '🔄 تم العثور على حالة محفوظة',
        'no_state': '🆕 لا توجد حالة محفوظة',
        'pause': '⏸️ إيقاف مؤقت',
        'resume': '▶️ استئناف',
        'paused': '⏸️ تم الإيقاف المؤقت',
        'elapsed_time': 'الوقت المنقضي',
        'estimated_remaining': 'الوقت المتبقي',
        'gap_by_cycles': 'الفجوة حسب الدورات',
        'gap_by_R': 'الفجوة بدلالة R',
        'convergence_plot': 'مخطط التقارب',
        'cycles_vs_gap': 'الدورات مقابل الفجوة',
        'R_vs_gap': 'R مقابل الفجوة',
        'vs': 'مقابل',
        'file_name': 'اسم الملف',
        'file_size': 'حجم الملف',
        'variables': 'المتغيرات',
        'constraints': 'القيود',
        'problem_type': 'نوع المسألة',
        'ilp': 'برمجة صحيحة (ILP)',
        'milp': 'برمجة صحيحة مختلطة (MILP)',
        'uflp': 'مسألة مواقع المرافق (UFLP)',
        'select_problem_type': 'اختر نوع المسألة',
        'contact_info': '📞 معلومات الاتصال',
        'email': 'البريد الإلكتروني',
        'phone': 'الهاتف',
        'fax': 'فاكس',
        'reset': '🔄 إعادة تعيين',
        'delete_state': '🗑️ حذف الحالة المحفوظة',
    },
    'English': {
        'app_title': '🧠 AHRH: Integer/Mixed-Integer Programming & UFLP Solver',
        'app_desc': 'Solve ILP, MILP, and UFLP problems using the patented AHRH algorithm.',
        'note_uflp': 'Note: UFLP problems are automatically converted to ILP.',
        'sidebar_algo': '⚙️ Algorithm Parameters',
        'max_cycles': 'Max Cycles',
        'k_coarse': 'Coarse Set Size (k)',
        'patience': 'Patience',
        'sidebar_stop': '⏹️ Stopping Criteria',
        'choose_criteria': 'Choose any combination:',
        'use_R': 'Use R threshold',
        'R_tol': 'R tolerance (ε)',
        'stable_gap': 'Stable gap cycles',
        'use_cost_repeat': 'Use cost repetition',
        'cost_repeat_times': 'Repetition count',
        'use_gap_repeat': 'Use gap repetition',
        'gap_repeat_times': 'Repetition count',
        'use_contraction': 'Use contraction criterion',
        'diff_tol': 'Difference tolerance (ε₁)',
        'workers': 'Workers',
        'tab_upload': '📂 Upload File',
        'tab_manual': '✍️ Manual Input',
        'upload_header': 'Upload Problem File',
        'upload_info': 'Supports all formats: ILP, MILP, UFLP, MPS, LP, KoerkelGhosh, gs250a-1',
        'choose_file': 'Choose a file',
        'supported_formats': 'Supported formats: txt, uflp, ilp, mps, lp, dat, gms, mod, gs250a-1',
        'ilp_format_help': 'ILP: first line: n m, then c, then m A rows, then b',
        'uflp_format_help': 'UFLP: first line: n m 0, then n lines: f_i + m costs',
        'mps_format_help': 'MPS: Standard MPS format',
        'koerkel_help': 'KoerkelGhosh: Auto-detected',
        'manual_header': 'Manual ILP Data Entry',
        'manual_warning': 'For small problems only (n ≤ 10, m ≤ 10)',
        'manual_n': 'Variables (n)',
        'manual_m': 'Constraints (m)',
        'manual_c': 'Objective coefficients c[i]',
        'manual_A': 'Constraint matrix A[i][j]',
        'manual_b': 'Right-hand side b[i]',
        'manual_f': 'Opening costs f[i]',
        'manual_costs': 'Transport costs c[i][j]',
        'solve_button': '🚀 Solve Problem',
        'results': '📊 Results',
        'best_cost': 'Best Cost',
        'lp_val': 'LP Value',
        'gap': 'Gap (%)',
        'open_fac': 'Open facilities',
        'cycles_done': 'Cycles Done',
        'time': 'Time (s)',
        'size': 'Size',
        'stop_reason': 'Stop Reason',
        'gap_plot': '📈 Gap & R Evolution',
        'gap_label': 'Gap',
        'R_label': 'R',
        'cycle_log': '📋 Cycle Log',
        'cycle': 'Cycle',
        'cost': 'Cost',
        'improved': 'Improved',
        'best_so_far': 'Best so far',
        'yes': 'Yes',
        'no': 'No',
        'download': '📥 Download',
        'info_placeholder': '👈 Choose data source',
        'footer': 'Developed by Zakarya Benregreg - AHRH patented algorithm',
        'acceleration_on': '⚡ Acceleration ON',
        'acceleration_off': 'Acceleration OFF',
        'save_state': '💾 State saved',
        'resume_state': '🔄 Found saved state',
        'no_state': '🆕 No saved state',
        'pause': '⏸️ Pause',
        'resume': '▶️ Resume',
        'paused': '⏸️ Paused',
        'elapsed_time': 'Elapsed Time',
        'estimated_remaining': 'Estimated Remaining',
        'gap_by_cycles': 'Gap by Cycles',
        'gap_by_R': 'Gap by R',
        'convergence_plot': 'Convergence Plot',
        'cycles_vs_gap': 'Cycles vs Gap',
        'R_vs_gap': 'R vs Gap',
        'vs': 'vs',
        'file_name': 'File Name',
        'file_size': 'File Size',
        'variables': 'Variables',
        'constraints': 'Constraints',
        'problem_type': 'Problem Type',
        'ilp': 'Integer Programming (ILP)',
        'milp': 'Mixed-Integer Programming (MILP)',
        'uflp': 'Facility Location (UFLP)',
        'select_problem_type': 'Select Problem Type',
        'contact_info': '📞 Contact Information',
        'email': 'Email',
        'phone': 'Phone',
        'fax': 'Fax',
        'reset': '🔄 Reset',
        'delete_state': '🗑️ Delete Saved State',
    },
    'Français': {
        'app_title': '🧠 AHRH: Solveur PLNE/PLMNE et UFLP',
        'app_desc': 'Résoudre des problèmes PLNE, PLMNE et UFLP avec l\'algorithme AHRH breveté.',
        'note_uflp': 'Note: Les problèmes UFLP sont convertis automatiquement en PLNE.',
        'sidebar_algo': '⚙️ Paramètres',
        'max_cycles': 'Cycles max',
        'k_coarse': 'Taille ensemble grossier (k)',
        'patience': 'Patience',
        'sidebar_stop': '⏹️ Critères d\'arrêt',
        'choose_criteria': 'Choisissez une combinaison:',
        'use_R': 'Utiliser seuil R',
        'R_tol': 'Tolérance R (ε)',
        'stable_gap': 'Cycles stables',
        'use_cost_repeat': 'Répétition coût',
        'cost_repeat_times': 'Nombre répétitions',
        'use_gap_repeat': 'Répétition écart',
        'gap_repeat_times': 'Nombre répétitions',
        'use_contraction': 'Critère contraction',
        'diff_tol': 'Tolérance différence (ε₁)',
        'workers': 'Travailleurs',
        'tab_upload': '📂 Fichier',
        'tab_manual': '✍️ Saisie',
        'upload_header': 'Charger fichier',
        'upload_info': 'Tous formats: ILP, MILP, UFLP, MPS, LP, KoerkelGhosh, gs250a-1',
        'choose_file': 'Choisir fichier',
        'supported_formats': 'Formats supportés: txt, uflp, ilp, mps, lp, dat, gms, mod, gs250a-1',
        'ilp_format_help': 'ILP: 1ère ligne: n m, puis c, puis m lignes A, puis b',
        'uflp_format_help': 'UFLP: 1ère ligne: n m 0, puis n lignes: f_i + m coûts',
        'mps_format_help': 'MPS: Format MPS standard',
        'koerkel_help': 'KoerkelGhosh: Auto-détecté',
        'manual_header': 'Saisie manuelle',
        'manual_warning': 'Petits problèmes seulement (n ≤ 10, m ≤ 10)',
        'manual_n': 'Variables (n)',
        'manual_m': 'Contraintes (m)',
        'manual_c': 'Coefficients c[i]',
        'manual_A': 'Matrice A[i][j]',
        'manual_b': 'Second membre b[i]',
        'manual_f': 'Coûts ouverture f[i]',
        'manual_costs': 'Coûts transport c[i][j]',
        'solve_button': '🚀 Résoudre',
        'results': '📊 Résultats',
        'best_cost': 'Meilleur coût',
        'lp_val': 'Valeur LP',
        'gap': 'Écart (%)',
        'open_fac': 'Sites ouverts',
        'cycles_done': 'Cycles',
        'time': 'Temps (s)',
        'size': 'Taille',
        'stop_reason': 'Raison arrêt',
        'gap_plot': '📈 Évolution',
        'gap_label': 'Écart',
        'R_label': 'R',
        'cycle_log': '📋 Journal',
        'cycle': 'Cycle',
        'cost': 'Coût',
        'improved': 'Amélioré',
        'best_so_far': 'Meilleur',
        'yes': 'Oui',
        'no': 'Non',
        'download': '📥 Télécharger',
        'info_placeholder': '👈 Choisissez source',
        'footer': 'Développé par Zakarya Benregreg - Algorithme AHRH breveté',
        'acceleration_on': '⚡ Mode accélération',
        'acceleration_off': 'Mode normal',
        'save_state': '💾 État sauvegardé',
        'resume_state': '🔄 État trouvé',
        'no_state': '🆕 Pas d\'état',
        'pause': '⏸️ Pause',
        'resume': '▶️ Reprendre',
        'paused': '⏸️ En pause',
        'elapsed_time': 'Temps écoulé',
        'estimated_remaining': 'Temps restant',
        'gap_by_cycles': 'Écart par cycles',
        'gap_by_R': 'Écart par R',
        'convergence_plot': 'Graphique convergence',
        'cycles_vs_gap': 'Cycles vs Écart',
        'R_vs_gap': 'R vs Écart',
        'vs': 'vs',
        'file_name': 'Nom fichier',
        'file_size': 'Taille',
        'variables': 'Variables',
        'constraints': 'Contraintes',
        'problem_type': 'Type problème',
        'ilp': 'PLNE',
        'milp': 'PLMNE',
        'uflp': 'UFLP',
        'select_problem_type': 'Choisir type',
        'contact_info': '📞 Contact',
        'email': 'Email',
        'phone': 'Téléphone',
        'fax': 'Fax',
        'reset': '🔄 Réinitialiser',
        'delete_state': '🗑️ Supprimer état',
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

# ------------------- معلومات الاتصال -------------------
CONTACT_EMAIL = "zakibeny@gmail.com"
CONTACT_PHONE = "0021355614305 / 00213779679073"
CONTACT_FAX = "036495241"

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
    for j in range(m):
        for i in range(n):
            A[j, n + i*m + j] = 1
        b[j] = 1
    for i in range(n):
        for j in range(m):
            idx = m + i*m + j
            A[idx, n + i*m + j] = 1
            A[idx, i] = -1
            b[idx] = 0
    return obj, A, b, n_vars, n_constraints, n

# ------------------- Solution Evaluation -------------------
def evaluate_solution(x, obj, A, b, uflp_info=None):
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
            return obj @ x_int
        return float('inf')

def lp_relaxation(obj, A, b):
    n = len(obj)
    m = len(b)
    prob = pulp.LpProblem("LP_Relax", pulp.LpMinimize)
    x = [pulp.LpVariable(f"x_{i}", lowBound=0, cat='Continuous') for i in range(n)]
    prob += pulp.lpSum(obj[i] * x[i] for i in range(n))
    for j in range(m):
        prob += pulp.lpSum(A[j][i] * x[i] for i in range(n)) <= b[j]
    solver = pulp.PULP_CBC_CMD(msg=False)
    prob.solve(solver)
    if prob.status == pulp.LpStatusOptimal:
        x_val = np.array([pulp.value(x[i]) for i in range(n)])
        obj_val = pulp.value(prob.objective)
        return x_val, obj_val
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

def hierarchical_radial_scan(x_center, R_val, frac_idx, obj, A, b, best_cost, best_x, x_lp=None, uflp_info=None):
    if len(frac_idx) == 0:
        return best_cost, best_x
    x_frac = x_center[frac_idx].copy()
    n_free = len(frac_idx)
    local_best = best_cost
    local_best_x = best_x
    alpha = R_val / (np.sqrt(n_free) + 1e-12)
    dirs = generate_biased_directions(x_lp, frac_idx, 5, alpha)
    for u in dirs:
        for sign in [1, -1]:
            x_cand = x_frac + sign * alpha * u
            x_cand_int = np.round(x_cand).astype(int)
            x_cand_int = np.maximum(x_cand_int, 0)
            x_full = x_center.copy()
            x_full[frac_idx] = x_cand_int
            cost = evaluate_solution(x_full, obj, A, b, uflp_info)
            if cost < local_best:
                local_best = cost
                local_best_x = x_full.copy()
    return local_best, local_best_x

def local_search(x, best_cost, obj, A, b, uflp_info=None):
    n = len(x)
    best_x = x.copy()
    best = best_cost
    for i in range(n):
        x_new = best_x.copy()
        x_new[i] = 1 - x_new[i]
        cost = evaluate_solution(x_new, obj, A, b, uflp_info)
        if cost < best:
            best = cost
            best_x = x_new
    return best, best_x

def vcycle(x, obj, A, b, coarse, x_lp, best_cost, uflp_info=None):
    frac_idx = get_fractional_indices(x)
    R_val = compute_R(x)
    new_cost, new_x = hierarchical_radial_scan(x, R_val, frac_idx, obj, A, b, best_cost, x, x_lp, uflp_info)
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
            cost = evaluate_solution(x_full, obj, A, b, uflp_info)
            if cost < best_coarse:
                best_coarse = cost
                best_x_coarse = x_full
        if best_coarse < best_cost:
            best_cost = best_coarse
            x = best_x_coarse
    best_cost, x = local_search(x, best_cost, obj, A, b, uflp_info)
    return best_cost, x

# ------------------- Multi-format File Reading -------------------
def detect_file_format(text, filename=""):
    lines = text.strip().splitlines()
    clean_lines = [line.strip() for line in lines if line.strip() and not line.startswith(('#', '!', '//', '%', '*'))]
    if not clean_lines:
        return 'unknown'
    
    if 'gs250a-1' in filename.lower() or 'gs250a1' in filename.lower():
        return 'gs250a'
    
    first_line = clean_lines[0].upper()
    if first_line.startswith('NAME') or 'NAME' in first_line:
        return 'mps'
    if any(word in first_line for word in ['MIN', 'MAX', 'MINIMIZE', 'MAXIMIZE']):
        return 'lp'
    parts = clean_lines[0].split()
    if len(parts) == 3 and parts[2] == '0':
        return 'uflp'
    if len(parts) == 2 and all(p.replace('-','').replace('.','').replace('e','').replace('E','').replace('+','').isdigit() for p in parts):
        return 'ilp'
    return 'unknown'

def read_gs250a_file(text):
    lines = text.strip().splitlines()
    numbers = []
    for line in lines:
        line = line.strip()
        if line and not line.startswith(('#', '!', '//', '%', '*')):
            parts = line.split()
            for part in parts:
                try:
                    num = float(part)
                    numbers.append(num)
                except:
                    pass
    
    if len(numbers) < 3:
        raise ValueError("ملف gs250a-1 لا يحتوي على بيانات كافية")
    
    n = int(numbers[0])
    m = int(numbers[1])
    idx = 2
    if idx + n <= len(numbers):
        c = np.array(numbers[idx:idx+n])
        idx += n
    else:
        c = np.zeros(n)
    A = np.zeros((m, n))
    for i in range(m):
        if idx + n <= len(numbers):
            A[i] = numbers[idx:idx+n]
            idx += n
    if idx + m <= len(numbers):
        b = np.array(numbers[idx:idx+m])
    else:
        b = np.zeros(m)
    return c, A, b, n, m

def read_ilp_file(text):
    lines = text.strip().splitlines()
    clean_lines = []
    for line in lines:
        line = line.strip()
        if line and not line.startswith(('#', '!', '//', 'FILE:')):
            clean_lines.append(line)
    parts = clean_lines[0].split()
    n = int(parts[0])
    m = int(parts[1])
    c = np.array(list(map(float, clean_lines[1].split())))
    if len(c) != n:
        c = np.pad(c, (0, n - len(c)))
    A = np.zeros((m, n))
    for i in range(min(m, len(clean_lines)-2)):
        row = list(map(float, clean_lines[2+i].split()))
        if len(row) >= n:
            A[i] = row[:n]
        else:
            A[i, :len(row)] = row
    b_line = clean_lines[2+min(m, len(clean_lines)-2)] if 2+m < len(clean_lines) else clean_lines[-1]
    b = np.array(list(map(float, b_line.split())))
    if len(b) != m:
        b = np.pad(b, (0, m - len(b)))
    return c, A, b, n, m

def read_uflp_file(text):
    lines = text.strip().splitlines()
    clean_lines = []
    for line in lines:
        line = line.strip()
        if line and not line.startswith(('#', '!', '//', 'FILE:')):
            clean_lines.append(line)
    parts = clean_lines[0].split()
    n = int(parts[0])
    m = int(parts[1])
    f = np.zeros(n)
    c = np.zeros((n, m))
    for i in range(n):
        if i+1 >= len(clean_lines):
            break
        line = clean_lines[i+1]
        nums = list(map(float, line.split()))
        if len(nums) >= 1:
            f[i] = nums[0]
            for j in range(min(m, len(nums)-1)):
                c[i, j] = nums[1+j]
    return f, c, n, m

def flexible_file_reader(text, filename=""):
    format_type = detect_file_format(text, filename)
    
    if format_type == 'gs250a':
        c_vec, A, b, n, m = read_gs250a_file(text)
        return c_vec, A, b, n, m, None, 'ilp'
    
    elif format_type == 'uflp':
        f, c, n, m = read_uflp_file(text)
        obj, A, b, n_vars, n_constraints, n_y = uflp_to_ilp(f, c)
        uflp_info = {'n_y': n_y, 'f': f, 'c': c}
        return obj, A, b, n, m, uflp_info, 'uflp'
    
    else:
        try:
            c_vec, A, b, n, m = read_ilp_file(text)
            return c_vec, A, b, n, m, None, 'ilp'
        except:
            try:
                f, c, n, m = read_uflp_file(text)
                obj, A, b, n_vars, n_constraints, n_y = uflp_to_ilp(f, c)
                uflp_info = {'n_y': n_y, 'f': f, 'c': c}
                return obj, A, b, n, m, uflp_info, 'uflp'
            except Exception as e:
                raise ValueError(f"Could not read file: {str(e)}")

# ------------------- State Management -------------------
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

def delete_state(filename):
    if os.path.exists(filename):
        os.remove(filename)
        return True
    return False

# ------------------- Main Solving Function -------------------
def solve_ahrh(obj, A, b, uflp_info, max_cycles, k_coarse, patience,
               use_R, R_tol, stable_gap_needed,
               use_cost_repeat, cost_repeat_times,
               use_gap_repeat, gap_repeat_times,
               use_contraction, diff_tol, resume=False):
    
    n = len(obj)
    x_lp, lp_val = lp_relaxation(obj, A, b)
    if x_lp is None:
        lp_val = float('inf')
    
    if not resume:
        x = np.round(x_lp).astype(int) if x_lp is not None else np.zeros(n, dtype=int)
        x = np.maximum(x, 0)
        best_cost = evaluate_solution(x, obj, A, b, uflp_info)
        if best_cost == float('inf'):
            x = np.zeros(n, dtype=int)
            best_cost = evaluate_solution(x, obj, A, b, uflp_info)
        history = []
        total_time = 0.0
        start_cycle = 1
    else:
        state = load_state(STATE_FILE)
        if state is None:
            st.warning(t('no_state'))
            x = np.round(x_lp).astype(int) if x_lp is not None else np.zeros(n, dtype=int)
            x = np.maximum(x, 0)
            best_cost = evaluate_solution(x, obj, A, b, uflp_info)
            if best_cost == float('inf'):
                x = np.zeros(n, dtype=int)
                best_cost = evaluate_solution(x, obj, A, b, uflp_info)
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

    if resume and len(history) > 0:
        cycles_log = history
        gap_history = [h['gap']]
