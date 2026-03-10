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
        'app_title': 'AHRH: حلول البرمجة الصحيحة (ILP/MILP) و UFLP',
        'app_desc': 'تطبيق لحل مسائل البرمجة الصحيحة (ILP) والبرمجة الصحيحة المختلطة (MILP) ومسائل مواقع المرافق (UFLP) باستخدام خوارزمية AHRH.',
        'note_uflp': 'ملاحظة: يتم تحويل مسائل UFLP تلقائياً إلى ILP.',
        'sidebar_algo': 'معاملات الخوارزمية',
        'max_cycles': 'عدد الدورات الأقصى',
        'k_coarse': 'حجم المجموعة الخشنة (k)',
        'patience': 'الصبر (عدد الدورات بدون تحسن)',
        'sidebar_stop': 'معايير التوقف',
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
        'tab_upload': 'رفع ملف',
        'tab_manual': 'إدخال يدوي',
        'upload_header': 'رفع ملف المسألة',
        'upload_info': 'يقبل أي ملف نصي ويحاول قراءته تلقائياً (ILP, MILP, UFLP, MPS, LP, gs250a-1)',
        'choose_file': 'اختر ملف المسألة',
        'supported_formats': 'جميع الملفات النصية مقبولة - يتم الكشف عن التنسيق تلقائياً',
        'ilp_format_help': 'ILP: السطر الأول: n m، ثم سطر c، ثم m سطر A، ثم سطر b',
        'uflp_format_help': 'UFLP: السطر الأول: n m 0، ثم n سطر: f_i + m تكلفة نقل',
        'mps_format_help': 'MPS: صيغة MPS القياسية',
        'koerkel_help': 'gs250a-1: يتم الكشف تلقائياً',
        'manual_header': 'إدخال بيانات ILP يدوياً',
        'manual_warning': 'للمسائل الصغيرة فقط (n ≤ 10, m ≤ 10)',
        'manual_n': 'عدد المتغيرات (n)',
        'manual_m': 'عدد القيود (m)',
        'manual_c': 'معاملات الهدف c[i]',
        'manual_A': 'مصفوفة القيود A[i][j]',
        'manual_b': 'الطرف الأيمن b[i]',
        'manual_f': 'تكاليف الفتح f[i]',
        'manual_costs': 'تكاليف النقل c[i][j]',
        'solve_button': 'حل المسألة',
        'results': 'النتائج',
        'best_cost': 'أفضل تكلفة',
        'lp_val': 'قيمة LP',
        'gap': 'الفجوة (%)',
        'open_fac': 'المرافق المفتوحة',
        'cycles_done': 'عدد الدورات',
        'time': 'الزمن (ث)',
        'size': 'الحجم',
        'stop_reason': 'سبب التوقف',
        'gap_plot': 'تطور الفجوة و R',
        'gap_label': 'الفجوة',
        'R_label': 'R',
        'cycle_log': 'سجل الدورات',
        'cycle': 'دورة',
        'cost': 'التكلفة',
        'improved': 'تحسن',
        'best_so_far': 'أفضل حتى الآن',
        'yes': 'نعم',
        'no': 'لا',
        'download': 'تحميل التطور',
        'info_placeholder': 'اختر مصدر المسألة',
        'footer': 'تم التطوير بواسطة Zakarya Benregreg - AHRH خوارزمية محمية ببراءة اختراع',
        'acceleration_on': 'وضع التسريع مُفعّل',
        'acceleration_off': 'وضع التسريع غير مُفعّل',
        'save_state': 'تم حفظ الحالة',
        'resume_state': 'تم العثور على حالة محفوظة',
        'no_state': 'لا توجد حالة محفوظة',
        'pause': 'إيقاف مؤقت',
        'resume': 'استئناف',
        'paused': 'تم الإيقاف المؤقت',
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
        'contact_info': 'معلومات الاتصال',
        'email': 'البريد الإلكتروني',
        'phone': 'الهاتف',
        'fax': 'فاكس',
        'reset': 'إعادة تعيين',
        'delete_state': 'حذف الحالة المحفوظة',
        'error': 'خطأ',
        'reading_file': 'جاري قراءة الملف...',
        'detecting_format': 'جاري الكشف عن تنسيق الملف...',
        'format_detected': 'تم الكشف عن التنسيق: ',
        'trying_uflp': 'محاولة قراءة كـ UFLP...',
        'trying_ilp': 'محاولة قراءة كـ ILP...',
        'trying_gs250a': 'محاولة قراءة كـ gs250a-1...',
        'trying_mps': 'محاولة قراءة كـ MPS...',
        'trying_lp': 'محاولة قراءة كـ LP...',
        'success': 'تم التحميل بنجاح',
        'failed': 'فشلت جميع محاولات القراءة',
    },
    'English': {
        'app_title': 'AHRH: Integer/Mixed-Integer Programming & UFLP Solver',
        'app_desc': 'Solve ILP, MILP, and UFLP problems using the patented AHRH algorithm.',
        'note_uflp': 'Note: UFLP problems are automatically converted to ILP.',
        'sidebar_algo': 'Algorithm Parameters',
        'max_cycles': 'Max Cycles',
        'k_coarse': 'Coarse Set Size (k)',
        'patience': 'Patience',
        'sidebar_stop': 'Stopping Criteria',
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
        'tab_upload': 'Upload File',
        'tab_manual': 'Manual Input',
        'upload_header': 'Upload Problem File',
        'upload_info': 'Accepts any text file and tries to read it automatically (ILP, MILP, UFLP, MPS, LP, gs250a-1)',
        'choose_file': 'Choose a file',
        'supported_formats': 'All text files accepted - format auto-detected',
        'ilp_format_help': 'ILP: first line: n m, then c, then m A rows, then b',
        'uflp_format_help': 'UFLP: first line: n m 0, then n lines: f_i + m costs',
        'mps_format_help': 'MPS: Standard MPS format',
        'koerkel_help': 'gs250a-1: Auto-detected',
        'manual_header': 'Manual ILP Data Entry',
        'manual_warning': 'For small problems only (n ≤ 10, m ≤ 10)',
        'manual_n': 'Variables (n)',
        'manual_m': 'Constraints (m)',
        'manual_c': 'Objective coefficients c[i]',
        'manual_A': 'Constraint matrix A[i][j]',
        'manual_b': 'Right-hand side b[i]',
        'manual_f': 'Opening costs f[i]',
        'manual_costs': 'Transport costs c[i][j]',
        'solve_button': 'Solve Problem',
        'results': 'Results',
        'best_cost': 'Best Cost',
        'lp_val': 'LP Value',
        'gap': 'Gap (%)',
        'open_fac': 'Open facilities',
        'cycles_done': 'Cycles Done',
        'time': 'Time (s)',
        'size': 'Size',
        'stop_reason': 'Stop Reason',
        'gap_plot': 'Gap & R Evolution',
        'gap_label': 'Gap',
        'R_label': 'R',
        'cycle_log': 'Cycle Log',
        'cycle': 'Cycle',
        'cost': 'Cost',
        'improved': 'Improved',
        'best_so_far': 'Best so far',
        'yes': 'Yes',
        'no': 'No',
        'download': 'Download',
        'info_placeholder': 'Choose data source',
        'footer': 'Developed by Zakarya Benregreg - AHRH patented algorithm',
        'acceleration_on': 'Acceleration ON',
        'acceleration_off': 'Acceleration OFF',
        'save_state': 'State saved',
        'resume_state': 'Found saved state',
        'no_state': 'No saved state',
        'pause': 'Pause',
        'resume': 'Resume',
        'paused': 'Paused',
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
        'contact_info': 'Contact Information',
        'email': 'Email',
        'phone': 'Phone',
        'fax': 'Fax',
        'reset': 'Reset',
        'delete_state': 'Delete Saved State',
        'error': 'Error',
        'reading_file': 'Reading file...',
        'detecting_format': 'Detecting file format...',
        'format_detected': 'Format detected: ',
        'trying_uflp': 'Trying UFLP format...',
        'trying_ilp': 'Trying ILP format...',
        'trying_gs250a': 'Trying gs250a-1 format...',
        'trying_mps': 'Trying MPS format...',
        'trying_lp': 'Trying LP format...',
        'success': 'Loaded successfully',
        'failed': 'All reading attempts failed',
    },
    'Français': {
        'app_title': 'AHRH: Solveur PLNE/PLMNE et UFLP',
        'app_desc': 'Résoudre des problèmes PLNE, PLMNE et UFLP avec l\'algorithme AHRH breveté.',
        'note_uflp': 'Note: Les problèmes UFLP sont convertis automatiquement en PLNE.',
        'sidebar_algo': 'Paramètres',
        'max_cycles': 'Cycles max',
        'k_coarse': 'Taille ensemble grossier (k)',
        'patience': 'Patience',
        'sidebar_stop': "Critères d'arrêt",
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
        'tab_upload': 'Fichier',
        'tab_manual': 'Saisie',
        'upload_header': 'Charger fichier',
        'upload_info': 'Accepte tout fichier texte et essaie de le lire automatiquement (ILP, MILP, UFLP, MPS, LP, gs250a-1)',
        'choose_file': 'Choisir fichier',
        'supported_formats': 'Tous les fichiers texte acceptés - format auto-détecté',
        'ilp_format_help': 'ILP: 1ère ligne: n m, puis c, puis m lignes A, puis b',
        'uflp_format_help': 'UFLP: 1ère ligne: n m 0, puis n lignes: f_i + m coûts',
        'mps_format_help': 'MPS: Format MPS standard',
        'koerkel_help': 'gs250a-1: Auto-détecté',
        'manual_header': 'Saisie manuelle',
        'manual_warning': 'Petits problèmes seulement (n ≤ 10, m ≤ 10)',
        'manual_n': 'Variables (n)',
        'manual_m': 'Contraintes (m)',
        'manual_c': 'Coefficients c[i]',
        'manual_A': 'Matrice A[i][j]',
        'manual_b': 'Second membre b[i]',
        'manual_f': "Coûts d'ouverture f[i]",
        'manual_costs': 'Coûts transport c[i][j]',
        'solve_button': 'Résoudre',
        'results': 'Résultats',
        'best_cost': 'Meilleur coût',
        'lp_val': 'Valeur LP',
        'gap': 'Écart (%)',
        'open_fac': 'Sites ouverts',
        'cycles_done': 'Cycles',
        'time': 'Temps (s)',
        'size': 'Taille',
        'stop_reason': "Raison d'arrêt",
        'gap_plot': 'Évolution',
        'gap_label': 'Écart',
        'R_label': 'R',
        'cycle_log': 'Journal',
        'cycle': 'Cycle',
        'cost': 'Coût',
        'improved': 'Amélioré',
        'best_so_far': 'Meilleur',
        'yes': 'Oui',
        'no': 'Non',
        'download': 'Télécharger',
        'info_placeholder': 'Choisissez source',
        'footer': 'Développé par Zakarya Benregreg - Algorithme AHRH breveté',
        'acceleration_on': 'Mode accélération',
        'acceleration_off': 'Mode normal',
        'save_state': 'État sauvegardé',
        'resume_state': 'État trouvé',
        'no_state': "Pas d'état",
        'pause': 'Pause',
        'resume': 'Reprendre',
        'paused': 'En pause',
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
        'contact_info': 'Contact',
        'email': 'Email',
        'phone': 'Téléphone',
        'fax': 'Fax',
        'reset': 'Réinitialiser',
        'delete_state': "Supprimer l'état",
        'error': 'Erreur',
        'reading_file': 'Lecture du fichier...',
        'detecting_format': 'Détection du format...',
        'format_detected': 'Format détecté: ',
        'trying_uflp': 'Essai format UFLP...',
        'trying_ilp': 'Essai format ILP...',
        'trying_gs250a': 'Essai format gs250a-1...',
        'trying_mps': 'Essai format MPS...',
        'trying_lp': 'Essai format LP...',
        'success': 'Chargement réussi',
        'failed': 'Tous les essais ont échoué',
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
def try_read_as_gs250a(text):
    """محاولة قراءة الملف كـ gs250a-1"""
    try:
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
            return None
        
        n = int(numbers[0])
        m = int(numbers[1])
        if n <= 0 or m <= 0 or n > 1000 or m > 1000:
            return None
        
        idx = 2
        if idx + n <= len(numbers):
            c = np.array(numbers[idx:idx+n])
            idx += n
        else:
            return None
        
        A = np.zeros((m, n))
        for i in range(m):
            if idx + n <= len(numbers):
                A[i] = numbers[idx:idx+n]
                idx += n
            else:
                return None
        
        if idx + m <= len(numbers):
            b = np.array(numbers[idx:idx+m])
        else:
            return None
        
        return c, A, b, n, m
    except:
        return None

def try_read_as_uflp(text):
    """محاولة قراءة الملف كـ UFLP"""
    try:
        lines = text.strip().splitlines()
        clean_lines = []
        for line in lines:
            line = line.strip()
            if line and not line.startswith(('#', '!', '//', 'FILE:')):
                clean_lines.append(line)
        if len(clean_lines) < 2:
            return None
        
        parts = clean_lines[0].split()
        if len(parts) < 2:
            return None
        
        n = int(parts[0])
        m = int(parts[1])
        if n <= 0 or m <= 0 or n > 1000 or m > 1000:
            return None
        
        f = np.zeros(n)
        c = np.zeros((n, m))
        for i in range(n):
            if i+1 >= len(clean_lines):
                return None
            line = clean_lines[i+1]
            nums = list(map(float, line.split()))
            if len(nums) >= m+1:
                f[i] = nums[0]
                for j in range(m):
                    c[i, j] = nums[1+j]
            else:
                return None
        
        return f, c, n, m
    except:
        return None

def try_read_as_ilp(text):
    """محاولة قراءة الملف كـ ILP"""
    try:
        lines = text.strip().splitlines()
        clean_lines = []
        for line in lines:
            line = line.strip()
            if line and not line.startswith(('#', '!', '//', 'FILE:')):
                clean_lines.append(line)
        if len(clean_lines) < 3:
            return None
        
        parts = clean_lines[0].split()
        if len(parts) < 2:
            return None
        
        n = int(parts[0])
        m = int(parts[1])
        if n <= 0 or m <= 0 or n > 1000 or m > 1000:
            return None
        
        c = np.array(list(map(float, clean_lines[1].split())))
        if len(c) != n:
            return None
        
        A = np.zeros((m, n))
        for i in range(m):
            if 2+i >= len(clean_lines):
                return None
            row = list(map(float, clean_lines[2+i].split()))
            if len(row) != n:
                return None
            A[i] = row
        
        if 2+m >= len(clean_lines):
            return None
        b = np.array(list(map(float, clean_lines[2+m].split())))
        if len(b) != m:
            return None
        
        return c, A, b, n, m
    except:
        return None

def flexible_file_reader(text):
    """محاولة قراءة الملف بكل الصيغ الممكنة"""
    
    # محاولة gs250a-1 أولاً
    with st.spinner(t('trying_gs250a')):
        result = try_read_as_gs250a(text)
        if result is not None:
            c, A, b, n, m = result
            return c, A, b, n, m, None, 'gs250a'
    
    # محاولة UFLP
    with st.spinner(t('trying_uflp')):
        result = try_read_as_uflp(text)
        if result is not None:
            f, c, n, m = result
            obj, A, b, n_vars, n_constraints, n_y = uflp_to_ilp(f, c)
            uflp_info = {'n_y': n_y, 'f': f, 'c': c}
            return obj, A, b, n, m, uflp_info, 'uflp'
    
    # محاولة ILP
    with st.spinner(t('trying_ilp')):
        result = try_read_as_ilp(text)
        if result is not None:
            c, A, b, n, m = result
            return c, A, b, n, m, None, 'ilp'
    
    # إذا فشلت كل المحاولات
    raise ValueError(t('failed'))

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
        gap_history = [h['gap'] for h in history]
        R_history = [h['R'] for h in history]
        diff_history = [h.get('diff', 0) for h in history]
        last_cost = history[-1]['cost']
        last_gap = history[-1]['gap']
        last_R = history[-1]['R']
        last_x = np.array(history[-1].get('best_so_far', x))
        acceleration_active = history[-1].get('acceleration', False)

    progress_bar = st.progress(0)
    status_placeholder = st.empty()
    details_placeholder = st.empty()
    time_placeholder = st.empty()
    pause_button_placeholder = st.empty()

    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        for cycle in range(start_cycle, max_cycles+1):
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
            
            new_cost, new_x = vcycle(x, obj, A, b, coarse, x_lp, best_cost, uflp_info)
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
            
            progress = (cycle - start_cycle + 1) / (max_cycles - start_cycle + 1)
            progress_bar.progress(progress)
            
            elapsed = total_time + (time.time() - start_time)
            if cycle > start_cycle:
                avg_time = elapsed / (cycle - start_cycle + 1)
                remaining = avg_time * (max_cycles - cycle)
            else:
                remaining = 0
            
            status_placeholder.info(f"Cycle {cycle} / {max_cycles}")
            details_placeholder.markdown(
                f"""
                <div style="background-color: #fff3f3; padding: 15px; border-radius: 10px; border-left: 8px solid #ff4b4b; margin: 10px 0; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
                    <h4 style="color: #ff4b4b; margin: 0; font-size: 20px;">معلومات التقدم</h4>
                    <p style="font-size: 18px;"><b>التكلفة الحالية:</b> {new_cost:,.2f}</p>
                    <p style="font-size: 18px;"><b>الفجوة:</b> {gap:.4f}%</p>
                    <p style="font-size: 18px;"><b>R:</b> {R_val:.6f}</p>
                    <p style="font-size: 18px;"><b>تحسن:</b> {'✅' if improved else '❌'}</p>
                    <p style="font-size: 18px;"><b>أفضل تكلفة:</b> {best_cost:,.2f}</p>
                    <p style="font-size: 16px; color: #666;"><b>وقت الدورة:</b> {time.time() - (start_time + total_time - elapsed):.3f} ث</p>
                </div>
                """,
                unsafe_allow_html=True
            )
            time_placeholder.markdown(
                f"""
                <div style="background-color: #e3f2fd; padding: 15px; border-radius: 10px; border-left: 8px solid #2196f3; margin: 10px 0; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
                    <h4 style="color: #2196f3; margin: 0; font-size: 20px;">الوقت</h4>
                    <p style="font-size: 18px;"><b>{t('elapsed_time')}:</b> {elapsed:.2f} ث</p>
                    <p style="font-size: 18px;"><b>{t('estimated_remaining')}:</b> {remaining:.2f} ث</p>
                </div>
                """,
                unsafe_allow_html=True
            )
            
            if pause_button_placeholder.button(t('pause' if not st.session_state.paused else 'resume'), key=f"pause_{cycle}"):
                st.session_state.paused = not st.session_state.paused
                if st.session_state.paused:
                    st.info(t('paused'))
            
            if not acceleration_active and gap < 2.0 and R_val < 0.01:
                acceleration_active = True
                st.success(t('acceleration_on'))
            
            stop_now = False
            if no_improve >= patience:
                stop_reason = f"{t('patience')} ({patience})"
                stop_now = True
            if not stop_now and use_R and R_val < R_tol:
                if last_gap is not None and abs(gap - last_gap) < 1e-6:
                    stable_gap_count += 1
                    if stable_gap_count >= stable_gap_needed:
                        stop_reason = f"R < {R_tol}"
                        stop_now = True
                else:
                    stable_gap_count = 0
            if not stop_now and use_cost_repeat:
                if last_cost is not None and abs(new_cost - last_cost) < 1e-6:
                    cost_repeat_count += 1
                    if cost_repeat_count >= cost_repeat_times:
                        stop_reason = t('use_cost_repeat')
                        stop_now = True
                else:
                    cost_repeat_count = 0
            if not stop_now and use_gap_repeat:
                if last_gap is not None and abs(gap - last_gap) < 1e-6:
                    gap_repeat_count += 1
                    if gap_repeat_count >= gap_repeat_times:
                        stop_reason = t('use_gap_repeat')
                        stop_now = True
                else:
                    gap_repeat_count = 0
            if not stop_now and use_contraction:
                if diff < diff_tol and R_val < R_tol:
                    stop_reason = t('use_contraction')
                    stop_now = True
            
            last_cost = new_cost
            last_gap = gap
            last_R = R_val
            last_x = new_x.copy()
            
            if time.time() - last_save_time > 5:
                problem_data = {
                    'type': 'UFLP' if uflp_info else 'ILP',
                    'n': n,
                }
                save_state(STATE_FILE, cycle, best_cost, x, cycles_log, lp_val, elapsed, problem_data)
                last_save_time = time.time()
            
            if stop_now:
                cycles_done = cycle
                break
    
    if cycles_done == 0:
        cycles_done = max_cycles
        stop_reason = f"{t('max_cycles')} ({max_cycles})"
    
    total_time = elapsed
    
    progress_bar.empty()
    status_placeholder.empty()
    details_placeholder.empty()
    time_placeholder.empty()
    pause_button_placeholder.empty()
    
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
st.set_page_config(
    page_title="AHRH ILP/MILP/UFLP Solver", 
    page_icon="🧠", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 25px;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 25px;
        box-shadow: 0 10px 20px rgba(0,0,0,0.2);
    }
    .info-box {
        background-color: #e8f0fe;
        padding: 20px;
        border-radius: 10px;
        border-left: 8px solid #2196f3;
        margin: 15px 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .success-box {
        background-color: #e8f5e8;
        padding: 20px;
        border-radius: 10px;
        border-left: 8px solid #4caf50;
        margin: 15px 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .warning-box {
        background-color: #fff3e0;
        padding: 20px;
        border-radius: 10px;
        border-left: 8px solid #ff9800;
        margin: 15px 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .contact-box {
        background: linear-gradient(135deg, #ff6b6b 0%, #ff4757 100%);
        padding: 25px;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 20px 0;
        box-shadow: 0 10px 20px rgba(255,75,75,0.3);
        border: 3px solid white;
    }
    .contact-text {
        font-size: 28px;
        font-weight: bold;
        margin: 10px 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    .metric-card {
        background: white;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
        margin: 10px;
        transition: transform 0.3s;
    }
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.15);
    }
    .metric-value {
        font-size: 28px;
        font-weight: bold;
        color: #333;
    }
    .metric-label {
        font-size: 16px;
        color: #666;
        text-transform: uppercase;
    }
    .stButton>button {
        width: 100%;
        border-radius: 10px;
        height: 50px;
        font-size: 18px;
        font-weight: bold;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        transform: scale(1.05);
        box-shadow: 0 5px 15px rgba(0,0,0,0.3);
    }
</style>
""", unsafe_allow_html=True)

# Language selector
col1, col2, col3 = st.columns([3, 1, 1])
with col2:
    language = st.selectbox("🌐", ['English', 'Français', 'العربية'], key='language_selector')
    st.session_state.language = language

# Main header
st.markdown(f"""
<div class="main-header">
    <h1 style="font-size: 48px;">{t('app_title')}</h1>
    <p style="font-size: 20px;">{t('app_desc')}</p>
</div>
""", unsafe_allow_html=True)

# Contact information in red box
st.markdown(f"""
<div class="contact-box">
    <h2 style="color: white; font-size: 36px; margin-bottom: 20px;">{t('contact_info')}</h2>
    <div class="contact-text">✉️ {t('email')}: {CONTACT_EMAIL}</div>
    <div class="contact-text">📞 {t('phone')}: {CONTACT_PHONE}</div>
    <div class="contact-text">📠 {t('fax')}: {CONTACT_FAX}</div>
</div>
""", unsafe_allow_html=True)

st.caption(t('note_uflp'))

# Sidebar parameters
with st.sidebar:
    st.markdown(f"## {t('sidebar_algo')}")
    
    max_cycles = st.slider(t('max_cycles'), 10, 200, 50, 10)
    k_coarse = st.slider(t('k_coarse'), 3, 20, 10)
    patience = st.slider(t('patience'), 5, 50, 15)
    
    st.markdown("---")
    st.markdown(f"## {t('sidebar_stop')}")
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
        diff_tol = st.number_input(t('diff_tol'), value=1e-8, format="%.0e", step=1e-8)
    else:
        diff_tol = 1e-8
    
    st.markdown("---")
    st.write(f"{t('workers')}: {NUM_WORKERS}")
    
    if os.path.exists(STATE_FILE):
        if st.button(t('delete_state')):
            if delete_state(STATE_FILE):
                st.success(t('delete_state'))
                st.rerun()

# Problem type selection
problem_type = st.radio(
    t('select_problem_type'),
    [t('ilp'), t('milp'), t('uflp')],
    horizontal=True
)
is_uflp = (problem_type == t('uflp'))

# Main tabs
tab1, tab2 = st.tabs([t('tab_upload'), t('tab_manual')])

with tab1:
    st.markdown(f"""
    <div class="info-box">
        <h3>{t('upload_header')}</h3>
        <p>{t('upload_info')}</p>
        <p><small>{t('supported_formats')}</small></p>
    </div>
    """, unsafe_allow_html=True)
    
    with st.expander(f"📄 {t('supported_formats')}"):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**ILP/MILP:**\n{t('ilp_format_help')}")
            st.markdown(f"**UFLP:**\n{t('uflp_format_help')}")
        with col2:
            st.markdown(f"**MPS:**\n{t('mps_format_help')}")
            st.markdown(f"**gs250a-1:**\n{t('koerkel_help')}")
    
    # نقبل أي ملف بدون تحديد نوع
    uploaded_file = st.file_uploader(
        t('choose_file'), 
        type=None,  # نقبل أي نوع ملف
        help=t('supported_formats')
    )
    
    if uploaded_file is not None:
        file_details = {
            t('file_name'): uploaded_file.name,
            t('file_size'): f"{uploaded_file.size / 1024:.2f} KB"
        }
        st.json(file_details)
        
        if st.button(t('solve_button'), key="solve_upload"):
            with st.spinner(t('reading_file')):
                try:
                    text = uploaded_file.getvalue().decode("utf-8", errors='ignore')
                except:
                    text = uploaded_file.getvalue().decode("latin-1", errors='ignore')
                
                try:
                    with st.spinner(t('detecting_format')):
                        obj, A, b, n, m, uflp_info, detected_type = flexible_file_reader(text)
                    
                    # Problem info
                    st.markdown(f"""
                    <div class="success-box">
                        <h4>{t('problem_info')}</h4>
                        <p><b>{t('problem_type')}:</b> {detected_type.upper()}</p>
                        <p><b>{t('variables')}:</b> {n}</p>
                        <p><b>{t('constraints')}:</b> {m}</p>
                        <p><b>{t('format_detected')}:</b> {detected_type}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Check for saved state
                    if os.path.exists(STATE_FILE):
                        col1, col2 = st.columns(2)
                        with col1:
                            if st.button(t('resume')):
                                result = solve_ahrh(obj, A, b, uflp_info,
                                                    max_cycles, k_coarse, patience,
                                                    use_R, R_tol, stable_gap_needed,
                                                    use_cost_repeat, cost_repeat_times,
                                                    use_gap_repeat, gap_repeat_times,
                                                    use_contraction, diff_tol,
                                                    resume=True)
                                st.session_state.result = result
                                st.session_state.n = n
                                st.session_state.m = m
                                st.session_state.problem_type = detected_type
                                st.rerun()
                        with col2:
                            if st.button(t('reset')):
                                if os.path.exists(STATE_FILE):
                                    os.remove(STATE_FILE)
                                result = solve_ahrh(obj, A, b, uflp_info,
                                                    max_cycles, k_coarse, patience,
                                                    use_R, R_tol, stable_gap_needed,
                                                    use_cost_repeat, cost_repeat_times,
                                                    use_gap_repeat, gap_repeat_times,
                                                    use_contraction, diff_tol,
                                                    resume=False)
                                st.session_state.result = result
                                st.session_state.n = n
                                st.session_state.m = m
                                st.session_state.problem_type = detected_type
                                st.rerun()
                    else:
                        result = solve_ahrh(obj, A, b, uflp_info,
                                            max_cycles, k_coarse, patience,
                                            use_R, R_tol, stable_gap_needed,
                                            use_cost_repeat, cost_repeat_times,
                                            use_gap_repeat, gap_repeat_times,
                                            use_contraction, diff_tol,
                                            resume=False)
                        st.session_state.result = result
                        st.session_state.n = n
                        st.session_state.m = m
                        st.session_state.problem_type = detected_type
                        st.rerun()
                
                except Exception as e:
                    st.error(f"{t('error')}: {str(e)}")
                    st.code(traceback.format_exc())

with tab2:
    st.markdown(f"""
    <div class="warning-box">
        <h3>{t('manual_header')}</h3>
        <p>{t('manual_warning')}</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        n_man = st.number_input(t('manual_n'), min_value=1, max_value=10, value=3, step=1, key="n_man")
    with col2:
        m_man = st.number_input(t('manual_m'), min_value=1, max_value=10, value=3, step=1, key="m_man")
    
    if is_uflp:
        # UFLP manual input
        st.subheader(t('manual_f'))
        if 'f_man' not in st.session_state or len(st.session_state.f_man) != n_man:
            st.session_state.f_man = np.ones(n_man) * 1000
        f_vals = []
        cols = st.columns(min(5, n_man))
        for i in range(n_man):
            with cols[i % 5]:
                val = st.number_input(f"f[{i}]", value=float(st.session_state.f_man[i]), key=f"f_man_{i}")
                f_vals.append(val)
        st.session_state.f_man = np.array(f_vals)
        
        st.subheader(t('manual_costs'))
        if 'c_man' not in st.session_state or st.session_state.c_man.shape != (n_man, m_man):
            st.session_state.c_man = np.ones((n_man, m_man)) * 100
        for i in range(n_man):
            st.write(f"**الموقع {i+1}:**")
            cols = st.columns(min(5, m_man))
            for j in range(m_man):
                with cols[j % 5]:
                    val = st.number_input(f"c[{i}][{j}]", value=float(st.session_state.c_man[i, j]), key=f"c_man_{i}_{j}")
                    st.session_state.c_man[i, j] = val
    else:
        # ILP/MILP manual input
        st.subheader(t('manual_c'))
        if 'c_man' not in st.session_state or len(st.session_state.c_man) != n_man:
            st.session_state.c_man = np.ones(n_man)
        c_vals = []
        cols = st.columns(min(5, n_man))
        for i in range(n_man):
            with cols[i % 5]:
                val = st.number_input(f"c[{i}]", value=float(st.session_state.c_man[i]), key=f"c_man_{i}")
                c_vals.append(val)
        st.session_state.c_man = np.array(c_vals)
        
        st.subheader(t('manual_A'))
        if 'A_man' not in st.session_state or st.session_state.A_man.shape != (m_man, n_man):
            st.session_state.A_man = np.ones((m_man, n_man))
        for i in range(m_man):
            st.write(f"**{t('constraints')} {i+1}:**")
            cols = st.columns(min(5, n_man))
            for j in range(n_man):
                with cols[j % 5]:
                    val = st.number_input(f"A[{i}][{j}]", value=float(st.session_state.A_man[i, j]), key=f"A_man_{i}_{j}")
                    st.session_state.A_man[i, j] = val
        
        st.subheader(t('manual_b'))
        if 'b_man' not in st.session_state or len(st.session_state.b_man) != m_man:
            st.session_state.b_man = np.ones(m_man) * 10
        b_vals = []
        cols = st.columns(min(5, m_man))
        for i in range(m_man):
            with cols[i % 5]:
                val = st.number_input(f"b[{i}]", value=float(st.session_state.b_man[i]), key=f"b_man_{i}")
                b_vals.append(val)
        st.session_state.b_man = np.array(b_vals)
    
    if st.button(t('solve_button'), key="solve_manual"):
        with st.spinner(t('solve_button')):
            try:
                if is_uflp:
                    f = st.session_state.f_man
                    c = st.session_state.c_man
                    obj, A, b, n_vars, n_constraints, n_y = uflp_to_ilp(f, c)
                    uflp_info = {'n_y': n_y, 'f': f, 'c': c}
                    detected_type = 'uflp'
                else:
                    obj = st.session_state.c_man
                    A = st.session_state.A_man
                    b = st.session_state.b_man
                    uflp_info = None
                    detected_type = 'ilp'
                
                if os.path.exists(STATE_FILE):
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button(t('resume'), key="resume_manual"):
                            result = solve_ahrh(obj, A, b, uflp_info,
                                                max_cycles, k_coarse, patience,
                                                use_R, R_tol, stable_gap_needed,
                                                use_cost_repeat, cost_repeat_times,
                                                use_gap_repeat, gap_repeat_times,
                                                use_contraction, diff_tol,
                                                resume=True)
                            st.session_state.result = result
                            st.session_state.n = n_man
                            st.session_state.m = m_man
                            st.session_state.problem_type = detected_type
                            st.rerun()
                    with col2:
                        if st.button(t('reset'), key="reset_manual"):
                            if os.path.exists(STATE_FILE):
                                os.remove(STATE_FILE)
                            result = solve_ahrh(obj, A, b, uflp_info,
                                                max_cycles, k_coarse, patience,
                                                use_R, R_tol, stable_gap_needed,
                                                use_cost_repeat, cost_repeat_times,
                                                use_gap_repeat, gap_repeat_times,
                                                use_contraction, diff_tol,
                                                resume=False)
                            st.session_state.result = result
                            st.session_state.n = n_man
                            st.session_state.m = m_man
                            st.session_state.problem_type = detected_type
                            st.rerun()
                else:
                    result = solve_ahrh(obj, A, b, uflp_info,
                                        max_cycles, k_coarse, patience,
                                        use_R, R_tol, stable_gap_needed,
                                        use_cost_repeat, cost_repeat_times,
                                        use_gap_repeat, gap_repeat_times,
                                        use_contraction, diff_tol,
                                        resume=False)
                    st.session_state.result = result
                    st.session_state.n = n_man
                    st.session_state.m = m_man
                    st.session_state.problem_type = detected_type
                    st.rerun()
            
            except Exception as e:
                st.error(f"{t('error')}: {str(e)}")
                st.code(traceback.format_exc())

# ------------------- Results Display -------------------
st.markdown("---")

if st.session_state.get('result') is not None:
    res = st.session_state.result
    
    st.markdown(f"""
    <div class="success-box">
        <h2>{t('results')}</h2>
    </div>
    """, unsafe_allow_html=True)
    
    # Metrics in cards
    colA, colB, colC, colD = st.columns(4)
    with colA:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{res['best_cost']:,.2f}</div>
            <div class="metric-label">{t('best_cost')}</div>
        </div>
        """, unsafe_allow_html=True)
    with colB:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{res['lp_val']:,.2f}</div>
            <div class="metric-label">{t('lp_val')}</div>
        </div>
        """, unsafe_allow_html=True)
    with colC:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{res['gap']:.4f}%</div>
            <div class="metric-label">{t('gap')}</div>
        </div>
        """, unsafe_allow_html=True)
    with colD:
        if res.get('open_fac') is not None:
            val = res['open_fac']
            label = t('open_fac')
        else:
            val = f"{st.session_state.get('n', 0)}×{st.session_state.get('m', 0)}"
            label = t('size')
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{val}</div>
            <div class="metric-label">{label}</div>
        </div>
        """, unsafe_allow_html=True)
    
    colE, colF, colG = st.columns(3)
    with colE:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{res['cycles_done']}</div>
            <div class="metric-label">{t('cycles_done')}</div>
        </div>
        """, unsafe_allow_html=True)
    with colF:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{res['total_time']:.2f}s</div>
            <div class="metric-label">{t('time')}</div>
        </div>
        """, unsafe_allow_html=True)
    with colG:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{res['stop_reason']}</div>
            <div class="metric-label">{t('stop_reason')}</div>
        </div>
        """, unsafe_allow_html=True)
    
    if res.get('acceleration_active'):
        st.success(t('acceleration_on'))
    
    # Plots
    if res['gap_history']:
        st.subheader(t('gap_plot'))
        
        plot_tab1, plot_tab2, plot_tab3 = st.tabs([t('gap_by_cycles'), t('gap_by_R'), t('convergence_plot')])
        
        with plot_tab1:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
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
            ax2.set_title(f'R {t("vs")} {t("cycle")}')
            ax2.grid(True, alpha=0.3)
            ax2.legend()
            
            plt.tight_layout()
            st.pyplot(fig)
        
        with plot_tab2:
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Gap vs R scatter with color gradient
            scatter = ax.scatter(res['R_history'], res['gap_history'], 
                               c=cycles, cmap='viridis', s=100, alpha=0.6, edgecolors='black', linewidth=0.5)
            ax.set_xlabel(t('R_label'))
            ax.set_ylabel(t('gap_label'))
            ax.set_title(t('R_vs_gap'))
            ax.grid(True, alpha=0.3)
            plt.colorbar(scatter, ax=ax, label=t('cycle'))
            
            # Trend line
            if len(res['R_history']) > 1:
                z = np.polyfit(res['R_history'], res['gap_history'], 1)
                p = np.poly1d(z)
                ax.plot(res['R_history'], p(res['R_history']), "r--", alpha=0.8, linewidth=2, label='Trend')
                ax.legend()
            
            plt.tight_layout()
            st.pyplot(fig)
        
        with plot_tab3:
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Convergence plot with log scale
            ax.semilogy(cycles, res['gap_history'], 'b-', linewidth=2, label=t('gap_label'))
            ax.semilogy(cycles, res['R_history'], 'r-', linewidth=2, label=t('R_label'))
            ax.set_xlabel(t('cycle'))
            ax.set_ylabel('Value (log scale)')
            ax.set_title(t('convergence_plot'))
            ax.grid(True, alpha=0.3, which='both')
            ax.legend()
            
            plt.tight_layout()
            st.pyplot(fig)
        
        # Cycle log
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
        
        # Download
        df = pd.DataFrame({
            t('cycle'): cycles,
            t('gap_label'): res['gap_history'],
            'R': res['R_history']
        })
        csv = df.to_csv(index=False)
        st.download_button(
            label=t('download'),
            data=csv,
            file_name=f"ahrh_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
else:
    st.markdown(f"""
    <div class="info-box">
        <h3>👈 {t('info_placeholder')}</h3>
        <p>{t('supported_formats')}</p>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown(f"""
<div style="text-align: center; color: #666; padding: 20px; background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%); border-radius: 15px;">
    <p style="font-size: 18px;">{t('footer')}</p>
    <p style="font-size: 16px;">🔬 AHRH Algorithm - Patented Technology | {CONTACT_EMAIL} | {CONTACT_PHONE}</p>
    <p style="font-size: 14px;">© 2024 Zakarya Benregreg. All rights reserved.</p>
</div>
""", unsafe_allow_html=True)
