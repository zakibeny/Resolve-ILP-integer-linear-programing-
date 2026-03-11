# -*- coding: utf-8 -*-
"""
AHRH: حل مسائل البرمجة الصحيحة (ILP/MILP) ومسائل مواقع المرافق (UFLP)
مع واجهة متعددة اللغات، عرض التقدم، معايير توقف متعددة، وحفظ الحالة
باستخدام النموذج الموحد (make_problem) ودوال التقييم المقدمة.
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
import uuid
import math

warnings.filterwarnings("ignore")

# ------------------- إعدادات -------------------
NUM_WORKERS = 4
STATE_FILE = "ahrh_state.json"
STATE_DIR = "checkpoints"
os.makedirs(STATE_DIR, exist_ok=True)

# ------------------- ترجمة النصوص -------------------
translations = {
    'العربية': {
        'app_title': 'AHRH: حلول البرمجة الصحيحة (ILP/MILP) و UFLP',
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
        'use_R_stability': 'استخدام استقرار R (التغير النسبي)',
        'R_stability_tol': 'عتبة التغير النسبي لـ R',
        'R_stability_cycles': 'عدد دورات استقرار R',
        'workers': 'عدد العمال (للتوازي)',
        'tab_upload': '📂 رفع ملف',
        'tab_manual': '✍️ إدخال يدوي',
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
        'manual_constraint_type': 'نوع القيد',
        'constraint_type_leq': '≤',
        'constraint_type_eq': '=',
        'constraint_type_geq': '≥',
        'solve_button': '🚀 حل المسألة',
        'results': '📊 النتائج',
        'problem_info': 'معلومات المسألة',
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
        'sidebar_algo': '⚙️ Algorithm Parameters',
        'max_cycles': 'Max Cycles',
        'k_coarse': 'Coarse Set Size (k)',
        'patience': 'Patience',
        'sidebar_stop': '⏹️ Stopping Criteria',
        'choose_criteria': 'Choose any combination:',
        'use_R': 'Use R threshold (with gap stability)',
        'R_tol': 'R tolerance (ε)',
        'stable_gap': 'Stable gap cycles',
        'use_cost_repeat': 'Use cost repetition',
        'cost_repeat_times': 'Repetition count',
        'use_gap_repeat': 'Use gap repetition',
        'gap_repeat_times': 'Repetition count',
        'use_contraction': 'Use contraction criterion (diff + R)',
        'diff_tol': 'Difference tolerance (ε₁)',
        'use_R_stability': 'Use R relative change stability',
        'R_stability_tol': 'R relative change tolerance',
        'R_stability_cycles': 'R stability cycles',
        'workers': 'Workers',
        'tab_upload': '📂 Upload File',
        'tab_manual': '✍️ Manual Input',
        'upload_header': 'Upload Problem File',
        'upload_info': 'Accepts any text file and tries to read it automatically (ILP, MILP, UFLP, MPS, LP, gs250a-1)',
        'choose_file': 'Choose a file',
        'supported_formats': 'All text files accepted - format auto-detected',
        'ilp_format_help': 'ILP: first line: n m, then c, then m A rows, then b',
        'uflp_format_help': 'UFLP: first line: n m 0, then n lines: f_i + m costs',
        'mps_format_help': 'MPS: Standard MPS format',
        'koerkel_help': 'gs250a-1: Auto-detected',
        'manual_header': 'Manual Problem Data Entry',
        'manual_warning': 'For small problems only (n ≤ 10, m ≤ 10)',
        'manual_n': 'Variables (n)',
        'manual_m': 'Constraints (m)',
        'manual_c': 'Objective coefficients c[i]',
        'manual_A': 'Constraint matrix A[i][j]',
        'manual_b': 'Right-hand side b[i]',
        'manual_f': 'Opening costs f[i]',
        'manual_costs': 'Transport costs c[i][j]',
        'manual_constraint_type': 'Constraint type',
        'constraint_type_leq': '≤',
        'constraint_type_eq': '=',
        'constraint_type_geq': '≥',
        'solve_button': '🚀 Solve Problem',
        'results': '📊 Results',
        'problem_info': 'Problem Information',
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
        'download': '📥 Download Evolution',
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
        'sidebar_algo': '⚙️ Paramètres',
        'max_cycles': 'Cycles max',
        'k_coarse': 'Taille ensemble grossier (k)',
        'patience': 'Patience',
        'sidebar_stop': '⏹️ Critères d\'arrêt',
        'choose_criteria': 'Choisissez une combinaison:',
        'use_R': 'Utiliser seuil R (avec stabilité de l\'écart)',
        'R_tol': 'Tolérance R (ε)',
        'stable_gap': 'Cycles de stabilité de l\'écart',
        'use_cost_repeat': 'Répétition du coût',
        'cost_repeat_times': 'Nombre de répétitions',
        'use_gap_repeat': 'Répétition de l\'écart',
        'gap_repeat_times': 'Nombre de répétitions',
        'use_contraction': 'Critère de contraction (diff + R)',
        'diff_tol': 'Tolérance de différence (ε₁)',
        'use_R_stability': 'Stabilité du changement relatif de R',
        'R_stability_tol': 'Tolérance du changement relatif',
        'R_stability_cycles': 'Cycles de stabilité',
        'workers': 'Travailleurs',
        'tab_upload': '📂 Fichier',
        'tab_manual': '✍️ Saisie',
        'upload_header': 'Charger fichier',
        'upload_info': 'Accepte tout fichier texte et essaie de le lire automatiquement (ILP, MILP, UFLP, MPS, LP, gs250a-1)',
        'choose_file': 'Choisir fichier',
        'supported_formats': 'Tous les fichiers texte acceptés - format auto-détecté',
        'ilp_format_help': 'ILP: 1ère ligne: n m, puis c, puis m lignes A, puis b',
        'uflp_format_help': 'UFLP: 1ère ligne: n m 0, puis n lignes: f_i + m coûts',
        'mps_format_help': 'MPS: Format MPS standard',
        'koerkel_help': 'gs250a-1: Auto-détecté',
        'manual_header': 'Saisie manuelle des données',
        'manual_warning': 'Pour petits problèmes seulement (n ≤ 10, m ≤ 10)',
        'manual_n': 'Variables (n)',
        'manual_m': 'Contraintes (m)',
        'manual_c': 'Coefficients objectifs c[i]',
        'manual_A': 'Matrice des contraintes A[i][j]',
        'manual_b': 'Second membre b[i]',
        'manual_f': 'Coûts d\'ouverture f[i]',
        'manual_costs': 'Coûts de transport c[i][j]',
        'manual_constraint_type': 'Type de contrainte',
        'constraint_type_leq': '≤',
        'constraint_type_eq': '=',
        'constraint_type_geq': '≥',
        'solve_button': '🚀 Résoudre',
        'results': '📊 Résultats',
        'problem_info': 'Information',
        'best_cost': 'Meilleur coût',
        'lp_val': 'Valeur LP',
        'gap': 'Écart (%)',
        'open_fac': 'Sites ouverts',
        'cycles_done': 'Cycles effectués',
        'time': 'Temps (s)',
        'size': 'Taille',
        'stop_reason': 'Raison d\'arrêt',
        'gap_plot': '📈 Évolution',
        'gap_label': 'Écart',
        'R_label': 'R',
        'cycle_log': '📋 Journal',
        'cycle': 'Cycle',
        'cost': 'Coût',
        'improved': 'Amélioré',
        'best_so_far': 'Meilleur jusqu\'à présent',
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
        'estimated_remaining': 'Temps restant estimé',
        'gap_by_cycles': 'Écart par cycles',
        'gap_by_R': 'Écart par R',
        'convergence_plot': 'Graphique de convergence',
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
        'delete_state': '🗑️ Supprimer l\'état',
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

# ------------------- إدارة الجلسة -------------------
if 'language' not in st.session_state:
    st.session_state.language = 'English'
if 'paused' not in st.session_state:
    st.session_state.paused = False
if 'stop_requested' not in st.session_state:
    st.session_state.stop_requested = False
if 'issue_token' not in st.session_state:
    # Générer un token unique pour cette session
    st.session_state.issue_token = str(uuid.uuid4())

def ensure_issue_token():
    if 'issue_token' not in st.session_state:
        st.session_state.issue_token = str(uuid.uuid4())
    return st.session_state.issue_token

# ------------------- معلومات الاتصال -------------------
CONTACT_EMAIL = "zakibeny@gmail.com"
CONTACT_PHONE = "0021355614305 / 00213779679073"
CONTACT_FAX = "036495241"

# =========================
# 1) نموذج داخلي موحّد (من المستخدم)
# =========================
def make_problem(obj, A, b, senses, var_types, lb=None, ub=None,
                 original_type="ILP", meta=None):
    obj = np.asarray(obj, dtype=float).reshape(-1)
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float).reshape(-1)

    n = len(obj)
    m = len(b)

    if A.shape != (m, n):
        raise ValueError(f"A shape must be ({m}, {n}), got {A.shape}")
    if len(senses) != m:
        raise ValueError("senses length must equal number of constraints")
    if len(var_types) != n:
        raise ValueError("var_types length must equal number of variables")

    lb = np.zeros(n, dtype=float) if lb is None else np.asarray(lb, dtype=float).reshape(-1)
    if ub is None:
        ub = np.array([1.0 if vt == "B" else np.inf for vt in var_types], dtype=float)
    else:
        ub = np.asarray(ub, dtype=float).reshape(-1)

    return {
        "obj": obj,
        "A": A,
        "b": b,
        "senses": list(senses),        # <=, >=, =
        "var_types": list(var_types),  # B, I, C
        "lb": lb,
        "ub": ub,
        "original_type": original_type,
        "meta": meta or {},
    }

def uflp_to_ilp(f, c):
    """
    UFLP -> ILP canonical model
    y_i : open facility i   (binary)
    x_ij: customer j assigned to facility i (binary)
    min sum f_i y_i + sum c_ij x_ij
    s.t. sum_i x_ij = 1      for each customer j
         x_ij - y_i <= 0     for each i,j
    """
    f = np.asarray(f, dtype=float).reshape(-1)
    c = np.asarray(c, dtype=float)
    n = len(f)          # facilities
    m = c.shape[1]      # customers

    n_vars = n + n * m
    obj = np.concatenate([f, c.reshape(-1)])

    A = []
    b = []
    senses = []

    # assignment constraints: each customer assigned exactly once
    for j in range(m):
        row = np.zeros(n_vars, dtype=float)
        for i in range(n):
            row[n + i * m + j] = 1.0
        A.append(row)
        b.append(1.0)
        senses.append("=")

    # linking constraints: x_ij <= y_i
    for i in range(n):
        for j in range(m):
            row = np.zeros(n_vars, dtype=float)
            row[n + i * m + j] = 1.0
            row[i] = -1.0
            A.append(row)
            b.append(0.0)
            senses.append("<=")

    var_types = ["B"] * n_vars
    lb = np.zeros(n_vars, dtype=float)
    ub = np.ones(n_vars, dtype=float)

    return make_problem(
        obj=obj,
        A=np.array(A),
        b=np.array(b),
        senses=senses,
        var_types=var_types,
        lb=lb,
        ub=ub,
        original_type="UFLP",
        meta={"n_facilities": n, "n_customers": m},
    )

# =========================
# 2) تقييم عام صحيح لـ ILP/MILP/UFLP
# =========================
def _is_discrete(vt):
    return vt in ("B", "I")

def _bound_or_none(v):
    return None if not np.isfinite(v) else float(v)

def _sanitize_value(v, lo, hi, vartype):
    if vartype == "B":
        return float(int(np.clip(np.round(v), 0, 1)))
    if vartype == "I":
        lo2 = -1e18 if not np.isfinite(lo) else lo
        hi2 =  1e18 if not np.isfinite(hi) else hi
        return float(int(np.clip(np.round(v), lo2, hi2)))
    # Continuous
    lo2 = -1e18 if not np.isfinite(lo) else lo
    hi2 =  1e18 if not np.isfinite(hi) else hi
    return float(np.clip(v, lo2, hi2))

def _add_constraint(prob, expr, sense, rhs):
    if sense == "<=":
        prob += expr <= rhs
    elif sense == ">=":
        prob += expr >= rhs
    elif sense == "=":
        prob += expr == rhs
    else:
        raise ValueError(f"Unknown constraint sense: {sense}")

def _check_feasibility_numeric(x, problem, tol=1e-7):
    lhs = problem["A"] @ x
    for j, sense in enumerate(problem["senses"]):
        rhs = problem["b"][j]
        if sense == "<=" and lhs[j] > rhs + tol:
            return False
        if sense == ">=" and lhs[j] < rhs - tol:
            return False
        if sense == "=" and abs(lhs[j] - rhs) > tol:
            return False
    return True

def evaluate_solution(x, problem, tol=1e-7):
    """
    - للمتغيرات B/I: تثبيت بالتقريب والقص على الحدود.
    - للمتغيرات C في MILP: نعيد حل الجزء المستمر ببرنامج خطي بعد تثبيت الجزء المتقطع.
    """
    x = np.asarray(x, dtype=float).reshape(-1)
    n = len(problem["obj"])

    if len(x) != n:
        raise ValueError("Candidate vector has wrong size")

    vt = problem["var_types"]
    lb = problem["lb"]
    ub = problem["ub"]
    obj = problem["obj"]

    fixed_vals = [None] * n
    cont_idx = []

    for i in range(n):
        if _is_discrete(vt[i]):
            fixed_vals[i] = _sanitize_value(x[i], lb[i], ub[i], vt[i])
        else:
            cont_idx.append(i)

    # Pure ILP / pure binary path
    if not cont_idx:
        x_full = np.array(fixed_vals, dtype=float)
        if not _check_feasibility_numeric(x_full, problem, tol=tol):
            return float("inf"), None
        return float(obj @ x_full), x_full

    # MILP path: discrete vars fixed, continuous vars re-optimized
    prob = pulp.LpProblem("MILP_EVAL", pulp.LpMinimize)
    vars_lp = []

    for i in range(n):
        if _is_discrete(vt[i]):
            vars_lp.append(fixed_vals[i])
        else:
            vars_lp.append(
                pulp.LpVariable(
                    f"x_{i}",
                    lowBound=_bound_or_none(lb[i]),
                    upBound=_bound_or_none(ub[i]),
                    cat="Continuous"
                )
            )

    prob += pulp.lpSum(obj[i] * vars_lp[i] for i in range(n))

    for j, sense in enumerate(problem["senses"]):
        expr = pulp.lpSum(problem["A"][j, i] * vars_lp[i] for i in range(n))
        _add_constraint(prob, expr, sense, float(problem["b"][j]))

    solver = pulp.PULP_CBC_CMD(msg=False)
    prob.solve(solver)

    if prob.status != pulp.LpStatusOptimal:
        return float("inf"), None

    x_full = np.array([
        float(vars_lp[i]) if _is_discrete(vt[i]) else float(pulp.value(vars_lp[i]))
        for i in range(n)
    ], dtype=float)

    return float(pulp.value(prob.objective)), x_full

def lp_relaxation(problem):
    """
    Relax all B/I vars to Continuous, while respecting:
    - bounds
    - senses <= >= =
    """
    n = len(problem["obj"])
    m = len(problem["b"])

    prob = pulp.LpProblem("LP_Relax", pulp.LpMinimize)
    x = [
        pulp.LpVariable(
            f"x_{i}",
            lowBound=_bound_or_none(problem["lb"][i]),
            upBound=_bound_or_none(problem["ub"][i]),
            cat="Continuous"
        )
        for i in range(n)
    ]

    prob += pulp.lpSum(problem["obj"][i] * x[i] for i in range(n))

    for j in range(m):
        expr = pulp.lpSum(problem["A"][j, i] * x[i] for i in range(n))
        _add_constraint(prob, expr, problem["senses"][j], float(problem["b"][j]))

    solver = pulp.PULP_CBC_CMD(msg=False)
    prob.solve(solver)

    if prob.status != pulp.LpStatusOptimal:
        return None, None

    x_val = np.array([float(pulp.value(x[i])) for i in range(n)], dtype=float)
    obj_val = float(pulp.value(prob.objective))
    return x_val, obj_val

# =========================
# 3) الكسور و R يجب أن تعمل على B/I فقط
# =========================
def get_fractional_indices(x, problem, eps=1e-6):
    x = np.asarray(x, dtype=float)
    idx = []
    for i, vt in enumerate(problem["var_types"]):
        if vt in ("B", "I"):
            if abs(x[i] - np.round(x[i])) > eps:
                idx.append(i)
    return np.array(idx, dtype=int)

def compute_R(x, problem):
    x = np.asarray(x, dtype=float)
    vals = []
    for i, vt in enumerate(problem["var_types"]):
        if vt in ("B", "I"):
            vals.append(abs(x[i] - np.round(x[i])))
    return max(vals) if vals else 0.0

def local_search(x, best_cost, problem):
    x = np.asarray(x, dtype=float).copy()
    best_x = x.copy()
    best = best_cost

    for i, vt in enumerate(problem["var_types"]):
        candidates = []

        if vt == "B":
            candidates = [1.0 - round(best_x[i])]
        elif vt == "I":
            candidates = [best_x[i] - 1.0, best_x[i] + 1.0]
        else:
            continue  # لا نقلب المتغيرات المستمرة هنا

        for cand in candidates:
            x_new = best_x.copy()
            x_new[i] = _sanitize_value(cand, problem["lb"][i], problem["ub"][i], vt)
            cost, x_feas = evaluate_solution(x_new, problem)
            if cost < best and x_feas is not None:
                best = cost
                best_x = x_feas.copy()

    return best, best_x

# =========================
# 4) دوال المسح الشعاعي و V‑cycle (معدلة لاستخدام problem)
# =========================
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

def hierarchical_radial_scan(x_center, R_val, frac_idx, problem, best_cost, best_x, x_lp=None):
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
            cost, x_feas = evaluate_solution(x_full, problem)
            if cost < local_best and x_feas is not None:
                local_best = cost
                local_best_x = x_feas.copy()
    return local_best, local_best_x

def vcycle(x, problem, coarse, x_lp, best_cost):
    frac_idx = get_fractional_indices(x, problem)
    R_val = compute_R(x, problem)
    new_cost, new_x = hierarchical_radial_scan(x, R_val, frac_idx, problem, best_cost, x, x_lp)
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
            cost, x_feas = evaluate_solution(x_full, problem)
            if cost < best_coarse and x_feas is not None:
                best_coarse = cost
                best_x_coarse = x_feas
        if best_coarse < best_cost:
            best_cost = best_coarse
            x = best_x_coarse
    best_cost, x = local_search(x, best_cost, problem)
    return best_cost, x

# =========================
# 5) حفظ checkpoint مع issue_token
# =========================
def _json_safe_value(v):
    if isinstance(v, (np.floating, float)):
        return None if not np.isfinite(v) else float(v)
    if isinstance(v, (np.integer, int)):
        return int(v)
    return v

def _json_safe_array(a):
    return [_json_safe_value(v) for v in np.asarray(a).tolist()]

def serialize_problem(problem):
    return {
        "obj": _json_safe_array(problem["obj"]),
        "A": [_json_safe_array(row) for row in np.asarray(problem["A"])],
        "b": _json_safe_array(problem["b"]),
        "senses": list(problem["senses"]),
        "var_types": list(problem["var_types"]),
        "lb": _json_safe_array(problem["lb"]),
        "ub": _json_safe_array(problem["ub"]),
        "original_type": problem["original_type"],
        "meta": problem["meta"],
    }

def build_checkpoint(problem, runtime_info=None):
    ensure_issue_token()
    return {
        "issue_token": st.session_state.issue_token,
        "saved_at": datetime.utcnow().isoformat() + "Z",
        "language": st.session_state.get("language", "English"),
        "problem": serialize_problem(problem),
        "runtime_info": runtime_info or {},
    }

def save_checkpoint_local(state):
    path = os.path.join(STATE_DIR, f"ahrh_{state['issue_token']}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)
    return path

# ------------------- State Management (Legacy) -------------------
def save_state(filename, cycle, best_cost, best_x, history, lp_val, total_time, problem_data):
    # نحتفظ بالحالة القديمة للتوافق، ولكن نضيفها إلى checkpoint الجديد
    runtime_info = {
        'cycle': cycle,
        'best_cost': best_cost,
        'best_x': best_x.tolist(),
        'history': history,
        'lp_val': lp_val,
        'total_time': total_time,
    }
    checkpoint = build_checkpoint(problem_data.get('problem'), runtime_info)
    # نستخدم اسم ملف موحّد
    with open(filename, 'w') as f:
        json.dump(checkpoint, f)

def load_state(filename):
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            state = json.load(f)
        # نحاول استخراج best_x من runtime_info
        if 'runtime_info' in state and 'best_x' in state['runtime_info']:
            state['best_x'] = np.array(state['runtime_info']['best_x'])
        else:
            return None
        return state
    return None

def delete_state(filename):
    if os.path.exists(filename):
        os.remove(filename)
        return True
    return False

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
        
        # نصنع problem من النوع ILP (افتراض أن القيود ≤ والمتغيرات صحيحة)
        senses = ['<='] * m
        var_types = ['I'] * n
        problem = make_problem(c, A, b, senses, var_types)
        return problem, 'gs250a'
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
        
        problem = uflp_to_ilp(f, c)
        return problem, 'uflp'
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
        
        # محاولة قراءة c (قد تكون في عدة أسطر)
        c_vals = []
        idx = 1
        while len(c_vals) < n and idx < len(clean_lines):
            c_vals.extend(list(map(float, clean_lines[idx].split())))
            idx += 1
        if len(c_vals) < n:
            return None
        c = np.array(c_vals[:n])
        
        # قراءة A
        A = np.zeros((m, n))
        for i in range(m):
            if idx >= len(clean_lines):
                return None
            row_vals = list(map(float, clean_lines[idx].split()))
            if len(row_vals) < n:
                # قد يمتد السطر
                while len(row_vals) < n and idx+1 < len(clean_lines):
                    idx += 1
                    row_vals.extend(list(map(float, clean_lines[idx].split())))
            if len(row_vals) >= n:
                A[i] = row_vals[:n]
                idx += 1
            else:
                return None
        
        # قراءة b
        if idx < len(clean_lines):
            b_vals = list(map(float, clean_lines[idx].split()))
        else:
            return None
        if len(b_vals) < m:
            # قد يمتد
            while len(b_vals) < m and idx+1 < len(clean_lines):
                idx += 1
                b_vals.extend(list(map(float, clean_lines[idx].split())))
        if len(b_vals) >= m:
            b = np.array(b_vals[:m])
        else:
            return None
        
        # نفترض أن القيود من نوع ≤ والمتغيرات صحيحة
        senses = ['<='] * m
        var_types = ['I'] * n
        problem = make_problem(c, A, b, senses, var_types)
        return problem, 'ilp'
    except:
        return None

def flexible_file_reader(text, filename=""):
    """محاولة قراءة الملف بكل الصيغ الممكنة"""
    
    # محاولة gs250a-1 أولاً
    with st.spinner(t('trying_gs250a')):
        result = try_read_as_gs250a(text)
        if result is not None:
            problem, detected_type = result
            st.success(f"{t('success')}: {detected_type}")
            return problem, detected_type
    
    # محاولة UFLP
    with st.spinner(t('trying_uflp')):
        result = try_read_as_uflp(text)
        if result is not None:
            problem, detected_type = result
            st.success(f"{t('success')}: {detected_type}")
            return problem, detected_type
    
    # محاولة ILP
    with st.spinner(t('trying_ilp')):
        result = try_read_as_ilp(text)
        if result is not None:
            problem, detected_type = result
            st.success(f"{t('success')}: {detected_type}")
            return problem, detected_type
    
    # إذا فشلت كل المحاولات
    raise ValueError(t('failed'))

# ------------------- Main Solving Function (معادلة للـ problem) -------------------
def solve_ahrh(problem, max_cycles, k_coarse, patience,
               use_R, R_tol, stable_gap_needed,
               use_cost_repeat, cost_repeat_times,
               use_gap_repeat, gap_repeat_times,
               use_contraction, diff_tol,
               use_R_stability, R_stability_tol, R_stability_cycles,
               resume=False):
    
    n = len(problem["obj"])
    x_lp, lp_val = lp_relaxation(problem)
    if x_lp is None:
        lp_val = float('inf')
    
    if not resume:
        # حل ابتدائي: تقريب حل LP مع ضمان الجدوى
        x = np.round(x_lp).astype(int) if x_lp is not None else np.zeros(n, dtype=int)
        x = np.clip(x, problem["lb"], problem["ub"])
        best_cost, x = evaluate_solution(x, problem)
        if best_cost == float('inf') or x is None:
            # حاول بحل الصفري
            x = np.zeros(n, dtype=int)
            best_cost, x = evaluate_solution(x, problem)
        if best_cost == float('inf') or x is None:
            # حاول بحل عشوائي
            for _ in range(10):
                x_rand = np.random.uniform(problem["lb"], problem["ub"])
                x_rand = np.round(x_rand).astype(int)
                cost, xr = evaluate_solution(x_rand, problem)
                if cost < best_cost:
                    best_cost = cost
                    x = xr
        if best_cost == float('inf') or x is None:
            st.error("لا يمكن إيجاد حل ابتدائي مجدٍ.")
            return None
        
        history = []
        total_time = 0.0
        start_cycle = 1
    else:
        state = load_state(STATE_FILE)
        if state is None:
            st.warning(t('no_state'))
            x = np.round(x_lp).astype(int) if x_lp is not None else np.zeros(n, dtype=int)
            x = np.clip(x, problem["lb"], problem["ub"])
            best_cost, x = evaluate_solution(x, problem)
            if best_cost == float('inf') or x is None:
                x = np.zeros(n, dtype=int)
                best_cost, x = evaluate_solution(x, problem)
            history = []
            total_time = 0.0
            start_cycle = 1
        else:
            st.info(t('resume_state'))
            start_cycle = state['runtime_info']['cycle'] + 1
            best_cost = state['runtime_info']['best_cost']
            x = np.array(state['runtime_info']['best_x'])
            history = state['runtime_info']['history']
            total_time = state['runtime_info']['total_time']
            lp_val = state['runtime_info']['lp_val']

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
    r_stable_count = 0
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
            
            # اختيار المجموعة الخشنة
            if x_lp is not None:
                # نأخذ المتغيرات المفتوحة حالياً (أكبر من 0.5)
                open_now = np.where(x > 0.5)[0].tolist()
                # وأيضاً أهم المتغيرات في حل LP
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
            
            new_cost, new_x = vcycle(x, problem, coarse, x_lp, best_cost)
            if new_cost == float('inf') or new_x is None:
                # إذا فشلت الدورة V، نبقى على الحل السابق
                new_cost = best_cost
                new_x = x.copy()
            
            gap = (new_cost - lp_val) / lp_val * 100 if lp_val not in [0, float('inf')] else 0
            R_val = compute_R(new_x, problem)
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
            
            status_placeholder.info(f"**Cycle {cycle} / {max_cycles}**")
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
            
            # تفعيل التسريع التلقائي
            if not acceleration_active and gap < 2.0 and R_val < 0.01:
                acceleration_active = True
                st.success(t('acceleration_on'))
            elif acceleration_active and (gap >= 2.0 or R_val >= 0.01):
                acceleration_active = False
            
            # معايير التوقف
            stop_now = False
            
            # 1. الصبر
            if no_improve >= patience:
                stop_reason = f"{t('patience')} ({patience})"
                stop_now = True
            
            # 2. R threshold مع استقرار الفجوة
            if not stop_now and use_R and R_val < R_tol:
                if last_gap is not None and abs(gap - last_gap) < 1e-6:
                    stable_gap_count += 1
                    if stable_gap_count >= stable_gap_needed:
                        stop_reason = f"R < {R_tol} and gap stable"
                        stop_now = True
                else:
                    stable_gap_count = 0
            
            # 3. تكرار التكلفة
            if not stop_now and use_cost_repeat:
                if last_cost is not None and abs(new_cost - last_cost) < 1e-6:
                    cost_repeat_count += 1
                    if cost_repeat_count >= cost_repeat_times:
                        stop_reason = t('use_cost_repeat')
                        stop_now = True
                else:
                    cost_repeat_count = 0
            
            # 4. تكرار الفجوة
            if not stop_now and use_gap_repeat:
                if last_gap is not None and abs(gap - last_gap) < 1e-6:
                    gap_repeat_count += 1
                    if gap_repeat_count >= gap_repeat_times:
                        stop_reason = t('use_gap_repeat')
                        stop_now = True
                else:
                    gap_repeat_count = 0
            
            # 5. الانكماش
            if not stop_now and use_contraction:
                if diff < diff_tol and R_val < R_tol:
                    stop_reason = t('use_contraction')
                    stop_now = True
            
            # 6. استقرار R (التغير النسبي)
            if not stop_now and use_R_stability:
                if len(R_history) >= 2:
                    prev_R = R_history[-2]
                    if prev_R > 1e-12:
                        rel_change = abs(R_val - prev_R) / prev_R
                        if rel_change < R_stability_tol:
                            r_stable_count += 1
                            if r_stable_count >= R_stability_cycles:
                                stop_reason = f"R stability ({R_stability_tol})"
                                stop_now = True
                        else:
                            r_stable_count = 0
                    else:
                        # R صفر عملياً
                        r_stable_count += 1
                        if r_stable_count >= R_stability_cycles:
                            stop_reason = "R ~ 0"
                            stop_now = True
                else:
                    r_stable_count = 0
            
            last_cost = new_cost
            last_gap = gap
            last_R = R_val
            last_x = new_x.copy()
            
            # حفظ checkpoint دوري
            if time.time() - last_save_time > 5:
                runtime_info = {
                    'cycle': cycle,
                    'best_cost': best_cost,
                    'best_x': x.tolist(),
                    'history': cycles_log,
                    'lp_val': lp_val,
                    'total_time': elapsed,
                }
                checkpoint = build_checkpoint(problem, runtime_info)
                save_checkpoint_local(checkpoint)
                # أيضاً حفظ في الملف القديم للتوافق
                save_state(STATE_FILE, cycle, best_cost, x, cycles_log, lp_val, elapsed, {'problem': problem})
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
        'open_fac': None,  # لم نعد نخزنها هنا، يمكن استخراجها من meta
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
    
    max_cycles = st.slider(t('max_cycles'), 10, 500, 100, 10)
    k_coarse = st.slider(t('k_coarse'), 3, 30, 15)
    patience = st.slider(t('patience'), 5, 100, 20)
    
    st.markdown("---")
    st.markdown(f"## {t('sidebar_stop')}")
    st.markdown(t('choose_criteria'))
    
    use_R = st.checkbox(t('use_R'), value=True)
    if use_R:
        R_tol = st.number_input(t('R_tol'), value=1e-4, format="%.0e", step=1e-4)
        stable_gap_needed = st.number_input(t('stable_gap'), min_value=1, max_value=20, value=5)
    else:
        R_tol, stable_gap_needed = 1e-4, 5
    
    use_cost_repeat = st.checkbox(t('use_cost_repeat'), value=False)
    if use_cost_repeat:
        cost_repeat_times = st.number_input(t('cost_repeat_times'), min_value=2, max_value=50, value=10)
    else:
        cost_repeat_times = 10
    
    use_gap_repeat = st.checkbox(t('use_gap_repeat'), value=False)
    if use_gap_repeat:
        gap_repeat_times = st.number_input(t('gap_repeat_times'), min_value=2, max_value=50, value=10)
    else:
        gap_repeat_times = 10
    
    use_contraction = st.checkbox(t('use_contraction'), value=True)
    if use_contraction:
        diff_tol = st.number_input(t('diff_tol'), value=1e-8, format="%.0e", step=1e-8)
    else:
        diff_tol = 1e-8
    
    # معيار استقرار R
    use_R_stability = st.checkbox(t('use_R_stability'), value=True)
    if use_R_stability:
        R_stability_tol = st.number_input(t('R_stability_tol'), value=1e-3, format="%.0e", step=1e-3)
        R_stability_cycles = st.number_input(t('R_stability_cycles'), min_value=2, max_value=20, value=5)
    else:
        R_stability_tol, R_stability_cycles = 1e-3, 5
    
    st.markdown("---")
    st.write(f"👥 {t('workers')}: {NUM_WORKERS}")
    
    if os.path.exists(STATE_FILE):
        if st.button(t('delete_state')):
            if delete_state(STATE_FILE):
                st.success(t('delete_state'))
                st.rerun()

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
                        problem, detected_type = flexible_file_reader(text, uploaded_file.name)
                    
                    n = len(problem["obj"])
                    m = len(problem["b"])
                    
                    # Problem info
                    st.markdown(f"""
                    <div class="success-box">
                        <h4>{t('problem_info')}</h4>
                        <p><b>{t('problem_type')}:</b> {problem['original_type']} ({detected_type})</p>
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
                                result = solve_ahrh(problem,
                                                    max_cycles, k_coarse, patience,
                                                    use_R, R_tol, stable_gap_needed,
                                                    use_cost_repeat, cost_repeat_times,
                                                    use_gap_repeat, gap_repeat_times,
                                                    use_contraction, diff_tol,
                                                    use_R_stability, R_stability_tol, R_stability_cycles,
                                                    resume=True)
                                if result:
                                    st.session_state.result = result
                                    st.session_state.problem = problem
                                    st.rerun()
                        with col2:
                            if st.button(t('reset')):
                                if os.path.exists(STATE_FILE):
                                    os.remove(STATE_FILE)
                                result = solve_ahrh(problem,
                                                    max_cycles, k_coarse, patience,
                                                    use_R, R_tol, stable_gap_needed,
                                                    use_cost_repeat, cost_repeat_times,
                                                    use_gap_repeat, gap_repeat_times,
                                                    use_contraction, diff_tol,
                                                    use_R_stability, R_stability_tol, R_stability_cycles,
                                                    resume=False)
                                if result:
                                    st.session_state.result = result
                                    st.session_state.problem = problem
                                    st.rerun()
                    else:
                        result = solve_ahrh(problem,
                                            max_cycles, k_coarse, patience,
                                            use_R, R_tol, stable_gap_needed,
                                            use_cost_repeat, cost_repeat_times,
                                            use_gap_repeat, gap_repeat_times,
                                            use_contraction, diff_tol,
                                            use_R_stability, R_stability_tol, R_stability_cycles,
                                            resume=False)
                        if result:
                            st.session_state.result = result
                            st.session_state.problem = problem
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
    
    # هنا يمكن توسعة الإدخال اليدوي لاحقاً
    st.info("هذا الجزء قيد التطوير. استخدم رفع الملفات حالياً.")

# ------------------- Results Display -------------------
st.markdown("---")

if st.session_state.get('result') is not None:
    res = st.session_state.result
    problem = st.session_state.get('problem', None)
    
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
        if problem and problem['original_type'] == 'UFLP':
            # نحاول استخراج عدد المرافق المفتوحة (يمكن إضافتها لاحقاً)
            val = "—"
            label = t('open_fac')
        else:
            val = f"{len(problem['obj']) if problem else 0}×{len(problem['b']) if problem else 0}"
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
