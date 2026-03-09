# -*- coding: utf-8 -*-
"""
تطبيق AHRH لحل مسائل البرمجة الصحيحة (ILP) و UFLP
مع واجهة متعددة اللغات، عرض التقدم، معايير توقف متعددة، وإرسال التعليقات إلى GitHub Issues
(تم إزالة التبويب العشوائي، مع بقاء كل التحسينات الأخرى)
"""

import streamlit as st
import numpy as np
import pulp
import time
import matplotlib.pyplot as plt
import pandas as pd
import requests
import json
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
warnings.filterwarnings("ignore")

# ------------------- معلومات الاتصال -------------------
CONTACT_EMAIL = "zakibeny@gmail.com"
CONTACT_PHONE = "0021355614305 / 00213779679073"
CONTACT_FAX = "036495241"

# ------------------- إعدادات التوازي -------------------
NUM_WORKERS = 4

# ------------------- ترجمة النصوص (عربي/إنجليزي/فرنسي) -------------------
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
        'ilp': 'برمجة خطية صحيحة عامة (ILP)',
        'uflp': 'مسألة مواقع المرافق (UFLP)',
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
        'tab_manual': '✍️ إدخال يدوي',
        'upload_header': 'رفع ملف المسألة',
        'upload_info': 'يدعم ملفات UFLP (مثل gs250) والملفات العامة بصيغة ILP.',
        'choose_file': 'اختر ملف المسألة',
        'file_type': 'نوع الملف',
        'uflp_file': 'ملف UFLP (n m 0 + بيانات)',
        'ilp_file': 'ملف ILP عام (n m + c + A + b)',
        'ilp_format_help': """
**تنسيق ملف ILP العام:**
- السطر الأول: `n m` (عدد المتغيرات، عدد القيود)
- السطر الثاني: معاملات الهدف c (n قيمة)
- ثم m سطر: مصفوفة القيود A (كل سطر n قيمة)
- السطر الأخير: الطرف الأيمن b (m قيمة)
""",
        'uflp_format_help': """
**تنسيق ملف UFLP:**
- السطر الأول: `n m 0` (عدد المواقع، عدد العملاء)
- ثم n سطر: `رقم الموقع` `تكلفة الفتح` `تكاليف النقل إلى m عميل`
""",
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
        'open_fac': 'المرافق المفتوحة',
        'cycles_done': 'عدد الدورات',
        'time': 'الزمن (ث)',
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
        'feedback_section': '💬 تواصل معنا - أرسل تعليقك',
        'feedback_placeholder': 'اكتب تعليقك هنا... (سيتم إرساله كـ Issue في GitHub)',
        'feedback_submit': 'إرسال التعليق',
        'feedback_success': '✅ تم الإرسال بنجاح! يمكنك متابعة الـ issue على الرابط:',
        'feedback_error': '❌ فشل الإرسال:',
        'feedback_missing_token': '⚠️ خدمة إرسال التعليقات غير مفعلة حالياً.',
        'feedback_warning': 'الرجاء كتابة تعليق قبل الإرسال.',
        'pause_button': '⏸️ تعليق',
        'resume_button': '▶️ استئناف',
        'paused_message': '⏸️ الخوارزمية معلقة. اضغط استئناف للمتابعة.',
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
        'ilp': 'General Integer Programming (ILP)',
        'uflp': 'Facility Location (UFLP)',
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
        'tab_manual': '✍️ Manual Input',
        'upload_header': 'Upload Problem File',
        'upload_info': 'Supports UFLP files (e.g., gs250) and general ILP files.',
        'choose_file': 'Choose a file',
        'file_type': 'File Type',
        'uflp_file': 'UFLP file (n m 0 + data)',
        'ilp_file': 'General ILP file (n m + c + A + b)',
        'ilp_format_help': """
**General ILP File Format:**
- First line: `n m` (number of variables, number of constraints)
- Second line: objective coefficients c (n values)
- Then m lines: constraint matrix A (n values per line)
- Last line: right-hand side b (m values)
""",
        'uflp_format_help': """
**UFLP File Format:**
- First line: `n m 0` (number of facilities, number of customers)
- Then n lines: `facility_index` `opening_cost` `transport_costs to m customers`
""",
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
        'open_fac': 'Open Facilities',
        'cycles_done': 'Cycles Done',
        'time': 'Time (s)',
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
        'feedback_section': '💬 Contact Us - Send your feedback',
        'feedback_placeholder': 'Write your comment here... (will be sent as a GitHub Issue)',
        'feedback_submit': 'Send Feedback',
        'feedback_success': '✅ Sent successfully! You can track the issue at:',
        'feedback_error': '❌ Sending failed:',
        'feedback_missing_token': '⚠️ Feedback service is currently disabled.',
        'feedback_warning': 'Please write a comment before sending.',
        'pause_button': '⏸️ Pause',
        'resume_button': '▶️ Resume',
        'paused_message': '⏸️ Algorithm paused. Press Resume to continue.',
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
        'ilp': 'Programmation en nombres entiers générale (ILP)',
        'uflp': 'Localisation d\'installations (UFLP)',
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
        'tab_manual': '✍️ Saisie manuelle',
        'upload_header': 'Télécharger le fichier problème',
        'upload_info': 'Accepte les fichiers UFLP (ex. gs250) et les fichiers ILP généraux.',
        'choose_file': 'Choisir un fichier',
        'file_type': 'Type de fichier',
        'uflp_file': 'Fichier UFLP (n m 0 + données)',
        'ilp_file': 'Fichier ILP général (n m + c + A + b)',
        'ilp_format_help': """
**Format du fichier ILP général :**
- Première ligne : `n m` (nombre de variables, nombre de contraintes)
- Deuxième ligne : coefficients objectifs c (n valeurs)
- Ensuite m lignes : matrice des contraintes A (n valeurs par ligne)
- Dernière ligne : second membre b (m valeurs)
""",
        'uflp_format_help': """
**Format du fichier UFLP :**
- Première ligne : `n m 0` (nombre de sites, nombre de clients)
- Ensuite n lignes : `indice_site` `coût_ouverture` `coûts_transport vers m clients`
""",
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
        'open_fac': 'Sites ouverts',
        'cycles_done': 'Cycles effectués',
        'time': 'Temps (s)',
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
        'feedback_section': '💬 Contactez-nous - Envoyez votre commentaire',
        'feedback_placeholder': 'Écrivez votre commentaire ici... (sera envoyé comme Issue GitHub)',
        'feedback_submit': 'Envoyer',
        'feedback_success': '✅ Envoyé avec succès ! Suivez l\'issue sur :',
        'feedback_error': '❌ Échec de l\'envoi :',
        'feedback_missing_token': '⚠️ Le service de commentaires est actuellement désactivé.',
        'feedback_warning': 'Veuillez écrire un commentaire avant d\'envoyer.',
        'pause_button': '⏸️ Pause',
        'resume_button': '▶️ Reprendre',
        'paused_message': '⏸️ Algorithme en pause. Appuyez sur Reprendre pour continuer.',
    }
}

def t(key):
    return translations[st.session_state.language][key]

if 'language' not in st.session_state:
    st.session_state.language = 'English'
if 'paused' not in st.session_state:
    st.session_state.paused = False

# ------------------- دوال أساسية مشتركة -------------------
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

# ------------------- دوال خاصة بـ UFLP -------------------
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

def vcycle_uflp(y, f, c, coarse, y_lp=None, gap_threshold=5.0):
    cost1, y1 = smooth_uflp(y, f, c, y_lp=y_lp, iters=1, gap_threshold=gap_threshold)
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
    cost2, y2 = smooth_uflp(best_y, f, c, y_lp=y_lp, iters=1, gap_threshold=gap_threshold)
    cost3, y3 = local_search_advanced_uflp(y2, cost2, f, c, max_iter=5)
    return cost3, y3

def smooth_uflp(y, f, c, y_lp=None, iters=1, gap_threshold=5.0):
    best = solve_lp_fixed_y_uflp(y, f, c)
    best_y = y.copy()
    for _ in range(iters):
        R_val = compute_R(y)
        n_free = len(get_fractional_indices(y))
        new_cost, new_y = hierarchical_radial_scan_parallel_uflp(
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

def hierarchical_radial_scan_parallel_uflp(y_center, R, n_free, f, c, best_cost, best_y,
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
        if hasattr(st.session_state, 'acceleration') and st.session_state.acceleration:
            dirs_count = max(2, dirs_per_layer // 2)
        else:
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

def local_search_advanced_uflp(y, best_cost, f, c, max_iter=10):
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

def solve_ahrh_uflp(f, c, max_cycles, k_coarse, patience,
                    use_R, R_tol, stable_gap_needed,
                    use_cost_repeat, cost_repeat_times,
                    use_gap_repeat, gap_repeat_times,
                    use_contraction, diff_tol,
                    resume_state=None):
    n, m = len(f), c.shape[1]
    y_lp, lp_val = lp_relaxation_uflp(f, c)
    if lp_val is None or lp_val == 0:
        lp_val = 1e-12  # تجنب القسمة على صفر

    if resume_state is not None:
        y = resume_state['y']
        best = resume_state['best']
        cycles_log = resume_state['cycles_log']
        gap_history = resume_state['gap_history']
        R_history = resume_state['R_history']
        diff_history = resume_state['diff_history']
        no_improve = resume_state['no_improve']
        cycles_done = resume_state['cycles_done']
        stop_reason = resume_state['stop_reason']
        acceleration_active = resume_state['acceleration_active']
        cost_repeat_count = resume_state['cost_repeat_count']
        gap_repeat_count = resume_state['gap_repeat_count']
        stable_gap_count = resume_state['stable_gap_count']
        last_cost = resume_state['last_cost']
        last_gap = resume_state['last_gap']
        last_R = resume_state['last_R']
        last_y = resume_state['last_y']
        start_cycle = resume_state['cycle'] + 1
        total_time_so_far = resume_state['total_time_so_far']
    else:
        # استخدام حل LP التقريبي كنقطة بداية أفضل
        if y_lp is not None:
            y = (y_lp > 0.5).astype(int)
        else:
            y = np.ones(n, dtype=int)
        best = solve_lp_fixed_y_uflp(y, f, c)
        # إذا كان الحل غير ممكن، نفتح جميع المرافق
        if best == float('inf'):
            y = np.ones(n, dtype=int)
            best = solve_lp_fixed_y_uflp(y, f, c)

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
        last_y = y.copy()
        start_cycle = 0
        total_time_so_far = 0.0

    progress_bar = st.progress(start_cycle / max_cycles if max_cycles>0 else 0)
    status_placeholder = st.empty()
    details_placeholder = st.empty()
    start_time = time.time()

   
