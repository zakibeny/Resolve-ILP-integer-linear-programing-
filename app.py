```python
# -*- coding: utf-8 -*-
"""
تطبيق AHRH المتكامل لحل مسائل UFLP و ILP
مع تحسينات السرعة التقليم دعم MP مؤقت واضح، وإمكانية الإيقاف المؤقت والاستئناف
نسخة نهائية مع وصف متعدد اللغات وخيار حفظ على Google Drive (يدوي)
"""

import streamlit as st
import numpy as np
import pulp
import time
import matplotlib.pyplot as plt
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
import tempfile
import os
import base64
from datetime import datetime
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
        'intro': '''
        ### 📦 هذا التطبيق يحل مسائل:
        - **البرمجة الخطية الصحيحة العامة** (ILP)
        - **مسائل مواقع المرافق** (UFLP)
        
        وهو مبني على خوارزمية AHRH الحاصلة على براءة اختراع.
        ''',
        'problem_type': 'نوع المسألة',
        'uflp': 'مسألة مواقع المرافق (UFLP)',
        'ilp': 'برمجة خطية صحيحة عامة (ILP)',
        'sidebar_algo': '⚙️ معاملات الخوارزمية',
        'max_cycles': 'عدد الدورات الأقصى',
        'k_coarse': 'حجم المجموعة الخشنة (k)',
        'patience': 'الصبر (عدد الدورات بدون تحسن)',
        'tiny_patience': 'الصبر للتحسينات الضئيلة',
        'tiny_improve_threshold': 'عتبة التحسين الضئيل',
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
        'upload_info': 'يدعم ملفات UFLP، ILP عام، و MPS',
        'choose_file': 'اختر ملف المسألة',
        'file_type': 'نوع الملف',
        'uflp_format': """
**تنسيق ملف UFLP:**
- السطر الأول: `n m 0` (عدد المواقع، عدد العملاء)
- ثم n سطر: `[رقم الموقع] [تكلفة الفتح] [تكاليف النقل إلى m عميل]`
- مثال:
```

5 3 0
1 100 5 6 7
2 150 8 9 10
3 120 4 5 6
4 130 7 8 9
5 110 3 4 5

```
""",
        'general_format': """
**تنسيق الملف العام (ILP):**
- السطر الأول: `n m` (عدد المتغيرات، عدد القيود)
- السطر الثاني: معاملات الهدف `c` (n قيمة)
- ثم m سطر: مصفوفة القيود `A` (كل سطر n قيمة)
- السطر الأخير: الطرف الأيمن `b` (m قيمة)
- مثال:
```

2 1
1 2
1 2
3

```
""",
        'mps_format': """
**ملفات MPS:**
- صيغة قياسية لمسائل البرمجة الرياضية.
- سيتم تحويلها تلقائياً إلى الصيغة الداخلية للتطبيق.
- يدعم المسائل الخطية والصحيحة (بافتراض أن جميع المتغيرات صحيحة).
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
        'save_drive': '💾 حفظ النتائج على Google Drive',
        'save_drive_instructions': 'اضغط على الزر لتحميل ملف CSV، ثم ارفعه يدويًا إلى Google Drive.',
        'info_placeholder': '👈 اختر مصدر المسألة من التبويبات أعلاه واضغط على زر التشغيل.',
        'footer': 'تم التطوير بواسطة Zakarya Benregreg - خوارزمية AHRH محمية ببراءة اختراع.',
        'acceleration_on': '⚡ وضع التسريع مُفعّل (فجوة < 2% و R < 0.01)',
        'acceleration_off': 'وضع التسريع غير مُفعّل',
        'pause_button': '⏸️ إيقاف مؤقت',
        'resume_button': '▶️ استئناف',
        'paused_message': '⏸️ الخوارزمية في وضع الإيقاف المؤقت. اضغط استئناف للمتابعة.',
        'control_panel': '⏯️ لوحة التحكم',
        'elapsed_time': '⏱️ الوقت المنقضي',
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
        'intro': '''
        ### 📦 This app solves:
        - **General Integer Linear Programming** (ILP)
        - **Uncapacitated Facility Location Problems** (UFLP)
        
        It is based on the patented AHRH algorithm.
        ''',
        'problem_type': 'Problem Type',
        'uflp': 'Facility Location (UFLP)',
        'ilp': 'General Integer Programming (ILP)',
        'sidebar_algo': '⚙️ Algorithm Parameters',
        'max_cycles': 'Max Cycles',
        'k_coarse': 'Coarse Set Size (k)',
        'patience': 'Patience (cycles without improvement)',
        'tiny_patience': 'Tiny improvement patience',
        'tiny_improve_threshold': 'Tiny improvement threshold',
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
        'upload_info': 'Supports UFLP, general ILP, and MPS files',
        'choose_file': 'Choose a file',
        'file_type': 'File Type',
        'uflp_format': """
**UFLP File Format:**
- First line: `n m 0` (number of facilities, number of customers)
- Then n lines: `[facility index] [opening cost] [transport costs to m customers]`
- Example:
```

5 3 0
1 100 5 6 7
2 150 8 9 10
3 120 4 5 6
4 130 7 8 9
5 110 3 4 5

```
""",
        'general_format': """
**General ILP File Format:**
- First line: `n m` (number of variables, number of constraints)
- Second line: objective coefficients `c` (n values)
- Then m lines: constraint matrix `A` (n values per line)
- Last line: right-hand side `b` (m values)
- Example:
```

2 1
1 2
1 2
3

```
""",
        'mps_format': """
**MPS Files:**
- Standard format for mathematical programming problems.
- Will be automatically converted to the internal format.
- Supports linear and integer problems (assuming all variables are integer).
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
        'save_drive': '💾 Save results to Google Drive',
        'save_drive_instructions': 'Click the button to download a CSV file, then upload it manually to Google Drive.',
        'info_placeholder': '👈 Choose a data source and click the run button.',
        'footer': 'Developed by Zakarya Benregreg - AHRH algorithm patented.',
        'acceleration_on': '⚡ Acceleration mode ON (gap < 2% and R < 0.01)',
        'acceleration_off': 'Acceleration mode OFF',
        'pause_button': '⏸️ Pause',
        'resume_button': '▶️ Resume',
        'paused_message': '⏸️ Algorithm paused. Press Resume to continue.',
        'control_panel': '⏯️ Control Panel',
        'elapsed_time': '⏱️ Elapsed Time',
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
        'intro': '''
        ### 📦 Cette application résout :
        - **Programmation linéaire en nombres entiers générale** (ILP)
        - **Problèmes de localisation d\'installations** (UFLP)
        
        Elle est basée sur l\'algorithme breveté AHRH.
        ''',
        'problem_type': 'Type de problème',
        'uflp': 'Localisation d\'installations (UFLP)',
        'ilp': 'Programmation en nombres entiers générale (ILP)',
        'sidebar_algo': '⚙️ Paramètres de l\'algorithme',
        'max_cycles': 'Cycles max',
        'k_coarse': 'Taille de l\'ensemble grossier (k)',
        'patience': 'Patience (cycles sans amélioration)',
        'tiny_patience': 'Patience pour petites améliorations',
        'tiny_improve_threshold': 'Seuil de petite amélioration',
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
        'upload_info': 'Accepte les fichiers UFLP, ILP général et MPS',
        'choose_file': 'Choisir un fichier',
        'file_type': 'Type de fichier',
        'uflp_format': """
**Format du fichier UFLP :**
- Première ligne : `n m 0` (nombre de sites, nombre de clients)
- Ensuite n lignes : `[indice site] [coût ouverture] [coûts transport vers m clients]`
- Exemple :
```

5 3 0
1 100 5 6 7
2 150 8 9 10
3 120 4 5 6
4 130 7 8 9
5 110 3 4 5

```
""",
        'general_format': """
**Format du fichier général ILP :**
- Première ligne : `n m` (nombre de variables, nombre de contraintes)
- Deuxième ligne : coefficients objectifs `c` (n valeurs)
- Ensuite m lignes : matrice des contraintes `A` (n valeurs par ligne)
- Dernière ligne : second membre `b` (m valeurs)
- Exemple :
```

2 1
1 2
1 2
3

```
""",
        'mps_format': """
**Fichiers MPS :**
- Format standard pour les problèmes d'optimisation mathématique.
- Sera automatiquement converti au format interne.
- Supporte les problèmes linéaires et en nombres entiers.
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
        'save_drive': '💾 Sauvegarder les résultats sur Google Drive',
        'save_drive_instructions': 'Cliquez sur le bouton pour télécharger un fichier CSV, puis téléversez-le manuellement sur Google Drive.',
        'info_placeholder': '👈 Choisissez une source de données et cliquez sur le bouton.',
        'footer': 'Développé par Zakarya Benregreg - Algorithme AHRH breveté.',
        'acceleration_on': '⚡ Mode accélération ACTIVÉ (gap < 2% et R < 0.01)',
        'acceleration_off': 'Mode accélération DÉSACTIVÉ',
        'pause_button': '⏸️ Pause',
        'resume_button': '▶️ Reprendre',
        'paused_message': '⏸️ Algorithme en pause. Appuyez sur Reprendre pour continuer.',
        'control_panel': '⏯️ Panneau de contrôle',
        'elapsed_time': '⏱️ Temps écoulé',
    }
}

def t(key):
    return translations[st.session_state.language][key]

# ------------------- إعداد الحالة -------------------
if 'language' not in st.session_state:
    st.session_state.language = 'English'
if 'paused' not in st.session_state:
    st.session_state.paused = False
if 'resume_state' not in st.session_state:
    st.session_state.resume_state = None
if 'last_params' not in st.session_state:
    st.session_state.last_params = None

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

# ------------------- دوال UFLP -------------------
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

def hierarchical_radial_scan_parallel_uflp(y_center, R, n_free, f, c, best_cost, best_y,
                                      n_layers=2, dirs_per_layer=10, alpha_schedule='adaptive',
                                      y_lp=None, gap_threshold=5.0):
    frac_idx = get_fractional_indices(y_center)
    if len(frac_idx) == 0:
        return best_cost, best_y
    y_frac = y_center[frac_idx].copy()
    local_best = best_cost
    local_best_y = best_y
    base_alpha = R / (np.sqrt(n_free) + 1e-12) if n_free > 0 else 0.1

    # تقدير مسبق لأقل تكلفة نقل (للتقليم)
    min_transport_per_customer = np.min(c, axis=0)

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
                # تقدير سريع
                y_cand_frac = y_frac + sign * alpha_k * u
                y_cand_int_frac = (y_cand_frac > 0.5).astype(int)
                temp_y = y_center.copy()
                temp_y[frac_idx] = y_cand_int_frac
                open_count = np.sum(temp_y > 0.5)
                if open_count == 0:
                    estimated_cost = float('inf')
                else:
                    estimated_cost = np.sum(f[temp_y > 0.5])
                    # أقل تكلفة نقل من المرافق المفتوحة
                    for cust in range(c.shape[1]):
                        estimated_cost += np.min(c[temp_y > 0.5, cust])
                # تقليم
                if estimated_cost >= local_best - 1e-6:
                    continue

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

def vcycle_uflp(y, f, c, coarse, y_lp=None, gap_threshold=5.0):
    cost1, y1 = smooth_uflp(y, f, c, y_lp=y_lp, iters=1, gap_threshold=gap_threshold)
    if not coarse:
        return cost1, y1
    best = cost1
    best_y = y1
    n_coarse = len(coarse)

    min_transport_per_customer = np.min(c, axis=0)

    def evaluate_bits(bits):
        yc = np.array([(bits >> i) & 1 for i in range(n_coarse)])
        # تقدير سريع
        estimated_cost = 0.0
        open_indices = []
        for idx, val in zip(coarse, yc):
            if val:
                estimated_cost += f[idx]
                open_indices.append(idx)
        if len(open_indices) == 0:
            estimated_cost = float('inf')
        else:
            for cust in range(c.shape[1]):
                estimated_cost += min_transport_per_customer[cust]
        if estimated_cost >= best - 1e-6:
            return float('inf'), None

        y_full = y1.copy()
        for idx, val in zip(coarse, yc):
            y_full[idx] = val
        cost = solve_lp_fixed_y_uflp(y_full, f, c)
        return cost, y_full

    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = [executor.submit(evaluate_bits, bits) for bits in range(1 << n_coarse)]
        for future in as_completed(futures):
            cost, y_cand = future.result()
            if cost is not None and cost < best - 1e-6:
                best = cost
                best_y = y_cand
    cost2, y2 = smooth_uflp(best_y, f, c, y_lp=y_lp, iters=1, gap_threshold=gap_threshold)
    cost3, y3 = local_search_advanced_uflp(y2, cost2, f, c, max_iter=5)
    return cost3, y3

def solve_ahrh_uflp(f, c, max_cycles, k_coarse, patience, tiny_patience,
                    use_R, R_tol, stable_gap_needed,
                    use_cost_repeat, cost_repeat_times,
                    use_gap_repeat, gap_repeat_times,
                    use_contraction, diff_tol,
                    use_tiny_improve, tiny_improve_threshold,
                    resume_state=None):
    n, m = len(f), c.shape[1]
    y_lp, lp_val = lp_relaxation_uflp(f, c)
    if lp_val is None or lp_val == 0:
        lp_val = 1e-12

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
        tiny_improve_count = resume_state['tiny_improve_count']
        last_cost = resume_state['last_cost']
        last_gap = resume_state['last_gap']
        last_R = resume_state['last_R']
        last_y = resume_state['last_y']
        start_cycle = resume_state['cycle'] + 1
        total_time_so_far = resume_state['total_time_so_far']
    else:
        if y_lp is not None:
            y = (y_lp > 0.5).astype(int)
            if np.sum(y) == 0:
                cheapest = np.argmin(f)
                y[cheapest] = 1
        else:
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
        tiny_improve_count = 0
        last_cost = None
        last_gap = None
        last_R = None
        last_y = y.copy()
        start_cycle = 0
        total_time_so_far = 0.0

    progress_bar = st.progress(start_cycle / max_cycles if max_cycles > 0 else 0)
    status_placeholder = st.empty()
    details_placeholder = st.empty()
    start_time = time.time()

    cycle = start_cycle
    while cycle < max_cycles:
        if st.session_state.get('paused', False):
            st.session_state.resume_state = {
                'y': y,
                'best': best,
                'cycles_log': cycles_log,
                'gap_history': gap_history,
                'R_history': R_history,
                'diff_history': diff_history,
                'no_improve': no_improve,
                'cycles_done': cycle,
                'stop_reason': stop_reason,
                'acceleration_active': acceleration_active,
                'cost_repeat_count': cost_repeat_count,
                'gap_repeat_count': gap_repeat_count,
                'stable_gap_count': stable_gap_count,
                'tiny_improve_count': tiny_improve_count,
                'last_cost': last_cost,
                'last_gap': last_gap,
                'last_R': last_R,
                'last_y': last_y,
                'cycle': cycle,
                'total_time_so_far': total_time_so_far + (time.time() - start_time)
            }
            st.info(t('paused_message'))
            break

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

        if acceleration_active:
            st.session_state.acceleration = True
        else:
            st.session_state.acceleration = False

        new_cost, new_y = vcycle_uflp(y, f, c, coarse, y_lp=y_lp, gap_threshold=3.0)
        gap = (new_cost - lp_val) / lp_val * 100
        R_val = compute_R(new_y)
        diff = np.linalg.norm(new_y - last_y)

        improved = new_cost < best - 1e-6
        if improved:
            improvement_amount = best - new_cost
            best = new_cost
            y = new_y
            no_improve = 0
        else:
            improvement_amount = 0
            no_improve += 1

        if use_tiny_improve and improved and improvement_amount < tiny_improve_threshold:
            tiny_improve_count += 1
        else:
            tiny_improve_count = 0

        gap_history.append(gap)
        R_history.append(R_val)
        diff_history.append(diff)
        cycles_log.append({
            'cycle': cycle+1,
            'cost': new_cost,
            'gap': gap,
            'R': R_val,
            'diff': diff,
            'improved': improved,
            'best_so_far': best
        })

        elapsed = total_time_so_far + (time.time() - start_time)
        progress_bar.progress((cycle + 1) / max_cycles)
        status_placeholder.info(f"**{t('cycle')} {cycle+1} / {max_cycles}**")
        details_placeholder.markdown(
            f"**{t('best_cost')}:** {new_cost:,.2f}  \n"
            f"**{t('gap')}:** {gap:.4f}%  \n"
            f"**R:** {R_val:.6f}  \n"
            f"**{t('improved')}:** {'✅' if improved else '❌'}  \n"
            f"**{t('best_so_far')}:** {best:,.2f}  \n"
            f"**{t('elapsed_time')}:** {elapsed:.2f} ثانية"
        )
        time.sleep(0.1)

        if not acceleration_active and gap < 2.0 and R_val < 0.01:
            acceleration_active = True
            st.session_state.acceleration = True
            st.info(t('acceleration_on'))
        elif acceleration_active and (gap >= 2.0 or R_val >= 0.01):
            acceleration_active = False
            st.session_state.acceleration = False

        stop_now = False

        if no_improve >= patience:
            stop_reason = f"Patience ({patience} cycles without improvement)"
            stop_now = True
        if not stop_now and use_tiny_improve and tiny_improve_count >= tiny_patience:
            stop_reason = f"Tiny improvements ({tiny_improve_count} cycles with improvement < {tiny_improve_threshold:.2e})"
            stop_now = True
        if not stop_now and use_R and R_val < R_tol:
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
        if not stop_now and use_contraction and diff < diff_tol and R_val < R_tol:
            stop_reason = f"Contraction: diff < {diff_tol} and R < {R_tol}"
            stop_now = True

        last_cost = new_cost
        last_gap = gap
        last_R = R_val
        last_y = new_y.copy()

        if stop_now:
            cycles_done = cycle + 1
            break

        cycle += 1

    total_time = total_time_so_far + (time.time() - start_time)

    if cycle >= max_cycles:
        cycles_done = max_cycles
        stop_reason = f"Max cycles ({max_cycles}) reached"
        if 'resume_state' in st.session_state:
            st.session_state.resume_state = None
        st.session_state.paused = False

    progress_bar.empty()
    status_placeholder.empty()
    details_placeholder.empty()

    if acceleration_active:
        st.info(t('acceleration_on'))

    return {
        'best_cost': best,
        'lp_val': lp_val,
        'gap': (best - lp_val) / lp_val * 100,
        'open_fac': len(np.where(y > 0.5)[0]),
        'cycles_done': cycles_done,
        'gap_history': gap_history,
        'R_history': R_history,
        'diff_history': diff_history,
        'cycles_log': cycles_log,
        'stop_reason': stop_reason,
        'acceleration_active': acceleration_active,
        'total_time': total_time
    }

# ------------------- دوال ILP العامة -------------------
def solve_lp_pulp(c, A, b, integer=False):
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

def evaluate_integer_solution(x, c, A, b):
    x_int = np.round(x).astype(int)
    x_int = np.maximum(x_int, 0)
    if np.all(A @ x_int <= b + 1e-6) and np.all(x_int >= 0):
        return c @ x_int, x_int
    else:
        return float('inf'), None

def vcycle_general_ilp(y, c, A, b, coarse, y_lp, R):
    n = len(y)
    y_smooth = y.copy()
    best_cost, _ = evaluate_integer_solution(y, c, A, b)
    if best_cost == float('inf'):
        best_cost = c @ y

    alpha = R / 5
    frac_idx = np.where((y > 0.01) & (y < 0.99))[0]
    if len(frac_idx) == 0:
        frac_idx = list(range(n))

    dirs = generate_biased_directions(y_lp, frac_idx, 10, alpha, bias_strength=0.5)
    for u in dirs:
        for sign in [1, -1]:
            y_cand = y[frac_idx] + sign * alpha * u
            y_cand_int = np.round(y_cand).astype(int)
            y_cand_int = np.maximum(y_cand_int, 0)
            y_full = y.copy()
            y_full[frac_idx] = y_cand_int
            cost, feasible = evaluate_integer_solution(y_full, c, A, b)
            if feasible is not None and cost < best_cost - 1e-6:
                best_cost = cost
                y_smooth = y_full.copy()

    if len(coarse) > 0:
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
                cost, feasible = evaluate_integer_solution(y_new, c, A, b)
                if feasible is not None and cost < best_cost - 1e-6:
                    best_cost = cost
                    y_smooth = y_new.copy()
    return best_cost, y_smooth

def solve_ahrh_general_ilp(c, A, b, max_cycles, k_coarse, patience, tiny_patience,
                           use_R, R_tol, stable_gap_needed,
                           use_cost_repeat, cost_repeat_times,
                           use_gap_repeat, gap_repeat_times,
                           use_contraction, diff_tol,
                           use_tiny_improve, tiny_improve_threshold,
                           resume_state=None):
    n = len(c)
    m = len(b)

    x_lp, lp_val = solve_lp_pulp(c, A, b, integer=False)
    if x_lp is None:
        return None, None, None
    if lp_val == 0:
        lp_val = 1e-12

    R_initial = compute_R(x_lp)
    R = R_initial if R_initial > 0 else 1.0

    if resume_state is not None:
        x = resume_state['x']
        best_cost = resume_state['best_cost']
        cycles_log = resume_state['cycles_log']
        gap_history = resume_state['gap_history']
        R_history = resume_state['R_history']
        no_improve = resume_state['no_improve']
        cycles_done = resume_state['cycles_done']
        stop_reason = resume_state['stop_reason']
        acceleration_active = resume_state['acceleration_active']
        cost_repeat_count = resume_state['cost_repeat_count']
        gap_repeat_count = resume_state['gap_repeat_count']
        stable_gap_count = resume_state['stable_gap_count']
        tiny_improve_count = resume_state['tiny_improve_count']
        last_cost = resume_state['last_cost']
        last_gap = resume_state['last_gap']
        last_R = resume_state['last_R']
        last_x = resume_state['last_x']
        start_cycle = resume_state['cycle'] + 1
        total_time_so_far = resume_state['total_time_so_far']
    else:
        x = np.round(x_lp).astype(int)
        x = np.maximum(x, 0)
        best_cost, _ = evaluate_integer_solution(x, c, A, b)
        if best_cost == float('inf'):
            x = np.zeros(n, dtype=int)
            best_cost, _ = evaluate_integer_solution(x, c, A, b)

        cycles_log = []
        gap_history = []
        R_history = []
        no_improve = 0
        cycles_done = 0
        stop_reason = ""
        acceleration_active = False
        cost_repeat_count = 0
        gap_repeat_count = 0
        stable_gap_count = 0
        tiny_improve_count = 0
        last_cost = None
        last_gap = None
        last_R = None
        last_x = x.copy()
        start_cycle = 0
        total_time_so_far = 0.0

    progress_bar = st.progress(start_cycle / max_cycles if max_cycles > 0 else 0)
    status_placeholder = st.empty()
    details_placeholder = st.empty()
    start_time = time.time()

    cycle = start_cycle
    while cycle < max_cycles:
        if st.session_state.get('paused', False):
            st.session_state.resume_state = {
                'x': x,
                'best_cost': best_cost,
                'cycles_log': cycles_log,
                'gap_history': gap_history,
                'R_history': R_history,
                'no_improve': no_improve,
                'cycles_done': cycle,
                'stop_reason': stop_reason,
                'acceleration_active': acceleration_active,
                'cost_repeat_count': cost_repeat_count,
                'gap_repeat_count': gap_repeat_count,
                'stable_gap_count': stable_gap_count,
                'tiny_improve_count': tiny_improve_count,
                'last_cost': last_cost,
                'last_gap': last_gap,
                'last_R': last_R,
                'last_x': last_x,
                'cycle': cycle,
                'total_time_so_far': total_time_so_far + (time.time() - start_time)
            }
            st.info(t('paused_message'))
            break

        current_R = R / (cycle + 1)

        coarse = []
        if x_lp is not None:
            open_now = np.where(x > 0.5)[0].tolist()
            top_lp = np.argsort(-x_lp)[:k_coarse].tolist()
            coarse = list(set(open_now + top_lp))
            if len(coarse) > 10:
                importance = [(i, x_lp[i]) for i in coarse]
                importance.sort(key=lambda x: x[1], reverse=True)
                coarse = [i for i, _ in importance[:10]]

        new_cost, new_x = vcycle_general_ilp(x, c, A, b, coarse, x_lp, current_R)
        gap = (new_cost - lp_val) / lp_val * 100 if lp_val != 0 else 0
        R_val = compute_R(new_x)
        diff = np.linalg.norm(new_x - last_x)

        improved = new_cost < best_cost - 1e-6
        if improved:
            improvement_amount = best_cost - new_cost
            best_cost = new_cost
            x = new_x
            no_improve = 0
        else:
            improvement_amount = 0
            no_improve += 1

        if use_tiny_improve and improved and improvement_amount < tiny_improve_threshold:
            tiny_improve_count += 1
        else:
            tiny_improve_count = 0

        gap_history.append(gap)
        R_history.append(R_val)
        cycles_log.append({
            'cycle': cycle+1,
            'cost': new_cost,
            'gap': gap,
            'R': R_val,
            'diff': diff,
            'improved': improved,
            'best_so_far': best_cost
        })

        elapsed = total_time_so_far + (time.time() - start_time)
        progress_bar.progress((cycle + 1) / max_cycles)
        status_placeholder.info(f"**{t('cycle')} {cycle+1} / {max_cycles}**")
        details_placeholder.markdown(
            f"**{t('best_cost')}:** {new_cost:,.2f}  \n"
            f"**{t('gap')}:** {gap:.4f}%  \n"
            f"**R:** {R_val:.6f}  \n"
            f"**{t('improved')}:** {'✅' if improved else '❌'}  \n"
            f"**{t('best_so_far')}:** {best_cost:,.2f}  \n"
            f"**{t('elapsed_time')}:** {elapsed:.2f} ثانية"
        )
        time.sleep(0.1)

        if not acceleration_active and gap < 2.0 and R_val < 0.01:
            acceleration_active = True
            st.info(t('acceleration_on'))
        elif acceleration_active and (gap >= 2.0 or R_val >= 0.01):
            acceleration_active = False

        stop_now = False

        if no_improve >= patience:
            stop_reason = f"Patience ({patience} cycles without improvement)"
            stop_now = True
        if not stop_now and use_tiny_improve and tiny_improve_count >= tiny_patience:
            stop_reason = f"Tiny improvements ({tiny_improve_count} cycles with improvement < {tiny_improve_threshold:.2e})"
            stop_now = True
        if not stop_now and use_R and R_val < R_tol:
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
        if not stop_now and use_contraction and diff < diff_tol and R_val < R_tol:
            stop_reason = f"Contraction: diff < {diff_tol} and R < {R_tol}"
            stop_now = True

        last_cost = new_cost
        last_gap = gap
        last_R = R_val
        last_x = new_x.copy()

        if stop_now:
            cycles_done = cycle + 1
            break

        cycle += 1

    total_time = total_time_so_far + (time.time() - start_time)

    if cycle >= max_cycles:
        cycles_done = max_cycles
        stop_reason = f"Max cycles ({max_cycles}) reached"
        if 'resume_state' in st.session_state:
            st.session_state.resume_state = None
        st.session_state.paused = False

    progress_bar.empty()
    status_placeholder.empty()
    details_placeholder.empty()

    if acceleration_active:
        st.info(t('acceleration_on'))

    return {
        'best_cost': best_cost,
        'lp_val': lp_val,
        'gap': (best_cost - lp_val) / lp_val * 100 if lp_val != 0 else 0,
        'cycles_done': cycles_done,
        'gap_history': gap_history,
        'R_history': R_history,
        'cycles_log': cycles_log,
        'stop_reason': stop_reason,
        'total_time': total_time
    }, x

# ------------------- دوال قراءة الملفات -------------------
def read_uflp_file(text):
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

def read_general_file(text):
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

def mps_to_ahrh_format_in_memory(mps_content, sense=1):
    """تحويل محتوى ملف MPS إلى بيانات AHRH (c, A, b, n, m)"""
    import pulp
    import tempfile
    import os

    with tempfile.NamedTemporaryFile(mode='w', suffix='.mps', delete=False) as tmp:
        tmp.write(mps_content)
        tmp_path = tmp.name

    try:
        var_dict, problem = pulp.LpProblem.fromMPS(tmp_path, sense=sense)
        vars_list = list(var_dict.values())
        n = len(vars_list)
        constraints_list = list(problem.constraints.values())
        m = len(constraints_list)

        c = np.zeros(n)
        for i, var in enumerate(vars_list):
            if var in problem.objective:
                c[i] = problem.objective.get(var, 0.0)

        A = np.zeros((m, n))
        b = np.zeros(m)

        for i, (con_name, constraint) in enumerate(problem.constraints.items()):
            b[i] = -constraint.constant
            for var, coeff in constraint.items():
                if var in var_dict.values():
                    var_index = list(var_dict.values()).index(var)
                    A[i, var_index] = coeff

        return c, A, b, n, m
    except Exception as e:
        raise ValueError(f"فشل تحويل ملف MPS: {e}")
    finally:
        os.unlink(tmp_path)

# ------------------- واجهة Streamlit -------------------
st.set_page_config(page_title="AHRH Solver - ILP & UFLP", layout="wide")

col1, col2 = st.columns([4, 1])
with col2:
    language = st.selectbox("", ['English', 'Français', 'العربية'], key='language_selector')
    st.session_state.language = language

st.title(t('app_title'))

# وصف توضيحي متعدد اللغات
st.markdown(t('intro'))

st.markdown(t('app_desc'))
st.markdown(f"- {t('feature1')}\n- {t('feature2')}\n- {t('feature3')}\n- {t('feature4')}\n- {t('feature5')}\n- {t('feature6')}\n- {t('feature7')}")

with st.sidebar:
    st.header(t('sidebar_algo'))
    max_cycles = st.slider(t('max_cycles'), 5, 150, 15, 5)
    k_coarse = st.slider(t('k_coarse'), 3, 10, 5)
    patience = st.slider(t('patience'), 2, 10, 3)

    use_tiny_improve = st.checkbox(t('use_tiny_improve'), value=True)
    if use_tiny_improve:
        tiny_patience = st.number_input(t('tiny_patience'), min_value=2, max_value=10, value=3)
        tiny_improve_threshold = st.number_input(t('tiny_improve_threshold'), value=1e-3, format="%.0e", step=1e-3)
    else:
        tiny_patience = 3
        tiny_improve_threshold = 1e-3

    st.subheader(t('sidebar_stop'))
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

    st.markdown("---")
    st.subheader(t('control_panel'))
    col_p1, col_p2 = st.columns(2)
    with col_p1:
        if st.button(t('pause_button')):
            st.session_state.paused = True
    with col_p2:
        if st.button(t('resume_button')):
            st.session_state.paused = False
            if st.session_state.resume_state is not None and st.session_state.last_params is not None:
                params = st.session_state.last_params
                with st.spinner("استئناف التشغيل..."):
                    if params['type'] == 'uflp':
                        result = solve_ahrh_uflp(
                            params['f'], params['c'],
                            params['max_cycles'], params['k_coarse'], params['patience'], params['tiny_patience'],
                            params['use_R'], params['R_tol'], params['stable_gap_needed'],
                            params['use_cost_repeat'], params['cost_repeat_times'],
                            params['use_gap_repeat'], params['gap_repeat_times'],
                            params['use_contraction'], params['diff_tol'],
                            params['use_tiny_improve'], params['tiny_improve_threshold'],
                            resume_state=st.session_state.resume_state
                        )
                        st.session_state.result = result
                    else:  # ilp
                        result, x = solve_ahrh_general_ilp(
                            params['c'], params['A'], params['b'],
                            params['max_cycles'], params['k_coarse'], params['patience'], params['tiny_patience'],
                            params['use_R'], params['R_tol'], params['stable_gap_needed'],
                            params['use_cost_repeat'], params['cost_repeat_times'],
                            params['use_gap_repeat'], params['gap_repeat_times'],
                            params['use_contraction'], params['diff_tol'],
                            params['use_tiny_improve'], params['tiny_improve_threshold'],
                            resume_state=st.session_state.resume_state
                        )
                        st.session_state.result = result
                        st.session_state.x = x
                    st.rerun()

# تبويبان فقط
tab1, tab2 = st.tabs([t('tab_upload'), t('tab_manual')])

with tab1:
    st.header(t('upload_header'))
    st.info(t('upload_info'))

    file_type = st.radio(
        t('file_type'),
        [t('uflp'), t('ilp'), "MPS"],
        key="file_type_radio"
    )
    is_uflp = (file_type == t('uflp'))
    is_ilp = (file_type == t('ilp'))
    is_mps = (file_type == "MPS")

    with st.expander("📄 مساعدة حول تنسيق الملف"):
        if is_uflp:
            st.markdown(t('uflp_format'))
        elif is_ilp:
            st.markdown(t('general_format'))
        else:
            st.markdown(t('mps_format'))

    uploaded_file = st.file_uploader(t('choose_file'), type=['txt', 'dat', 'mps', 'lp', ''])

    if uploaded_file is not None:
        with st.spinner("جارٍ معالجة الملف..."):
            try:
                text = uploaded_file.getvalue().decode("utf-8")
            except:
                text = uploaded_file.getvalue().decode("latin-1")

            try:
                if is_uflp:
                    f, c_mat, n, m = read_uflp_file(text)
                    st.success(f"تم تحميل ملف UFLP: {n} موقع، {m} عميل")
                    st.session_state.last_params = {
                        'type': 'uflp',
                        'f': f,
                        'c': c_mat,
                        'max_cycles': max_cycles,
                        'k_coarse': k_coarse,
                        'patience': patience,
                        'tiny_patience': tiny_patience,
                        'use_R': use_R,
                        'R_tol': R_tol,
                        'stable_gap_needed': stable_gap_needed,
                        'use_cost_repeat': use_cost_repeat,
                        'cost_repeat_times': cost_repeat_times,
                        'use_gap_repeat': use_gap_repeat,
                        'gap_repeat_times': gap_repeat_times,
                        'use_contraction': use_contraction,
                        'diff_tol': diff_tol,
                        'use_tiny_improve': use_tiny_improve,
                        'tiny_improve_threshold': tiny_improve_threshold,
                    }
                    result = solve_ahrh_uflp(
                        f, c_mat,
                        max_cycles, k_coarse, patience, tiny_patience,
                        use_R, R_tol, stable_gap_needed,
                        use_cost_repeat, cost_repeat_times,
                        use_gap_repeat, gap_repeat_times,
                        use_contraction, diff_tol,
                        use_tiny_improve, tiny_improve_threshold,
                        resume_state=None
                    )
                    st.session_state.result = result
                    st.session_state.n = n
                    st.session_state.m = m

                elif is_ilp:
                    c, A, b, n, m = read_general_file(text)
                    st.success(f"تم تحميل ملف ILP عام: {n} متغير، {m} قيد")
                    st.session_state.last_params = {
                        'type': 'ilp',
                        'c': c,
                        'A': A,
                        'b': b,
                        'max_cycles': max_cycles,
                        'k_coarse': k_coarse,
                        'patience': patience,
                        'tiny_patience': tiny_patience,
                        'use_R': use_R,
                        'R_tol': R_tol,
                        'stable_gap_needed': stable_gap_needed,
                        'use_cost_repeat': use_cost_repeat,
                        'cost_repeat_times': cost_repeat_times,
                        'use_gap_repeat': use_gap_repeat,
                        'gap_repeat_times': gap_repeat_times,
                        'use_contraction': use_contraction,
                        'diff_tol': diff_tol,
                        'use_tiny_improve': use_tiny_improve,
                        'tiny_improve_threshold': tiny_improve_threshold,
                    }
                    result, x = solve_ahrh_general_ilp(
                        c, A, b,
                        max_cycles, k_coarse, patience, tiny_patience,
                        use_R, R_tol, stable_gap_needed,
                        use_cost_repeat, cost_repeat_times,
                        use_gap_repeat, gap_repeat_times,
                        use_contraction, diff_tol,
                        use_tiny_improve, tiny_improve_threshold,
                        resume_state=None
                    )
                    st.session_state.result = result
                    st.session_state.x = x
                    st.session_state.n = n
                    st.session_state.m = m

                else:  # MPS
                    c, A, b, n, m = mps_to_ahrh_format_in_memory(text, sense=1)
                    st.success(f"تم تحميل ملف MPS: {n} متغير، {m} قيد")
                    st.session_state.last_params = {
                        'type': 'ilp',
                        'c': c,
                        'A': A,
                        'b': b,
                        'max_cycles': max_cycles,
                        'k_coarse': k_coarse,
                        'patience': patience,
                        'tiny_patience': tiny_patience,
                        'use_R': use_R,
                        'R_tol': R_tol,
                        'stable_gap_needed': stable_gap_needed,
                        'use_cost_repeat': use_cost_repeat,
                        'cost_repeat_times': cost_repeat_times,
                        'use_gap_repeat': use_gap_repeat,
                        'gap_repeat_times': gap_repeat_times,
                        'use_contraction': use_contraction,
                        'diff_tol': diff_tol,
                        'use_tiny_improve': use_tiny_improve,
                        'tiny_improve_threshold': tiny_improve_threshold,
                    }
                    result, x = solve_ahrh_general_ilp(
                        c, A, b,
                        max_cycles, k_coarse, patience, tiny_patience,
                        use_R, R_tol, stable_gap_needed,
                        use_cost_repeat, cost_repeat_times,
                        use_gap_repeat, gap_repeat_times,
                        use_contraction, diff_tol,
                        use_tiny_improve, tiny_improve_threshold,
                        resume_state=None
                    )
                    st.session_state.result = result
                    st.session_state.x = x
                    st.session_state.n = n
                    st.session_state.m = m

            except Exception as e:
                st.error(f"خطأ في قراءة الملف: {e}")

with tab2:
    st.header(t('manual_header'))
    st.warning(t('manual_warning'))
    col1, col2 = st.columns(2)
    with col1:
        n_man = st.number_input(t('manual_n'), min_value=1, max_value=10, value=3, step=1, key="n_man")
    with col2:
        m_man = st.number_input(t('manual_m'), min_value=1, max_value=10, value=3, step=1, key="m_man")

    prob_type_manual = st.radio("", [t('uflp'), t('ilp')], key="manual_prob_type")
    is_manual_uflp = (prob_type_manual == t('uflp'))

    if is_manual_uflp:
        # إدخال UFLP يدوي
        if 'f_man' not in st.session_state or st.session_state.get('n_man_prev') != n_man:
            st.session_state['f_man'] = np.zeros(n_man)
            st.session_state['n_man_prev'] = n_man
        if 'c_man' not in st.session_state or st.session_state.get('n_man_prev') != n_man or st.session_state.get('m_man_prev') != m_man:
            st.session_state['c_man'] = np.zeros((n_man, m_man))
            st.session_state['m_man_prev'] = m_man

        st.subheader(t('manual_f'))
        f_vals = []
        cols = st.columns(min(5, n_man))
        for i in range(n_man):
            with cols[i % 5]:
                val = st.number_input(f"f[{i}]", value=float(st.session_state['f_man'][i]), key=f"f_man_{i}")
                f_vals.append(val)
        st.session_state['f_man'] = np.array(f_vals)

        st.subheader(t('manual_c'))
        c_vals = np.zeros((n_man, m_man))
        for i in range(n_man):
            st.write(f"**{i}:**")
            cols = st.columns(min(5, m_man))
            for j in range(m_man):
                with cols[j % 5]:
                    val = st.number_input(f"c[{i}][{j}]", value=float(st.session_state['c_man'][i, j]), key=f"c_man_{i}_{j}")
                    c_vals[i, j] = val
        st.session_state['c_man'] = c_vals

        if st.button(t('solve_button'), key="solve_manual"):
            with st.spinner("Running algorithm..."):
                st.session_state.last_params = {
                    'type': 'uflp',
                    'f': st.session_state['f_man'],
                    'c': st.session_state['c_man'],
                    'max_cycles': max_cycles,
                    'k_coarse': k_coarse,
                    'patience': patience,
                    'tiny_patience': tiny_patience,
                    'use_R': use_R,
                    'R_tol': R_tol,
                    'stable_gap_needed': stable_gap_needed,
                    'use_cost_repeat': use_cost_repeat,
                    'cost_repeat_times': cost_repeat_times,
                    'use_gap_repeat': use_gap_repeat,
                    'gap_repeat_times': gap_repeat_times,
                    'use_contraction': use_contraction,
                    'diff_tol': diff_tol,
                    'use_tiny_improve': use_tiny_improve,
                    'tiny_improve_threshold': tiny_improve_threshold,
                }
                result = solve_ahrh_uflp(
                    st.session_state['f_man'], st.session_state['c_man'],
                    max_cycles, k_coarse, patience, tiny_patience,
                    use_R, R_tol, stable_gap_needed,
                    use_cost_repeat, cost_repeat_times,
                    use_gap_repeat, gap_repeat_times,
                    use_contraction, diff_tol,
                    use_tiny_improve, tiny_improve_threshold,
                    resume_state=None
                )
                st.session_state.result = result
                st.session_state.n = n_man
                st.session_state.m = m_man
                st.success("Done!")
    else:
        # إدخال ILP عام يدوي
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

        if st.button(t('solve_button'), key="solve_manual_ilp"):
            with st.spinner("Running algorithm..."):
                st.session_state.last_params = {
                    'type': 'ilp',
                    'c': st.session_state.c_man,
                    'A': st.session_state.A_man,
                    'b': st.session_state.b_man,
                    'max_cycles': max_cycles,
                    'k_coarse': k_coarse,
                    'patience': patience,
                    'tiny_patience': tiny_patience,
                    'use_R': use_R,
                    'R_tol': R_tol,
                    'stable_gap_needed': stable_gap_needed,
                    'use_cost_repeat': use_cost_repeat,
                    'cost_repeat_times': cost_repeat_times,
                    'use_gap_repeat': use_gap_repeat,
                    'gap_repeat_times': gap_repeat_times,
                    'use_contraction': use_contraction,
                    'diff_tol': diff_tol,
                    'use_tiny_improve': use_tiny_improve,
                    'tiny_improve_threshold': tiny_improve_threshold,
                }
                result, x = solve_ahrh_general_ilp(
                    st.session_state.c_man, st.session_state.A_man, st.session_state.b_man,
                    max_cycles, k_coarse, patience, tiny_patience,
                    use_R, R_tol, stable_gap_needed,
                    use_cost_repeat, cost_repeat_times,
                    use_gap_repeat, gap_repeat_times,
                    use_contraction, diff_tol,
                    use_tiny_improve, tiny_improve_threshold,
                    resume_state=None
                )
                st.session_state.result = result
                st.session_state.x = x
                st.session_state.n = n_man
                st.session_state.m = m_man
                st.success("Done!")

# ------------------- عرض النتائج -------------------
st.markdown("---")
st.header(t('results'))

if 'result' in st.session_state:
    res = st.session_state['result']
    if 'open_fac' in res:  # UFLP
        colA, colB, colC, colD = st.columns(4)
        colA.metric(t('best_cost'), f"{res['best_cost']:,.0f}")
        colB.metric(t('lp_val'), f"{res['lp_val']:,.0f}")
        colC.metric(t('gap'), f"{res['gap']:.4f}%")
        colD.metric(t('open_fac'), res['open_fac'])
    else:  # ILP
        colA, colB, colC = st.columns(3)
        colA.metric(t('best_cost'), f"{res['best_cost']:,.2f}")
        colB.metric(t('lp_val'), f"{res['lp_val']:,.2f}")
        colC.metric(t('gap'), f"{res['gap']:.4f}%")

    colE, colF, colG = st.columns(3)
    colE.metric(t('cycles_done'), res['cycles_done'])
    colF.metric(t('time'), f"{res['total_time']:.2f}")
    colG.metric(t('size'), f"{st.session_state['n']}×{st.session_state['m']}")

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
            'best_so_far': t('best_so_far')
        })
        df_cycles[t('improved')] = df_cycles[t('improved')].apply(lambda x: t('yes') if x else t('no'))
        st.dataframe(df_cycles, use_container_width=True)

        # زر تحميل CSV
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

        # زر حفظ على Google Drive (تعليمات)
        with st.expander(t('save_drive')):
            st.markdown(t('save_drive_instructions'))
            st.download_button(
                label="📥 تحميل ملف CSV للحفظ",
                data=csv,
                file_name=f"ahrh_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
else:
    st.info(t('info_placeholder'))

st.markdown("---")
st.caption(t('footer'))
```
