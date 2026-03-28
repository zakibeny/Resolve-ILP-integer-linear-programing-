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
import base64
import os
warnings.filterwarnings("ignore")

# ------------------- معلومات الاتصال -------------------
CONTACT_EMAIL = "zakibeny@gmail.com"
CONTACT_PHONE = "0021355614305 / 00213779679073"
CONTACT_FAX = "036495241"

# ------------------- إعدادات التوازي -------------------
NUM_WORKERS = 4

# ------------------- ترجمة النصوص (عربي/إنجليزي/فرنسي/روسي) -------------------
translations = {
    'العربية': {
        'app_title': '🧠 MARIA: خوارزمية البرمجة الخطية الصحيحة ILP UFPS mbs.transport',
        'app_desc': 'هذا التطبيق يطبق خوارزمية MARIA المتقدمة التي تجمع بين:',
        'feature1': 'المسح الشعاعي الهرمي مع اتجاهات موجهة',
        'feature2': 'الرفع الهرمي للاتجاهات',
        'feature3': 'إزاحة الاسترخاء الديناميكية',
        'feature4': 'بحث محلي متقدم بأنماط تبادل متعددة (1-1، 2-1، 1-2، 2-2)',
        'feature5': 'توازي الحسابات لتسريع الأداء',
        'feature6': 'معايير توقف متعددة قابلة للاختيار',
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
        'footer': 'تم التطوير بواسطة Zakarya Benregreg - خوارزمية MARIA محمية ببراءة اختراع.',
        'acceleration_on': '⚡ وضع التسريع مُفعّل (فجوة < 2% و R < 0.01)',
        'acceleration_off': 'وضع التسريع غير مُفعّل',
        'feedback_section': '💬 تواصل معنا - أرسل تعليقك',
        'feedback_placeholder': 'اكتب تعليقك هنا... (سيتم إرساله كـ Issue في GitHub)',
        'feedback_submit': 'إرسال التعليق',
        'feedback_success': '✅ تم الإرسال بنجاح! يمكنك متابعة الـ issue على الرابط:',
        'feedback_error': '❌ فشل الإرسال:',
        'feedback_missing_token': '⚠️ خدمة إرسال التعليقات غير مفعلة حالياً.',
        'feedback_warning': 'الرجاء كتابة تعليق قبل الإرسال.',
        'progress_text': 'التقدم:',
        'estimated_time': 'الوقت المقدر للانتهاء:',
        'seconds': 'ثانية',
        'processing_file': 'جاري معالجة الملف وتشغيل الخوارزمية...',
        'file_processed': 'تمت معالجة الملف وتطبيق الخوارزمية.',
        'constraint_label': 'القيود',
    },
    'English': {
        'app_title': '🧠 MARIA: Integer Linear Programming ILP UFPS mbs.transport',
        'app_desc': 'This app implements the MARIA algorithm, combining:',
        'feature1': 'Hierarchical radial scan with biased directions',
        'feature2': 'Hierarchical direction lifting',
        'feature3': 'Dynamic relaxation shift',
        'feature4': 'Advanced local search (1-1, 2-1, 1-2, 2-2 swaps)',
        'feature5': 'Parallel computing for speed',
        'feature6': 'Multiple customizable stopping criteria',
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
        'footer': 'Developed by Zakarya Benregreg - MARIA algorithm patented.',
        'acceleration_on': '⚡ Acceleration mode ON (gap < 2% and R < 0.01)',
        'acceleration_off': 'Acceleration mode OFF',
        'feedback_section': '💬 Contact Us - Send your feedback',
        'feedback_placeholder': 'Write your comment here... (will be sent as a GitHub Issue)',
        'feedback_submit': 'Send Feedback',
        'feedback_success': '✅ Sent successfully! You can track the issue at:',
        'feedback_error': '❌ Sending failed:',
        'feedback_missing_token': '⚠️ Feedback service is currently disabled.',
        'feedback_warning': 'Please write a comment before sending.',
        'progress_text': 'Progress:',
        'estimated_time': 'Estimated time remaining:',
        'seconds': 'seconds',
        'processing_file': 'Processing file and running algorithm...',
        'file_processed': 'File processed and algorithm applied.',
        'constraint_label': 'Constraint',
    },
    'Français': {
        'app_title': '🧠 MARIA: Programmation Linéaire en Nombres Entiers ILP UFPS mbs.transport',
        'app_desc': 'Cette application implémente l\'algorithme MARIA, combinant :',
        'feature1': 'Balayage radial hiérarchique avec directions orientées',
        'feature2': 'Relèvement hiérarchique des directions',
        'feature3': 'Décalage dynamique de relaxation',
        'feature4': 'Recherche locale avancée (échanges 1-1, 2-1, 1-2, 2-2)',
        'feature5': 'Calcul parallèle pour la rapidité',
        'feature6': 'Critères d\'arrêt multiples personnalisables',
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
        'manual_warning': 'Pour petits problèmes seulement (n ≤ 10, m ≤ 10)',
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
        'footer': 'Développé par Zakarya Benregreg - Algorithme MARIA breveté.',
        'acceleration_on': '⚡ Mode accélération ACTIVÉ (gap < 2% et R < 0.01)',
        'acceleration_off': 'Mode accélération DÉSACTIVÉ',
        'feedback_section': '💬 Contactez-nous - Envoyez votre commentaire',
        'feedback_placeholder': 'Écrivez votre commentaire ici... (sera envoyé comme Issue GitHub)',
        'feedback_submit': 'Envoyer',
        'feedback_success': '✅ Envoyé avec succès ! Suivez l\'issue sur :',
        'feedback_error': '❌ Échec de l\'envoi :',
        'feedback_missing_token': '⚠️ Le service de commentaires est actuellement désactivé.',
        'feedback_warning': 'Veuillez écrire un commentaire avant d\'envoyer.',
        'progress_text': 'Progression :',
        'estimated_time': 'Temps estimé restant :',
        'seconds': 'secondes',
        'processing_file': 'Traitement du fichier et exécution de l\'algorithme...',
        'file_processed': 'Fichier traité et algorithme appliqué.',
        'constraint_label': 'Contrainte',
    },
    'Русский': {
        'app_title': '🧠 MARIA: Целочисленное линейное программирование ILP UFPS mbs.transport',
        'app_desc': 'Это приложение реализует алгоритм MARIA, объединяющий:',
        'feature1': 'Иерархическое радиальное сканирование с направленными направлениями',
        'feature2': 'Иерархический подъём направлений',
        'feature3': 'Динамический сдвиг релаксации',
        'feature4': 'Продвинутый локальный поиск (обмены 1-1, 2-1, 1-2, 2-2)',
        'feature5': 'Параллельные вычисления для скорости',
        'feature6': 'Несколько настраиваемых критериев остановки',
        'problem_type': 'Тип задачи',
        'ilp': 'Общее целочисленное программирование (ILP)',
        'uflp': 'Размещение объектов (UFLP)',
        'sidebar_algo': '⚙️ Параметры алгоритма',
        'max_cycles': 'Макс. циклов',
        'k_coarse': 'Размер грубого набора (k)',
        'patience': 'Терпение (циклов без улучшения)',
        'sidebar_stop': '⏹️ Критерии остановки',
        'choose_criteria': 'Выберите любую комбинацию (алгоритм останавливается при выполнении любого условия):',
        'use_R': 'Использовать порог R (с устойчивостью разрыва)',
        'R_tol': 'Допуск R (ε)',
        'stable_gap': 'Требуемые циклы устойчивости разрыва',
        'use_cost_repeat': 'Использовать повторение стоимости',
        'cost_repeat_times': 'Количество повторений',
        'use_gap_repeat': 'Использовать повторение разрыва',
        'gap_repeat_times': 'Количество повторений',
        'use_contraction': 'Использовать критерий сжатия (diff + R)',
        'diff_tol': 'Допуск разницы решений (ε₁)',
        'workers': 'Рабочие (параллельные потоки)',
        'tab_upload': '📂 Загрузить файл',
        'tab_manual': '✍️ Ручной ввод',
        'upload_header': 'Загрузить файл задачи',
        'upload_info': 'Поддерживает файлы UFLP (например, gs250) и общие файлы ILP.',
        'choose_file': 'Выберите файл',
        'file_type': 'Тип файла',
        'uflp_file': 'Файл UFLP (n m 0 + данные)',
        'ilp_file': 'Общий файл ILP (n m + c + A + b)',
        'ilp_format_help': """
**Формат общего файла ILP:**
- Первая строка: `n m` (количество переменных, количество ограничений)
- Вторая строка: коэффициенты цели c (n значений)
- Затем m строк: матрица ограничений A (n значений в строке)
- Последняя строка: правая часть b (m значений)
""",
        'uflp_format_help': """
**Формат файла UFLP:**
- Первая строка: `n m 0` (количество объектов, количество клиентов)
- Затем n строк: `индекс_объекта` `стоимость_открытия` `стоимости_транспорта до m клиентов`
""",
        'manual_header': 'Ручной ввод данных',
        'manual_warning': 'Только для небольших задач (n ≤ 10, m ≤ 10)',
        'manual_n': 'Количество переменных (n)',
        'manual_m': 'Количество ограничений (m)',
        'manual_c': 'Коэффициенты цели c[i]',
        'manual_A': 'Матрица ограничений A[i][j]',
        'manual_b': 'Правая часть b[i]',
        'solve_button': '🚀 Решить введённую задачу',
        'results': '📊 Результаты',
        'best_cost': 'Лучшая стоимость',
        'lp_val': 'Значение LP',
        'gap': 'Разрыв',
        'open_fac': 'Открытые объекты',
        'cycles_done': 'Выполнено циклов',
        'time': 'Время (с)',
        'size': 'Размер задачи',
        'stop_reason': 'Причина остановки',
        'gap_plot': '📈 Эволюция разрыва и R',
        'gap_label': 'Разрыв (%)',
        'R_label': 'R',
        'cycle_log': '📋 Журнал циклов',
        'cycle': 'Цикл',
        'cost': 'Стоимость',
        'improved': 'Улучшение?',
        'best_so_far': 'Лучший на данный момент',
        'yes': 'Да',
        'no': 'Нет',
        'download': '📥 Скачать эволюцию (CSV)',
        'info_placeholder': '👈 Выберите источник данных и нажмите кнопку запуска.',
        'footer': 'Разработано Закарией Бенрегрег - Алгоритм MARIA запатентован.',
        'acceleration_on': '⚡ Режим ускорения ВКЛЮЧЁН (разрыв < 2% и R < 0.01)',
        'acceleration_off': 'Режим ускорения ВЫКЛЮЧЁН',
        'feedback_section': '💬 Свяжитесь с нами - Отправьте отзыв',
        'feedback_placeholder': 'Напишите ваш комментарий здесь... (будет отправлен как Issue на GitHub)',
        'feedback_submit': 'Отправить отзыв',
        'feedback_success': '✅ Отправлено успешно! Следите за задачей по ссылке:',
        'feedback_error': '❌ Ошибка отправки:',
        'feedback_missing_token': '⚠️ Сервис отзывов в настоящее время отключён.',
        'feedback_warning': 'Пожалуйста, напишите комментарий перед отправкой.',
        'progress_text': 'Прогресс:',
        'estimated_time': 'Оценочное оставшееся время:',
        'seconds': 'секунд',
        'processing_file': 'Обработка файла и запуск алгоритма...',
        'file_processed': 'Файл обработан, алгоритм применён.',
        'constraint_label': 'Ограничение',
    }
}

def t(key):
    return translations[st.session_state.language][key]

# ------------------- دوال الخوارزمية (من الكود الأصلي) -------------------
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

# دوال UFLP
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

def solve_ahrh_uflp(f, c, max_cycles, k_coarse, patience,
                    use_R, R_tol, stable_gap_needed,
                    use_cost_repeat, cost_repeat_times,
                    use_gap_repeat, gap_repeat_times,
                    use_contraction, diff_tol,
                    progress_callback=None):
    n, m = len(f), c.shape[1]
    y_lp, lp_val = lp_relaxation_uflp(f, c)
    if lp_val is None:
        lp_val = float('inf')

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

        if acceleration_active:
            st.session_state.acceleration = True
        else:
            st.session_state.acceleration = False

        new_cost, new_y = vcycle_uflp(y, f, c, coarse, y_lp=y_lp, gap_threshold=3.0)
        gap = (new_cost - lp_val) / lp_val * 100 if lp_val != float('inf') else 0
        R_val = compute_R(new_y)
        diff = np.linalg.norm(new_y - last_y)

        improved = new_cost < best - 1e-6
        if improved:
            best = new_cost
            y = new_y
            no_improve = 0
        else:
            no_improve += 1

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

        if progress_callback:
            progress_callback(cycle, max_cycles, best, gap, R_val)

        if not acceleration_active and gap < 2.0 and R_val < 0.01:
            acceleration_active = True
            st.session_state.acceleration = True

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
        last_y = new_y.copy()

        if stop_now:
            cycles_done = cycle + 1
            break

    if cycles_done == 0:
        cycles_done = max_cycles
        stop_reason = f"Max cycles ({max_cycles}) reached"

    total_time = time.time() - start_time

    return {
        'best_cost': best,
        'lp_val': lp_val,
        'gap': (best - lp_val) / lp_val * 100 if lp_val != float('inf') else 0,
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

# دوال ILP العامة
def solve_lp_pulp_integer(c, A, b):
    n = len(c)
    m = len(b)
    prob = pulp.LpProblem("LP_Relax", pulp.LpMinimize)
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

def solve_ahrh_general_ilp(c, A, b, max_cycles=20, k_coarse=5, patience=5,
                           use_R=False, R_tol=1e-6, stable_gap_needed=2,
                           use_cost_repeat=False, cost_repeat_times=2,
                           use_gap_repeat=False, gap_repeat_times=2,
                           use_contraction=False, diff_tol=1e-12,
                           progress_callback=None):
    n = len(c)
    m = len(b)

    x_lp, lp_val = solve_lp_pulp_integer(c, A, b)
    if x_lp is None:
        return None, None, None

    R_initial = compute_R(x_lp)
    R = R_initial if R_initial > 0 else 1.0

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
    last_cost = None
    last_gap = None
    last_R = None
    last_x = x.copy()

    start_time = time.time()

    for cycle in range(max_cycles):
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
            best_cost = new_cost
            x = new_x
            no_improve = 0
        else:
            no_improve += 1

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

        if progress_callback:
            progress_callback(cycle, max_cycles, best_cost, gap, R_val)

        if not acceleration_active and gap < 2.0 and R_val < 0.01:
            acceleration_active = True

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

    total_time = time.time() - start_time

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

# دوال قراءة الملفات
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

def read_ilp_file(text):
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

def generate_random_ilp(n, m):
    c = np.random.uniform(1, 10, n)
    A = np.random.uniform(0, 5, (m, n))
    b = np.random.uniform(5, 20, m)
    return c, A, b

def generate_random_uflp(n, m):
    f = np.random.uniform(1000, 20000, n)
    c = np.random.uniform(100, 500, (n, m))
    return f, c

# دوال GitHub
def send_to_github_issue(comment, repo_owner, repo_name, token):
    url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/issues"
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json"
    }
    title = f"تعليق من مستخدم في {datetime.now().strftime('%Y-%m-%d %H:%M')}"
    data = {"title": title, "body": comment}
    try:
        response = requests.post(url, headers=headers, data=json.dumps(data))
        if response.status_code == 201:
            return True, response.json().get("html_url")
        else:
            return False, f"خطأ {response.status_code}: {response.text}"
    except Exception as e:
        return False, str(e)

# ----------------------------------------------------------------------
# واجهة Streamlit مع التصميم المطلوب
# ----------------------------------------------------------------------
st.set_page_config(page_title="MARIA Solver", layout="wide")

# تهيئة اللغة (الإنجليزية افتراضياً)
if 'language' not in st.session_state:
    st.session_state.language = 'English'

# ------------------- CSS لتحسين القراءة -------------------
st.markdown("""
<style>
    /* تعتيم الخلفية */
    .stApp::before {
        content: "";
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background-color: rgba(0, 0, 0, 0.65);
        z-index: -1;
    }
    .stApp {
        background-color: transparent !important;
    }
    /* خلفية بيضاء شفافة لجميع العناصر النصية */
    div[data-testid="stMarkdownContainer"], 
    .stMarkdown, 
    .stText, 
    .stNumberInput, 
    .stSelectbox, 
    .stTextInput, 
    .stTextArea, 
    .stButton button,
    .stAlert,
    .stSuccess,
    .stError,
    .stWarning,
    .stInfo,
    .stExpander,
    .stTabs,
    .stTab,
    .element-container,
    .stHeader,
    .stSubheader,
    .stCaption,
    .stForm {
        background-color: rgba(255, 255, 255, 0.9) !important;
        border-radius: 12px;
        padding: 8px 12px;
        margin: 8px 0;
        color: #000000 !important;
        font-size: 18px;
        line-height: 1.5;
        backdrop-filter: blur(2px);
    }
    h1, h2, h3, h4, h5, h6 {
        background-color: rgba(255, 255, 255, 0.9) !important;
        border-radius: 12px;
        padding: 8px 12px;
        margin: 12px 0;
        color: #000000 !important;
        font-weight: bold;
        font-size: 1.5em;
    }
    .stApp > header, .stApp > .main {
        background: transparent !important;
    }
    input, textarea, select {
        background-color: white !important;
        color: black !important;
        font-size: 16px !important;
        border-radius: 8px;
    }
    .stButton button {
        background-color: #ffffffcc !important;
        color: black !important;
        font-weight: bold;
        border: 1px solid #aaa;
    }
    .stButton button:hover {
        background-color: white !important;
    }
    .stTabs [data-baseweb="tab-list"] button {
        background-color: rgba(255, 255, 255, 0.85) !important;
        color: black !important;
        font-weight: bold;
    }
    /* المربع الأحمر */
    .contact-box {
        background-color: #ffeeee !important;
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        margin: 20px 0;
        border: 3px solid red;
        color: red;
    }
    .contact-box span {
        color: red;
        font-size: 32px;
        font-weight: bold;
    }
    body, .stApp, div, p, span, label {
        font-size: 18px !important;
        font-weight: 500 !important;
    }
</style>
""", unsafe_allow_html=True)

# خلفية ثابتة
BACKGROUND_IMAGE = "background.png"
if os.path.exists(BACKGROUND_IMAGE):
    with open(BACKGROUND_IMAGE, "rb") as f:
        img_base64 = base64.b64encode(f.read()).decode()
    if BACKGROUND_IMAGE.endswith('.png'):
        mime = "image/png"
    elif BACKGROUND_IMAGE.endswith('.jpg') or BACKGROUND_IMAGE.endswith('.jpeg'):
        mime = "image/jpeg"
    else:
        mime = "image/png"
    st.markdown(f"""
    <style>
    .stApp {{
        background-image: url("data:{mime};base64,{img_base64}");
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
        background-color: transparent;
    }}
    </style>
    """, unsafe_allow_html=True)
else:
    st.sidebar.warning(f"⚠️ لم يتم العثور على ملف الخلفية '{BACKGROUND_IMAGE}'. سيتم استخدام خلفية بيضاء.")

# مربع معلومات الاتصال
st.markdown(f"""
<div class="contact-box">
    <span>✉️ {CONTACT_EMAIL}</span><br>
    <span>📞 {CONTACT_PHONE}</span><br>
    <span>📠 {CONTACT_FAX}</span>
</div>
""", unsafe_allow_html=True)

# رأس الصفحة مع اختيار اللغة
col1, col2 = st.columns([4, 1])
with col1:
    st.title(t('app_title'))
with col2:
    lang = st.selectbox("", ['English', 'Français', 'العربية', 'Русский'], key='lang_selector')
    st.session_state.language = lang

# وصف الخوارزمية
st.markdown(t('app_desc'))
st.markdown(f"- {t('feature1')}\n- {t('feature2')}\n- {t('feature3')}\n- {t('feature4')}\n- {t('feature5')}\n- {t('feature6')}")

# شريط جانبي للمعاملات
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

# اختيار نوع المسألة
problem_type = st.radio(t('problem_type'), [t('ilp'), t('uflp')])
is_uflp = (problem_type == t('uflp'))

tab1, tab2 = st.tabs([t('tab_upload'), t('tab_manual')])

# ------------------- التبويب 1: رفع ملف -------------------
with tab1:
    st.header(t('upload_header'))
    st.info(t('upload_info'))
    with st.expander("📄 مساعدة حول تنسيق الملف"):
        if is_uflp:
            st.markdown(t('uflp_format_help'))
        else:
            st.markdown(t('ilp_format_help'))
    
    uploaded_file = st.file_uploader(t('choose_file'), type=None)
    if uploaded_file is not None:
        with st.spinner(t('processing_file')):
            try:
                text = uploaded_file.getvalue().decode("utf-8")
            except:
                text = uploaded_file.getvalue().decode("latin-1")
            try:
                if is_uflp:
                    f, c, n, m = read_uflp_file(text)
                    st.success(f"File loaded: {n} facilities, {m} customers")
                    
                    # عناصر التقدم
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    time_placeholder = st.empty()
                    start_time = time.time()
                    
                    def progress_callback(cycle, total, best_cost, gap, R):
                        percent = (cycle + 1) / total
                        progress_bar.progress(percent)
                        status_text.write(f"**{t('progress_text')}** {percent:.0%} (Cycle {cycle+1}/{total})")
                        elapsed = time.time() - start_time
                        if cycle > 0:
                            avg_time = elapsed / (cycle + 1)
                            remaining = avg_time * (total - cycle - 1)
                            time_placeholder.write(f"{t('estimated_time')} {remaining:.1f} {t('seconds')}")
                    
                    result = solve_ahrh_uflp(
                        f, c,
                        max_cycles, k_coarse, patience,
                        use_R, R_tol, stable_gap_needed,
                        use_cost_repeat, cost_repeat_times,
                        use_gap_repeat, gap_repeat_times,
                        use_contraction, diff_tol,
                        progress_callback=progress_callback
                    )
                    progress_bar.empty()
                    status_text.empty()
                    time_placeholder.empty()
                    st.session_state['result'] = result
                    st.session_state['n'] = n
                    st.session_state['m'] = m
                else:
                    c, A, b, n, m = read_ilp_file(text)
                    st.success(f"File loaded: {n} variables, {m} constraints")
                    
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    time_placeholder = st.empty()
                    start_time = time.time()
                    
                    def progress_callback(cycle, total, best_cost, gap, R):
                        percent = (cycle + 1) / total
                        progress_bar.progress(percent)
                        status_text.write(f"**{t('progress_text')}** {percent:.0%} (Cycle {cycle+1}/{total})")
                        elapsed = time.time() - start_time
                        if cycle > 0:
                            avg_time = elapsed / (cycle + 1)
                            remaining = avg_time * (total - cycle - 1)
                            time_placeholder.write(f"{t('estimated_time')} {remaining:.1f} {t('seconds')}")
                    
                    result, x = solve_ahrh_general_ilp(
                        c, A, b,
                        max_cycles=max_cycles, k_coarse=k_coarse, patience=patience,
                        use_R=use_R, R_tol=R_tol, stable_gap_needed=stable_gap_needed,
                        use_cost_repeat=use_cost_repeat, cost_repeat_times=cost_repeat_times,
                        use_gap_repeat=use_gap_repeat, gap_repeat_times=gap_repeat_times,
                        use_contraction=use_contraction, diff_tol=diff_tol,
                        progress_callback=progress_callback
                    )
                    progress_bar.empty()
                    status_text.empty()
                    time_placeholder.empty()
                    st.session_state['result'] = result
                    st.session_state['n'] = n
                    st.session_state['m'] = m
            except Exception as e:
                st.error(f"{t('upload_error')} {e}")

# ------------------- التبويب 2: إدخال يدوي -------------------
with tab2:
    st.header(t('manual_header'))
    st.warning(t('manual_warning'))
    
    if is_uflp:
        n_man = st.number_input(t('manual_n'), min_value=1, max_value=10, value=3)
        m_man = st.number_input(t('manual_m'), min_value=1, max_value=10, value=3)
        st.subheader("تكاليف فتح المرافق f[i]")
        f_vals = []
        cols = st.columns(min(5, n_man))
        for i in range(n_man):
            with cols[i % 5]:
                val = st.number_input(f"f[{i}]", value=1000.0, key=f"f_man_{i}")
                f_vals.append(val)
        st.subheader("تكاليف النقل c[i][j]")
        c_vals = np.zeros((n_man, m_man))
        for i in range(n_man):
            st.write(f"**الموقع {i}:**")
            cols = st.columns(min(5, m_man))
            for j in range(m_man):
                with cols[j % 5]:
                    val = st.number_input(f"c[{i}][{j}]", value=100.0, key=f"c_man_{i}_{j}")
                    c_vals[i, j] = val
        if st.button(t('solve_button')):
            with st.spinner("جاري الحل..."):
                f = np.array(f_vals)
                c = c_vals
                progress_bar = st.progress(0)
                status_text = st.empty()
                time_placeholder = st.empty()
                start_time = time.time()
                def progress_callback(cycle, total, best_cost, gap, R):
                    percent = (cycle + 1) / total
                    progress_bar.progress(percent)
                    status_text.write(f"**{t('progress_text')}** {percent:.0%} (Cycle {cycle+1}/{total})")
                    elapsed = time.time() - start_time
                    if cycle > 0:
                        avg_time = elapsed / (cycle + 1)
                        remaining = avg_time * (total - cycle - 1)
                        time_placeholder.write(f"{t('estimated_time')} {remaining:.1f} {t('seconds')}")
                result = solve_ahrh_uflp(
                    f, c,
                    max_cycles, k_coarse, patience,
                    use_R, R_tol, stable_gap_needed,
                    use_cost_repeat, cost_repeat_times,
                    use_gap_repeat, gap_repeat_times,
                    use_contraction, diff_tol,
                    progress_callback=progress_callback
                )
                progress_bar.empty()
                status_text.empty()
                time_placeholder.empty()
                st.session_state['result'] = result
                st.session_state['n'] = n_man
                st.session_state['m'] = m_man
                st.success("تم الحل!")
    else:
        n_man = st.number_input(t('manual_n'), min_value=1, max_value=10, value=3)
        m_man = st.number_input(t('manual_m'), min_value=1, max_value=10, value=3)
        st.subheader(t('manual_c'))
        c_vals = []
        cols = st.columns(min(5, n_man))
        for i in range(n_man):
            with cols[i % 5]:
                val = st.number_input(f"c[{i}]", value=0.0, key=f"c_man_{i}")
                c_vals.append(val)
        st.subheader(t('manual_A'))
        A_vals = np.zeros((m_man, n_man))
        for i in range(m_man):
            st.write(f"**{t('constraint_label')} {i+1}:**")
            cols = st.columns(min(5, n_man))
            for j in range(n_man):
                with cols[j % 5]:
                    val = st.number_input(f"A[{i}][{j}]", value=0.0, key=f"A_man_{i}_{j}")
                    A_vals[i, j] = val
        st.subheader(t('manual_b'))
        b_vals = []
        cols = st.columns(min(5, m_man))
        for i in range(m_man):
            with cols[i % 5]:
                val = st.number_input(f"b[{i}]", value=0.0, key=f"b_man_{i}")
                b_vals.append(val)
        error_msg = ""
        if any(c < 0 for c in c_vals):
            error_msg += t('error_negative') + "\n"
        if np.any(A_vals < 0):
            error_msg += t('error_negative') + "\n"
        if any(b < 0 for b in b_vals):
            error_msg += t('error_negative') + "\n"
        if error_msg:
            st.error(error_msg)
        if st.button(t('solve_button')):
            if error_msg:
                st.error(t('error_negative'))
            else:
                with st.spinner("جاري الحل..."):
                    c = np.array(c_vals)
                    A = A_vals
                    b = np.array(b_vals)
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    time_placeholder = st.empty()
                    start_time = time.time()
                    def progress_callback(cycle, total, best_cost, gap, R):
                        percent = (cycle + 1) / total
                        progress_bar.progress(percent)
                        status_text.write(f"**{t('progress_text')}** {percent:.0%} (Cycle {cycle+1}/{total})")
                        elapsed = time.time() - start_time
                        if cycle > 0:
                            avg_time = elapsed / (cycle + 1)
                            remaining = avg_time * (total - cycle - 1)
                            time_placeholder.write(f"{t('estimated_time')} {remaining:.1f} {t('seconds')}")
                    result, x = solve_ahrh_general_ilp(
                        c, A, b,
                        max_cycles=max_cycles, k_coarse=k_coarse, patience=patience,
                        use_R=use_R, R_tol=R_tol, stable_gap_needed=stable_gap_needed,
                        use_cost_repeat=use_cost_repeat, cost_repeat_times=cost_repeat_times,
                        use_gap_repeat=use_gap_repeat, gap_repeat_times=gap_repeat_times,
                        use_contraction=use_contraction, diff_tol=diff_tol,
                        progress_callback=progress_callback
                    )
                    progress_bar.empty()
                    status_text.empty()
                    time_placeholder.empty()
                    st.session_state['result'] = result
                    st.session_state['n'] = n_man
                    st.session_state['m'] = m_man
                    st.success("تم الحل!")

# عرض النتائج
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
    else:
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

# ------------------- قسم GitHub للتعليقات -------------------
st.markdown("---")
st.header(t('feedback_section'))

# قراءة التوكن من secrets (بدون عرض حقول)
try:
    GITHUB_TOKEN = st.secrets["GITHUB_TOKEN"]
    REPO_OWNER = st.secrets.get("REPO_OWNER", "zakibeny")
    REPO_NAME = st.secrets.get("REPO_NAME", "resolve-ilp-integer-linear-programing-")
    token_available = True
except:
    token_available = False
    GITHUB_TOKEN = ""
    REPO_OWNER = "zakibeny"
    REPO_NAME = "resolve-ilp-integer-linear-programing-"

with st.form("feedback_form"):
    user_comment = st.text_area("", height=150, placeholder=t('feedback_placeholder'))
    submitted = st.form_submit_button(t('feedback_submit'))
    if submitted and user_comment.strip():
        if not token_available:
            st.warning(t('feedback_missing_token'))
        else:
            with st.spinner("جاري الإرسال..."):
                success, result = send_to_github_issue(user_comment, REPO_OWNER, REPO_NAME, GITHUB_TOKEN)
                if success:
                    st.success(f"{t('feedback_success')} [{result}]({result})")
                else:
                    st.error(f"{t('feedback_error')} {result}")
    elif submitted:
        st.warning(t('feedback_warning'))

st.markdown("---")
st.caption(t('footer'))
