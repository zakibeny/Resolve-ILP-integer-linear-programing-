import streamlit as st
import numpy as np
import time
import requests
import json
from datetime import datetime
import base64
import os

# ----------------------------------------------------------------------
# ترجمة النصوص (متعددة اللغات)
# ----------------------------------------------------------------------
translations = {
    'العربية': {
        'app_title': '🧠 MARIA: خوارزمية متطورة للحل الأمثل',
        'app_desc': 'هذا التطبيق يطبق خوارزمية MARIA المتقدمة التي تجمع بين:',
        'feature1': 'المسح الشعاعي الهرمي مع اتجاهات موجهة',
        'feature2': 'الرفع الهرمي للاتجاهات',
        'feature3': 'إزاحة الاسترخاء الديناميكية',
        'feature4': 'بحث محلي متقدم بأنماط تبادل متعددة (1-1، 2-1، 1-2، 2-2)',
        'feature5': 'توازي الحسابات لتسريع الأداء',
        'feature6': 'معايير توقف متعددة قابلة للاختيار',
        'tab_upload': '📂 رفع ملف',
        'tab_manual': '✍️ إدخال يدوي',
        'upload_header': 'رفع ملف المسألة',
        'choose_file': 'اختر ملف المسألة',
        'upload_success': 'تم رفع الملف بنجاح!',
        'upload_error': 'خطأ في قراءة الملف:',
        'manual_header': 'إدخال بيانات المسألة يدويًا',
        'manual_warning': 'للمسائل الصغيرة فقط (n ≤ 10, m ≤ 10)',
        'c_coeff': 'معاملات الهدف c[i]',
        'A_matrix': 'مصفوفة القيود A[i][j]',
        'b_rhs': 'الطرف الأيمن b[i]',
        'solve_btn': '🚀 حل المسألة المدخلة',
        'error_negative': '⚠️ القيم يجب أن تكون غير سالبة.',
        'solution_success': '✅ تم حل المسألة بنجاح! (عرض توضيحي)',
        'feedback_section': '💬 تواصل معنا - أرسل تعليقك',
        'comment_placeholder': 'اكتب تعليقك هنا...',
        'send_btn': 'إرسال التعليق',
        'token_missing': '⚠️ خدمة إرسال التعليقات غير متاحة حالياً.',
        'feedback_success': '✅ تم الإرسال بنجاح! تابع الـ Issue على:',
        'feedback_error': '❌ فشل الإرسال:',
        'write_comment': '⚠️ الرجاء كتابة تعليق قبل الإرسال.',
        'footer': 'تم التطوير بواسطة Zakarya Benregreg - خوارزمية MARIA محمية ببراءة اختراع.',
    },
    'English': {
        'app_title': '🧠 MARIA: Advanced Optimization Algorithm',
        'app_desc': 'This app implements the MARIA algorithm, combining:',
        'feature1': 'Hierarchical radial scan with biased directions',
        'feature2': 'Hierarchical direction lifting',
        'feature3': 'Dynamic relaxation shift',
        'feature4': 'Advanced local search (1-1, 2-1, 1-2, 2-2 swaps)',
        'feature5': 'Parallel computing for speed',
        'feature6': 'Multiple customizable stopping criteria',
        'tab_upload': '📂 Upload File',
        'tab_manual': '✍️ Manual Input',
        'upload_header': 'Upload Problem File',
        'choose_file': 'Choose a file',
        'upload_success': 'File uploaded successfully!',
        'upload_error': 'Error reading file:',
        'manual_header': 'Manual Data Entry',
        'manual_warning': 'For small problems only (n ≤ 10, m ≤ 10)',
        'c_coeff': 'Objective coefficients c[i]',
        'A_matrix': 'Constraint matrix A[i][j]',
        'b_rhs': 'Right-hand side b[i]',
        'solve_btn': '🚀 Solve Entered Problem',
        'error_negative': '⚠️ Values must be non-negative.',
        'solution_success': '✅ Problem solved successfully! (demo)',
        'feedback_section': '💬 Contact Us - Send your feedback',
        'comment_placeholder': 'Write your comment here...',
        'send_btn': 'Send Feedback',
        'token_missing': '⚠️ Feedback service is currently unavailable.',
        'feedback_success': '✅ Sent successfully! Track the issue at:',
        'feedback_error': '❌ Sending failed:',
        'write_comment': '⚠️ Please write a comment before sending.',
        'footer': 'Developed by Zakarya Benregreg - MARIA algorithm patented.',
    },
    'Français': {
        'app_title': '🧠 MARIA: Algorithme avancé d\'optimisation',
        'app_desc': 'Cette application implémente l\'algorithme MARIA, combinant :',
        'feature1': 'Balayage radial hiérarchique avec directions orientées',
        'feature2': 'Relèvement hiérarchique des directions',
        'feature3': 'Décalage dynamique de relaxation',
        'feature4': 'Recherche locale avancée (échanges 1-1, 2-1, 1-2, 2-2)',
        'feature5': 'Calcul parallèle pour la rapidité',
        'feature6': 'Critères d\'arrêt multiples personnalisables',
        'tab_upload': '📂 Télécharger un fichier',
        'tab_manual': '✍️ Saisie manuelle',
        'upload_header': 'Télécharger le fichier problème',
        'choose_file': 'Choisir un fichier',
        'upload_success': 'Fichier téléchargé avec succès !',
        'upload_error': 'Erreur de lecture du fichier :',
        'manual_header': 'Saisie manuelle des données',
        'manual_warning': 'Pour petits problèmes seulement (n ≤ 10, m ≤ 10)',
        'c_coeff': 'Coefficients objectifs c[i]',
        'A_matrix': 'Matrice des contraintes A[i][j]',
        'b_rhs': 'Second membre b[i]',
        'solve_btn': '🚀 Résoudre le problème saisi',
        'error_negative': '⚠️ Les valeurs doivent être non négatives.',
        'solution_success': '✅ Problème résolu avec succès ! (démonstration)',
        'feedback_section': '💬 Contactez-nous - Envoyez votre commentaire',
        'comment_placeholder': 'Écrivez votre commentaire ici...',
        'send_btn': 'Envoyer',
        'token_missing': '⚠️ Le service de commentaires est actuellement indisponible.',
        'feedback_success': '✅ Envoyé avec succès ! Suivez l\'issue sur :',
        'feedback_error': '❌ Échec de l\'envoi :',
        'write_comment': '⚠️ Veuillez écrire un commentaire avant d\'envoyer.',
        'footer': 'Développé par Zakarya Benregreg - Algorithme MARIA breveté.',
    },
    'Русский': {
        'app_title': '🧠 MARIA: Продвинутый алгоритм оптимизации',
        'app_desc': 'Это приложение реализует алгоритм MARIA, объединяющий:',
        'feature1': 'Иерархическое радиальное сканирование с направленными направлениями',
        'feature2': 'Иерархический подъём направлений',
        'feature3': 'Динамический сдвиг релаксации',
        'feature4': 'Продвинутый локальный поиск (обмены 1-1, 2-1, 1-2, 2-2)',
        'feature5': 'Параллельные вычисления для скорости',
        'feature6': 'Несколько настраиваемых критериев остановки',
        'tab_upload': '📂 Загрузить файл',
        'tab_manual': '✍️ Ручной ввод',
        'upload_header': 'Загрузить файл задачи',
        'choose_file': 'Выберите файл',
        'upload_success': 'Файл успешно загружен!',
        'upload_error': 'Ошибка чтения файла:',
        'manual_header': 'Ручной ввод данных',
        'manual_warning': 'Только для небольших задач (n ≤ 10, m ≤ 10)',
        'c_coeff': 'Коэффициенты целевой функции c[i]',
        'A_matrix': 'Матрица ограничений A[i][j]',
        'b_rhs': 'Правая часть b[i]',
        'solve_btn': '🚀 Решить введённую задачу',
        'error_negative': '⚠️ Значения должны быть неотрицательными.',
        'solution_success': '✅ Задача успешно решена! (демонстрация)',
        'feedback_section': '💬 Свяжитесь с нами - Отправьте отзыв',
        'comment_placeholder': 'Напишите ваш комментарий здесь...',
        'send_btn': 'Отправить отзыв',
        'token_missing': '⚠️ Сервис отзывов в настоящее время недоступен.',
        'feedback_success': '✅ Отправлено успешно! Следите за задачей по ссылке:',
        'feedback_error': '❌ Ошибка отправки:',
        'write_comment': '⚠️ Пожалуйста, напишите комментарий перед отправкой.',
        'footer': 'Разработано Закарией Бенрегрег - Алгоритм MARIA запатентован.',
    }
}

def t(key):
    return translations[st.session_state.language][key]

# ----------------------------------------------------------------------
# منطق الخوارزمية (MARIA) - لم يتغير
# ----------------------------------------------------------------------
def local_search_advanced(variables):
    n = len(variables)
    swaps = []
    for i in range(n):
        for j in range(i+1, n):
            swaps.append((i, j))
            for k in range(j+1, n):
                swaps.append((i, j, k))
                swaps.append((i, k, j))
                swaps.append((j, k, i))
    return swaps

def apply_swap(variables, swap):
    new_vars = variables.copy()
    for idx in swap:
        new_vars[idx] = 1 - new_vars[idx]
    return new_vars

def calculate_cost(variables):
    return np.sum(variables)

def algorithm_with_advanced_local_search(variables, max_iterations=100, 
                                         progress_callback=None, 
                                         checkpoint_callback=None):
    best_solution = variables.copy()
    best_cost = calculate_cost(variables)
    for iteration in range(max_iterations):
        if progress_callback:
            should_stop = progress_callback(iteration, max_iterations)
            if should_stop:
                break
        swaps = local_search_advanced(best_solution)
        for swap in swaps:
            new_solution = apply_swap(best_solution, swap)
            new_cost = calculate_cost(new_solution)
            if new_cost < best_cost:
                best_solution = new_solution.copy()
                best_cost = new_cost
        if checkpoint_callback:
            checkpoint_callback(iteration, best_solution, best_cost)
    return best_solution, best_cost

# ----------------------------------------------------------------------
# دوال GitHub (لإرسال التعليقات)
# ----------------------------------------------------------------------
def send_to_github_issue(comment, repo_owner, repo_name, token):
    url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/issues"
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json"
    }
    title = f"Comment from user at {datetime.now().strftime('%Y-%m-%d %H:%M')}"
    data = {"title": title, "body": comment}
    try:
        response = requests.post(url, headers=headers, data=json.dumps(data))
        if response.status_code == 201:
            return True, response.json().get("html_url")
        else:
            return False, f"Error {response.status_code}: {response.text}"
    except Exception as e:
        return False, str(e)

# ----------------------------------------------------------------------
# واجهة Streamlit
# ----------------------------------------------------------------------
st.set_page_config(page_title="MARIA Solver", layout="wide")

# تهيئة اللغة (الإنجليزية افتراضياً)
if 'language' not in st.session_state:
    st.session_state.language = 'English'

# CSS لتكبير الخط وجعله أسود
st.markdown("""
<style>
    html, body, [class*="css"]  {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        font-weight: 600;
        color: #000000;
    }
    .stApp {
        background-color: #f5f5f5;
    }
    h1, h2, h3, h4, h5, h6 {
        font-weight: 700;
        color: #000000;
    }
    p, div, span, label, .stMarkdown {
        font-weight: 500;
        color: #000000;
    }
</style>
""", unsafe_allow_html=True)

# خلفية ثابتة من ملف background.png
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
    }}
    </style>
    """, unsafe_allow_html=True)
else:
    st.sidebar.warning(f"⚠️ لم يتم العثور على ملف الخلفية '{BACKGROUND_IMAGE}'. تأكد من وجوده في المجلد الرئيسي.")

# معلومات الاتصال (المربع الأحمر)
CONTACT_EMAIL = "zakibeny@gmail.com"
CONTACT_PHONE = "0021355614305 / 00213779679073"
CONTACT_FAX = "036495241"

# رأس الصفحة مع اختيار اللغة
col1, col2 = st.columns([4, 1])
with col1:
    st.title(t('app_title'))
with col2:
    lang = st.selectbox("", ['English', 'Français', 'العربية', 'Русский'], key='lang_selector')
    st.session_state.language = lang

# المربع الأحمر
st.markdown(f"""
<div style="background-color: #ffeeee; padding: 20px; border-radius: 15px; text-align: center; margin: 20px 0; border: 3px solid red;">
    <span style="color: red; font-size: 32px; font-weight: bold;">✉️ {CONTACT_EMAIL}</span><br>
    <span style="color: red; font-size: 32px; font-weight: bold;">📞 {CONTACT_PHONE}</span><br>
    <span style="color: red; font-size: 32px; font-weight: bold;">📠 {CONTACT_FAX}</span>
</div>
""", unsafe_allow_html=True)

# وصف الخوارزمية
st.markdown(t('app_desc'))
st.markdown(f"- {t('feature1')}\n- {t('feature2')}\n- {t('feature3')}\n- {t('feature4')}\n- {t('feature5')}\n- {t('feature6')}")

# التبويبات (رفع ملف و إدخال يدوي فقط)
tab1, tab2 = st.tabs([t('tab_upload'), t('tab_manual')])

# ------------------- التبويب 1: رفع ملف -------------------
with tab1:
    st.header(t('upload_header'))
    uploaded_file = st.file_uploader(t('choose_file'), type=["txt", "csv", "json", "lp", "mps"])
    if uploaded_file is not None:
        try:
            content = uploaded_file.read().decode("utf-8")
            st.success(f"{t('upload_success')} ({len(content)} chars)")
        except Exception as e:
            st.error(f"{t('upload_error')} {e}")

# ------------------- التبويب 2: إدخال يدوي -------------------
with tab2:
    st.header(t('manual_header'))
    st.warning(t('manual_warning'))
    n_man = st.number_input(t('n_vars'), min_value=1, max_value=10, value=3)
    m_man = st.number_input(t('n_constraints'), min_value=1, max_value=10, value=3)

    st.subheader(t('c_coeff'))
    c_vals = []
    cols_c = st.columns(min(5, n_man))
    for i in range(n_man):
        with cols_c[i % 5]:
            val = st.number_input(f"c[{i}]", value=0.0, key=f"c_man_{i}")
            c_vals.append(val)

    st.subheader(t('A_matrix'))
    A_vals = np.zeros((m_man, n_man))
    for i in range(m_man):
        st.write(f"**Constraint {i+1}:**")
        cols_a = st.columns(min(5, n_man))
        for j in range(n_man):
            with cols_a[j % 5]:
                val = st.number_input(f"A[{i}][{j}]", value=0.0, key=f"A_man_{i}_{j}")
                A_vals[i, j] = val

    st.subheader(t('b_rhs'))
    b_vals = []
    cols_b = st.columns(min(5, m_man))
    for i in range(m_man):
        with cols_b[i % 5]:
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

    if st.button(t('solve_btn')):
        if error_msg:
            st.error(t('error_negative'))
        else:
            st.success(t('solution_success'))

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

# صندوق التعليقات فقط
with st.form("feedback_form"):
    user_comment = st.text_area("", height=150, placeholder=t('comment_placeholder'))
    submitted = st.form_submit_button(t('send_btn'))
    if submitted and user_comment.strip():
        if not token_available:
            st.warning(t('token_missing'))
        else:
            with st.spinner("Sending..."):
                success, result = send_to_github_issue(user_comment, REPO_OWNER, REPO_NAME, GITHUB_TOKEN)
                if success:
                    st.success(f"{t('feedback_success')} [{result}]({result})")
                else:
                    st.error(f"{t('feedback_error')} {result}")
    elif submitted:
        st.warning(t('write_comment'))

st.markdown("---")
st.caption(t('footer'))
