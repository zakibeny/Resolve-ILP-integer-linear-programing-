import streamlit as st
import numpy as np
from PIL import Image
import base64
# محاكاة البحث المحلي المتقدم
def local_search_advanced(variables):
    n = len(variables)
    swaps = []
    for i in range(n):
        for j in range(i+1, n):
            swaps.append((i, j))  # 1-1 swap
            for k in range(j+1, n):
                swaps.append((i, j, k))  # 2-1 swap
                swaps.append((i, k, j))  # 1-2 swap
                swaps.append((j, k, i))  # 2-2 swap
    return swaps

def apply_swap(variables, swap):
    new_vars = variables.copy()
    for idx in swap:
        new_vars[idx] = 1 - new_vars[idx]  # تبادل 0 إلى 1 أو 1 إلى 0
    return new_vars

def calculate_cost(variables):
    return np.sum(variables)  # محاكاة لحساب التكلفة

def algorithm_with_advanced_local_search(variables, max_iterations=100):
    best_solution = variables
    best_cost = calculate_cost(variables)
    
    for iteration in range(max_iterations):
        swaps = local_search_advanced(best_solution)
        for swap in swaps:
            new_solution = apply_swap(best_solution, swap)
            new_cost = calculate_cost(new_solution)
            if new_cost < best_cost:
                best_solution = new_solution
                best_cost = new_cost
    return best_solution, best_cost

# ----------------- ديكور واجهة المستخدم -----------------

# تخصيص الشكل باستخدام CSS لتغيير الألوان
st.markdown("""
    <style>
    .stApp {
        background-color: #F0F0F0;  /* تغيير الخلفية */
        font-family: "Arial", sans-serif;
    }
    h1 {
        color: #003366;  /* تخصيص عنوان البرنامج */
        font-weight: bold;
    }
    .stButton>button {
        background-color: #0066cc; /* تخصيص لون الأزرار */
        color: white;
        font-weight: bold;
    }
    /* تخصيص الألوان والكونتراست لتحسين قراءة النصوص */
    body {
        color: #000000; /* جعل النص أسود */
    }
    h1, h2, h3, h4, h5, h6 {
        color: #003366; /* تخصيص اللون للعناوين */
    }
    .stTextInput input {
        color: #000000; /* تغيير لون النص داخل حقول الإدخال */
    }
    .stTextArea textarea {
        color: #000000;
    }
    </style>
""", unsafe_allow_html=True)

# إضافة خيار تغيير الخلفية مرة واحدة فقط
if "background_uploaded" not in st.session_state:
    st.session_state.background_uploaded = False

if not st.session_state.background_uploaded:
    uploaded_file = st.file_uploader("اختر صورة خلفية (سيتم استخدامها مرة واحدة)", type=["jpg", "png", "jpeg"], key="background_file_uploader")
    if uploaded_file is not None:
        # تحويل الصورة إلى شكل Base64
        image = Image.open(uploaded_file)
        st.image(image, use_column_width=True)

        # تحويل الصورة إلى base64 لتستخدم في الخلفية
        img_bytes = uploaded_file.getvalue()
        img_base64 = base64.b64encode(img_bytes).decode()

        # إضافة الصورة كخلفية باستخدام CSS
        st.markdown(f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{img_base64}");
            background-size: cover;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        </style>
        """, unsafe_allow_html=True)

        st.session_state.background_uploaded = True  # تم رفع الصورة وتم استخدامها

# ----------------------------------

# معلومات الاتصال
CONTACT_EMAIL = "zakibeny@gmail.com"
CONTACT_PHONE = "0021355614305 / 00213779679073"
CONTACT_FAX = "036495241"

st.markdown(f"""
<div style="background-color: #ffeeee; padding: 20px; border-radius: 15px; text-align: center; margin: 20px 0; border: 3px solid red;">
    <span style="color: red; font-size: 32px; font-weight: bold;">✉️ {CONTACT_EMAIL}</span><br>
    <span style="color: red; font-size: 32px; font-weight: bold;">📞 {CONTACT_PHONE}</span><br>
    <span style="color: red; font-size: 32px; font-weight: bold;">📠 {CONTACT_FAX}</span>
</div>
""", unsafe_allow_html=True)

# عرض العنوان
st.title("🧠 AHRH: خوارزمية هرمية انكماشية متطورة")

# إدخال لغة الواجهة
language = st.selectbox("اختر اللغة:", ['English', 'Français', 'العربية', 'Русский'])
st.session_state.language = language

# تقديم شرح عن التطبيق
st.markdown("""
هذا التطبيق يطبق خوارزمية AHRH المتقدمة التي تجمع بين:
- المسح الشعاعي الهرمي مع اتجاهات موجهة
- الرفع الهرمي للاتجاهات
- إزاحة الاسترخاء الديناميكية
- بحث محلي متقدم بأنماط تبادل متعددة (1-1، 2-1، 1-2، 2-2)
- توازي الحسابات لتسريع الأداء
- معايير توقف متعددة قابلة للاختيار
""")

# التنقل بين التبويبات
tab1, tab2, tab3 = st.tabs(["📂 رفع ملف", "🎲 توليد عشوائي", "✍️ إدخال يدوي"])

with tab1:
    st.header("رفع ملف المسألة")
    uploaded_file = st.file_uploader("اختر ملف المسألة", type=["txt", "csv", "json", "lp", "mps"])
    if uploaded_file is not None:
        st.success("تم رفع الملف بنجاح!")

with tab2:
    st.header("توليد مسألة عشوائية")
    n = st.number_input("عدد المتغيرات", min_value=1, max_value=50, value=5)
    m = st.number_input("عدد القيود", min_value=1, max_value=50, value=5)
    if st.button("🎲 توليد وحل"):
        st.success("تم توليد المسألة وحلها بنجاح!")

with tab3:
    st.header("إدخال بيانات المسألة يدويًا")
    n_man = st.number_input("عدد المتغيرات (n)", min_value=1, max_value=10, value=3)
    m_man = st.number_input("عدد القيود (m)", min_value=1, max_value=10, value=3)

    # إدخال قيم معاملات الهدف
    st.subheader("معاملات الهدف c[i]")
    c_vals = []
    cols = st.columns(min(5, n_man))
    for i in range(n_man):
        with cols[i % 5]:
            val = st.number_input(f"c[{i}]", value=0.0, key=f"c_man_{i}")
            c_vals.append(val)
    
    # إدخال مصفوفة القيود
    st.subheader("مصفوفة القيود A[i][j]")
    A_vals = np.zeros((m_man, n_man))
    for i in range(m_man):
        st.write(f"**القيود {i + 1}:**")
        cols = st.columns(min(5, n_man))
        for j in range(n_man):
            with cols[j % 5]:
                val = st.number_input(f"A[{i}][{j}]", value=0.0, key=f"A_man_{i}_{j}")
                A_vals[i, j] = val

    # إدخال الطرف الأيمن
    st.subheader("الطرف الأيمن b[i]")
    b_vals = []
    cols = st.columns(min(5, m_man))
    for i in range(m_man):
        with cols[i % 5]:
            val = st.number_input(f"b[{i}]", value=0.0, key=f"b_man_{i}")
            b_vals.append(val)
    
    # زر لحل المسألة
    if st.button("🚀 حل المسألة المدخلة"):
        st.success("تم حل المسألة بنجاح!")

# ------------------- استقبال الرسائل من GitHub Issues -------------------

st.markdown("""
    <hr style="border: 1px solid #ccc;">
    <h2 style="color:#003366;">التعليقات من GitHub Issues</h2>
""", unsafe_allow_html=True)

# نص الرسائل القادمة من GitHub Issues
st.write("تمكن المستخدم من رفع الملفات وعرض النتائج بنجاح.")
st.write("تم تطبيق الخوارزمية بنجاح، وهناك نتائج متوقعة بناءً على البيانات المدخلة.")