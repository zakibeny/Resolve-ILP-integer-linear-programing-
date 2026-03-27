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

# إضافة الديكور باستخدام CSS لعرض صورة الخلفية
uploaded_file = st.file_uploader("اختر صورة خلفية", type=["jpg", "png", "jpeg"], key="background_file_uploader")

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
    }
    .stButton>button {
        background-color: #0066cc; /* تخصيص لون الأزرار */
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

# إعداد واجهة المستخدم
st.title("برنامج البرمجة الخطية مع البحث المحلي المتقدم")

# خيار إدخال يدوي
input_method = st.radio("اختر طريقة الإدخال:", ["إدخال يدوي", "رفع ملف"], key="input_method")

if input_method == "إدخال يدوي":
    size = st.number_input("عدد المتغيرات", min_value=1, max_value=50, value=5)
    variables = np.random.randint(0, 2, size=size)
    st.write(f"المتغيرات المدخلة: {variables}")
    
    # تنفيذ البحث المحلي المتقدم
    best_solution, best_cost = algorithm_with_advanced_local_search(variables)
    st.write(f"أفضل حل: {best_solution}")
    st.write(f"أفضل تكلفة: {best_cost}")

elif input_method == "رفع ملف":
    uploaded_file = st.file_uploader("اختر الملف", type=["txt", "csv", "json", "lp", "mps"], key="file_uploader")
    if uploaded_file is not None:
        # معالج الملفات
        st.write(f"تم رفع الملف: {uploaded_file.name}")
        # يمكن تعديل الكود هنا لاستيراد البيانات من الملفات
        st.write("ملف المعالجة غير مكتمل بعد!")
