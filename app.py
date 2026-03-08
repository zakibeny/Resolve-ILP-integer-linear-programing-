# -*- coding: utf-8 -*-
"""
تطبيق AHRH لحل ملفات ILP / UFLP متعددة دفعة واحدة
يشمل تحسينات الأداء وإصلاح مشاكل عدد المتغيرات
"""

import streamlit as st
import numpy as np
np.seterr(all='ignore')  # إيقاف فحص أخطاء NumPy الثقيلة

import pulp
solver = pulp.PULP_CBC_CMD(msg=False, threads=4)  # استخدام solver مثبت مع عدة أنوية

import matplotlib.pyplot as plt
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
warnings.filterwarnings("ignore")

st.title("تطبيق AHRH المتعدد الملفات ILP / UFLP")

# رفع ملفات متعددة
uploaded_files = st.file_uploader("اختر ملفات ILP / UFLP متعددة", type=["txt","csv"], accept_multiple_files=True)

if uploaded_files:
    results = []
    
    def process_file(file):
        try:
            df = pd.read_csv(file, delimiter=None, engine='python')
            data = df.values
            rows, cols = data.shape
            if np.any(pd.isna(data)):
                data = np.nan_to_num(data)  # معالجة الأعمدة أو الصفوف الناقصة
            
            # --- خوارزمية AHRH ---
            prob = pulp.LpProblem(f"AHRH_ILP_{file.name}", pulp.LpMinimize)
            x = [pulp.LpVariable(f"x_{i}", lowBound=0, cat='Integer') for i in range(cols)]
            prob += pulp.lpSum([x[i]*np.sum(data[:,i]) for i in range(cols)])
            for r in range(rows):
                prob += pulp.lpSum([x[c]*data[r,c] for c in range(cols)]) <= np.sum(data[r,:])
            
            prob.solve(solver)
            solution = [v.varValue for v in prob.variables()]
            status = pulp.LpStatus[prob.status]
            
            # --- الرسم ---
            plt.figure(figsize=(8,5))
            for i in range(cols):
                plt.plot(data[:,i], label=f'Column {i+1}')
            plt.title(f'Visualization: {file.name}')
            plt.xlabel('Row Index')
            plt.ylabel('Values')
            plt.legend()
            plt.grid(True)
            st.pyplot(plt)
            
            # حفظ النتائج
            processed_file = f"processed_{file.name}.csv"
            np.savetxt(processed_file, data, delimiter=",")
            
            return {"file": file.name, "status": status, "solution": solution, "saved_file": processed_file}
        
        except Exception as e:
            return {"file": file.name, "status": "Error", "solution": str(e), "saved_file": None}
    
    # --- التوازي لمعالجة جميع الملفات بسرعة ---
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_file, f) for f in uploaded_files]
        for future in as_completed(futures):
            results.append(future.result())
    
    # عرض ملخص النتائج
    st.subheader("ملخص النتائج")
    for res in results:
        st.write(f"**ملف:** {res['file']}")
        st.write(f"**الحالة:** {res['status']}")
        st.write(f"**الحل الأمثل:** {res['solution']}")
        if res['saved_file']:
            st.write(f"**تم حفظ البيانات في:** {res['saved_file']}")
        st.markdown("---")
