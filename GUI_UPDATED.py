# -*- coding: utf-8 -*-
"""
Created on Sat Oct 12 11:03:09 2024

@author: Home Photoworks Dago
"""

import tkinter as tk
from tkinter import ttk
from tkinter import messagebox, ttk
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import pickle
import os
# Fungsi untuk melatih dan menyimpan model (sama seperti sebelumnya)
def train_and_save_models():
    df = pd.read_csv('Medicaldataset.csv')
    
    X = df[['Age','Gender', 'Heart rate','Systolic blood pressure','Diastolic blood pressure','Blood sugar', 'CK-MB', 'Troponin']]
    X_gender_ckmb = df[['Gender', 'CK-MB']]
    y = df['Result'].apply(lambda x: 1 if x == 'positive' else 0)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    X_train_gc, X_test_gc, y_train_gc, y_test_gc = train_test_split(X_gender_ckmb, y, test_size=0.3, random_state=42)

    dt_model = DecisionTreeClassifier(criterion='entropy', max_depth=5)
    dt_model.fit(X_train, y_train)
    
    # Hitung dan tampilkan metriks evaluator model klasifikasi
    Y_pred = dt_model.predict(X_test)
    df_classes = df[['Result']].Result.unique()
    print(df_classes)
    
    from sklearn.metrics import classification_report
    print(classification_report(y_test, Y_pred, target_names = df_classes))
    
    
    with open('heart_attack_dt_model.pkl', 'wb') as f:
        pickle.dump(dt_model, f)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    knn_model = KNeighborsClassifier(n_neighbors=6)
    knn_model.fit(X_train_scaled, y_train)
    with open('heart_attack_knn_model.pkl', 'wb') as f:
        pickle.dump(knn_model, f)
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)

    dt_gc_model = DecisionTreeClassifier(criterion='entropy', max_depth=10)
    dt_gc_model.fit(X_train_gc, y_train_gc)
    
    # Hitung dan tampilkan metriks evaluator model klasifikasi
    Y_pred_gc = dt_gc_model.predict(X_test)
    df_classes = df[['Result']].Result.unique()
    print(df_classes)
    
    from sklearn.metrics import classification_report
    print(classification_report(y_test, Y_pred_gc, target_names = df_classes))
    
    with open('heart_attack_dt_gc_model.pkl', 'wb') as f:
        pickle.dump(dt_gc_model, f)

    scaler_gc = StandardScaler()
    X_train_gc_scaled = scaler_gc.fit_transform(X_train_gc)
    knn_gc_model = KNeighborsClassifier(n_neighbors=5)
    knn_gc_model.fit(X_train_gc_scaled, y_train_gc)
    with open('heart_attack_knn_gc_model.pkl', 'wb') as f:
        pickle.dump(knn_gc_model, f)
    with open('scaler_gc.pkl', 'wb') as f:
        pickle.dump(scaler_gc, f)

if not os.path.exists('heart_attack_dt_model.pkl') or not os.path.exists('heart_attack_knn_model.pkl') or \
   not os.path.exists('heart_attack_dt_gc_model.pkl') or not os.path.exists('heart_attack_knn_gc_model.pkl'):
    train_and_save_models()

with open('heart_attack_dt_model.pkl', 'rb') as f:
    dt_model = pickle.load(f)

with open('heart_attack_knn_model.pkl', 'rb') as f:
    knn_model = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('heart_attack_dt_gc_model.pkl', 'rb') as f:
    dt_gc_model = pickle.load(f)

with open('heart_attack_knn_gc_model.pkl', 'rb') as f:
    knn_gc_model = pickle.load(f)

with open('scaler_gc.pkl', 'rb') as f:
    scaler_gc = pickle.load(f)

# Fungsi untuk prediksi berdasarkan metode dan page yang dipilih

def predict_heart_attack(page):
    try:
        if page == 'all_features':
            # Mengambil input dari form untuk semua fitur
            age = entry_age.get()
            heart_rate = entry_heart_rate.get()
            systolic = entry_sys.get()
            diastolic = entry_dias.get()
            blood_sugar = entry_bloodsugar.get()
            ckmb = entry_ckmb.get()
            trop = entry_troponin.get()
            
            # Cetak nilai input untuk debugging
            print(f"Input: age={age}, heart_rate={heart_rate}, systolic={systolic}, diastolic={diastolic}, blood_sugar={blood_sugar}, ckmb={ckmb}, trop={trop}")
            
            # Validasi input (pastikan tidak ada input yang kosong)
            if not age or not heart_rate or not systolic or not diastolic or not blood_sugar or not ckmb or not trop:
                raise ValueError("Semua kolom input harus diisi.")
            
            # Konversi input ke tipe data yang sesuai dan validasi apakah input dapat dikonversi
            try:
                age = int(age)
                heart_rate = int(heart_rate)
                systolic = int(systolic)
                diastolic = int(diastolic)
                blood_sugar = int(blood_sugar)
                ckmb = float(ckmb)
                trop = float(trop)
            except ValueError:
                raise ValueError("Pastikan semua input numerik diisi dengan angka yang valid.")
            
            # Konversi gender
            gender = 1 if gender_var.get() == 'Laki-laki' else 0

            # Fitur untuk prediksi
            features = [[age, gender, heart_rate, systolic, diastolic, blood_sugar, ckmb, trop]]
            
            # Prediksi menggunakan model yang dipilih
            if model_var_all.get() == 'Decision Tree':
                prediction = dt_model.predict(features)
            else:
                features_scaled = scaler.transform(features)
                prediction = knn_model.predict(features_scaled)

        elif page == 'gc_features':
            # Mengambil input dari form untuk gender dan CKMB saja
            ckmb = entry_ckmb1.get()
            trop = entry_troponin1.get()

            # Cetak nilai input untuk debugging
            print(f"Input: ckmb={ckmb}, trop={trop}")

            # Validasi input
            if not ckmb or not trop:
                raise ValueError("CKMB dan Troponin harus diisi.")
            
            try:
                ckmb = float(ckmb)
                trop = float(trop)
            except ValueError:
                raise ValueError("Pastikan CKMB dan Troponin diisi dengan angka yang valid.")
            
            features_tc = [[ckmb, trop]]
            if model_var_gc.get() == 'Decision Tree':
                prediction = dt_gc_model.predict(features_tc)
            else:
                features_tc_scaled = scaler_gc.transform(features_tc)
                prediction = knn_gc_model.predict(features_tc_scaled)

        # Tampilkan hasil prediksi
        if prediction[0] == 1:
            result_label.config(text="Hasil Prediksi: Positive (Berpotensi Serangan Jantung)")
        else:
            result_label.config(text="Hasil Prediksi: Negative (Tidak Berpotensi Serangan Jantung)")

    except ValueError as e:
        messagebox.showerror("Input Error", f"Error: {e}")

import tkinter as tk
from tkinter import ttk

# Membuat GUI menggunakan Tkinter
root = tk.Tk()
root.title("Prediksi Serangan Jantung")
root.geometry("800x800")  # Ukuran jendela yang lebih besar
root.configure(bg='#ecf0f1')  # Latar belakang jendela

# Membuat Notebook
notebook = ttk.Notebook(root)
notebook.pack(pady=10, expand=True, fill='both')

# Frame utama semua fitur
all_features = ttk.Frame(notebook, padding=20)
all_features.pack(fill='both', expand=True)

# Frame 2 fitur
tc_features = ttk.Frame(notebook, padding=20)
tc_features.pack(fill='both', expand=True)

# Frame info
info_frame = ttk.Frame(notebook, padding=20)
info_frame.pack(fill='both', expand=True)

# Menambahkan frame ke dalam notebook
notebook.add(info_frame, text='Info Dataset')
notebook.add(all_features, text='Prediksi')
notebook.add(tc_features, text='Prediksi 2 Fitur')

######## MAIN FRAME ######################################################################
# Mengatur Style
style = ttk.Style()
style.configure('Header.TLabel', font=('Montserrat', 20, 'bold'), background='#3498db', foreground='white')
style.configure('TLabel', font=('Helvetica', 12), background='#ecf0f1')
style.configure('TButton', font=('Helvetica', 12), background='#2980b9', foreground='white', padding=10)
style.configure('TRadiobutton', font=('Helvetica', 12), background='#ecf0f1')
style.configure('TEntry', font=('Helvetica', 12))

# Header
header_label = ttk.Label(all_features, text="Prediksi Serangan Jantung", style='Header.TLabel', anchor='center')
header_label.pack(fill='x', pady=20)

# Frame untuk formulir input
form_frame = ttk.Frame(all_features)
form_frame.pack(pady=10)

# Menyusun label dan entry menggunakan grid
# Label dan entry untuk Usia
label_age = ttk.Label(form_frame, text="Usia:")
label_age.grid(row=0, column=0, sticky='e', pady=10, padx=10)
entry_age = ttk.Entry(form_frame)
entry_age.grid(row=0, column=1, pady=10, padx=10)

# Dropdown untuk Gender
gender_var = tk.StringVar()
label_gender = ttk.Label(form_frame, text="Gender:")
label_gender.grid(row=1, column=0, sticky='e', pady=10, padx=10)
gender_menu = ttk.Combobox(form_frame, textvariable=gender_var, state="readonly", font=('Helvetica', 12))
gender_menu['values'] = ("Perempuan", "Laki-laki")
gender_menu.current(0)  # Set default value
gender_menu.grid(row=1, column=1, pady=10, padx=10)

# Label dan entry untuk Heart Rate
label_heart_rate = ttk.Label(form_frame, text="Heart Rate (Denyut Jantung):")
label_heart_rate.grid(row=2, column=0, sticky='e', pady=10, padx=10)
entry_heart_rate = ttk.Entry(form_frame)
entry_heart_rate.grid(row=2, column=1, pady=10, padx=10)

# Label dan entry untuk Systolic Blood Pressure
label_sys = ttk.Label(form_frame, text="Systolic Blood Pressure:")
label_sys.grid(row=3, column=0, sticky='e', pady=10, padx=10)
entry_sys = ttk.Entry(form_frame)
entry_sys.grid(row=3, column=1, pady=10, padx=10)

# Label dan entry untuk Diastolic Blood Pressure
label_dias = ttk.Label(form_frame, text="Diastolic Blood Pressure:")
label_dias.grid(row=4, column=0, sticky='e', pady=10, padx=10)
entry_dias = ttk.Entry(form_frame)
entry_dias.grid(row=4, column=1, pady=10, padx=10)

# Label dan entry untuk Blood Sugar
label_sugar = ttk.Label(form_frame, text="Blood Sugar:")
label_sugar.grid(row=5, column=0, sticky='e', pady=10, padx=10)
entry_bloodsugar = ttk.Entry(form_frame)
entry_bloodsugar.grid(row=5, column=1, pady=10, padx=10)

# Label dan entry untuk CKMB
label_ckmb = ttk.Label(form_frame, text="CKMB (mg/dL):")
label_ckmb.grid(row=6, column=0, sticky='e', pady=10, padx=10)
entry_ckmb = ttk.Entry(form_frame)
entry_ckmb.grid(row=6, column=1, pady=10, padx=10)

# Label dan entry untuk Troponin
label_troponin = ttk.Label(form_frame, text="Troponin (ng/mL):")
label_troponin.grid(row=7, column=0, sticky='e', pady=10, padx=10)
entry_troponin = ttk.Entry(form_frame)
entry_troponin.grid(row=7, column=1, pady=10, padx=10)

# Pemilihan metode
model_var_all = tk.StringVar(value="Decision Tree")
label_model_all = ttk.Label(all_features, text="Pilih Metode:", font=('Helvetica', 14))
label_model_all.pack(pady=10)

radio_dt_all = ttk.Radiobutton(all_features, text="Decision Tree", variable=model_var_all, value="Decision Tree")
radio_dt_all.pack(pady=5)
radio_knn_all = ttk.Radiobutton(all_features, text="K-NN", variable=model_var_all, value="K-NN")
radio_knn_all.pack(pady=5)

# Tombol untuk prediksi
predict_button_all = ttk.Button(all_features, text="Prediksi", command=lambda: predict_heart_attack('all_features'))
predict_button_all.pack(pady=20)

##### FRAME 2 FITUR##########################################################################
# Header
header_label_gc = ttk.Label(tc_features, text="Prediksi Serangan Jantung", style='Header.TLabel', anchor='center')
header_label_gc.pack(fill='x', pady=20)

# Frame untuk formulir input
form_frame_gc = ttk.Frame(tc_features)
form_frame_gc.pack(pady=10)

# Label dan entry untuk CKMB
label_ckmb_gc = ttk.Label(form_frame_gc, text="CKMB (mg/dL):")
label_ckmb_gc.grid(row=0, column=0, sticky='e', pady=10, padx=10)
entry_ckmb1 = ttk.Entry(form_frame_gc)
entry_ckmb1.grid(row=0, column=1, pady=10, padx=10)

# Label dan entry untuk Troponin
label_troponin_gc = ttk.Label(form_frame_gc, text="Troponin (ng/mL):")
label_troponin_gc.grid(row=1, column=0, sticky='e', pady=10, padx=10)
entry_troponin1 = ttk.Entry(form_frame_gc)
entry_troponin1.grid(row=1, column=1, pady=10, padx=10)

# Pemilihan metode untuk frame 2 fitur
model_var_gc = tk.StringVar(value="Decision Tree")
label_model_gc = ttk.Label(tc_features, text="Pilih Metode:", font=('Helvetica', 14))
label_model_gc.pack(pady=10)

radio_dt_gc = ttk.Radiobutton(tc_features, text="Decision Tree", variable=model_var_gc, value="Decision Tree")
radio_dt_gc.pack(pady=5)
radio_knn_gc = ttk.Radiobutton(tc_features, text="K-NN", variable=model_var_gc, value="K-NN")
radio_knn_gc.pack(pady=5)

# Tombol untuk prediksi
predict_button_gc = ttk.Button(tc_features, text="Prediksi", command=lambda: predict_heart_attack('gc_features'))
predict_button_gc.pack(pady=20)

# Label untuk hasil prediksi
result_label = tk.Label(root, text="", font=("Helvetica", 14, "bold"), background='#ecf0f1', foreground='#34495e')
result_label.pack(pady=20)

# Tab 2: Halaman Informasi Dataset
info_label = tk.Label(info_frame, text=( 
    "Informasi Dataset:\n\n"
    "Dataset ini berisi data mengenai faktor risiko serangan jantung.\n"
    "- Gender: 0 untuk Perempuan, 1 untuk Laki-laki\n"
    "- Heart Rate: Denyut jantung dalam bpm (beats per minute)\n"
    "- CKMB: Konsentrasi Creatine Kinase-MB (mg/dL)\n"
    "- Troponin: Level Troponin dalam ng/mL\n"
    "- Result: Hasil diagnosis, bisa 'positive' atau 'negative'"
), justify="left", font=('Helvetica', 12), background='#ecf0f1')
info_label.pack(padx=10, pady=10)

# Tombol untuk menutup aplikasi
close_button = ttk.Button(root, text="Tutup", command=root.destroy)
close_button.pack(pady=10)

root.mainloop()

