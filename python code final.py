import numpy as np
import pandas as pd
import pyttsx3
import tkinter as tk
from tkinter import filedialog, messagebox
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
import threading
from time import time

# Initialize TTS
engine = pyttsx3.init()
engine.setProperty('rate', 150)

# Global variables
df = None
model = None
scaler = None
training_in_progress = False

def upload_file():
    file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
    if not file_path:
        messagebox.showerror("Error", "No file selected!")
        return
    try:
        global df
        df = pd.read_csv(file_path)
        required_columns = ["GPS speed", "OBD speed", "Engine RPM", "Throttle position", 
                          "Engine load", "Coolant Temperature", "Fuel Consumption", 
                          "CO2 Emission", "Engine Temperature", "Brake Oil", "Maintenance Score"]
        
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            messagebox.showerror("Error", f"Missing required columns: {', '.join(missing_cols)}")
            df = None
            return
            
        messagebox.showinfo("Success", f"File Uploaded Successfully!\n{len(df)} records loaded")
        result_label.config(text="Data uploaded successfully. Ready to train model.")
    except Exception as e:
        messagebox.showerror("Error", f"Failed to read file: {e}")
        df = None

def train_model_thread():
    global model, scaler, training_in_progress
    try:
        if df is None:
            app.after(0, lambda: messagebox.showerror("Error", "Please upload a CSV file first!"))
            return
            
        start_time = time()
        
        # Use all important features but with optimized parameters
        X = df[["GPS speed", "OBD speed", "Engine RPM", "Throttle position", 
               "Engine load", "Coolant Temperature"]]
        y = df[["Fuel Consumption", "CO2 Emission", "Engine Temperature", 
               "Brake Oil", "Maintenance Score"]]
        
        # Balanced train-test split for speed and accuracy
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

        app.after(0, lambda: result_label.config(text="Scaling data..."))
        
        # Efficient scaling
        scaler = MinMaxScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        app.after(0, lambda: result_label.config(text="Training model (10-20 seconds)..."))
        
        # Optimized Gradient Boosting model
        model = MultiOutputRegressor(
            GradientBoostingRegressor(
                n_estimators=40,           # Optimal balance between speed and accuracy
                max_depth=4,               # Slightly deeper for better accuracy
                learning_rate=0.1,        # Faster convergence
                min_samples_split=5,       # Prevent overfitting
                random_state=42,
                validation_fraction=0.1,
                n_iter_no_change=5
            ),
            n_jobs=-1  # Parallel processing
        )
        
        model.fit(X_train, y_train)
        
        training_time = round(time() - start_time, 2)
        app.after(0, lambda: messagebox.showinfo("Success", f"Model Trained in {training_time} seconds!"))
        app.after(0, lambda: result_label.config(text=f"Model trained in {training_time}s. Ready for evaluation."))
        
    except Exception as e:
        app.after(0, lambda: messagebox.showerror("Error", f"Failed to train model: {e}"))
    finally:
        training_in_progress = False

def train_model():
    global training_in_progress
    if df is None:
        messagebox.showerror("Error", "Please upload a CSV file first!")
        return
    
    if training_in_progress:
        messagebox.showwarning("Warning", "Model training already in progress!")
        return
        
    training_in_progress = True
    result_label.config(text="Training started... Please wait (10-20 seconds expected)")
    threading.Thread(target=train_model_thread, daemon=True).start()

def evaluate():
    if model is None:
        messagebox.showerror("Error", "Please train the model first!")
        return
        
    try:
        # Get all entries with proper validation
        entries = [
            ("GPS Speed", gps_speed_entry),
            ("OBD Speed", obd_speed_entry),
            ("Engine RPM", rpm_entry),
            ("Throttle Position", throttle_entry),
            ("Engine Load", load_entry),
            ("Coolant Temperature", coolant_temp_entry)
        ]
        
        user_input = []
        for name, entry in entries:
            value = entry.get().strip()
            if not value:
                messagebox.showerror("Error", f"Please enter {name}!")
                return
            try:
                user_input.append(float(value))
            except ValueError:
                messagebox.showerror("Error", f"Invalid number in {name}!")
                return

        user_input_scaled = scaler.transform([user_input])
        y_pred = model.predict(user_input_scaled)[0]

        # Unpack all outputs
        fuel_consumption, co2_emission, engine_temp, brake_wear, maintenance_score = map(lambda x: round(x, 2), y_pred)

        # Comprehensive evaluation criteria
        if (maintenance_score > 80 and 
            engine_temp < 90 and 
            brake_wear < 30 and 
            fuel_consumption < 10):
            engine_condition = "EXCELLENT ✅"
            solution = "Your engine is in perfect condition. Maintain current driving habits."
        elif (maintenance_score > 65 and 
              engine_temp < 95 and 
              brake_wear < 50 and 
              fuel_consumption < 15):
            engine_condition = "GOOD 🌤"
            solution = "Engine is performing well. Schedule routine maintenance soon."
        elif (maintenance_score > 50 or 
              engine_temp < 105 or 
              brake_wear < 70):
            engine_condition = "MODERATE 🚨"
            solution = "Attention needed. Schedule maintenance and check engine parameters."
        else:
            engine_condition = "CRITICAL ❌"
            solution = "Immediate maintenance required! Engine at risk of damage."

        report = f"""Engine Health Report:
Condition: {engine_condition}
Fuel Consumption: {fuel_consumption} L/100km
CO2 Emission: {co2_emission} g/km
Engine Temperature: {engine_temp}°C
Brake Wear: {brake_wear}%
Maintenance Score: {maintenance_score}/100

Recommendation: {solution}"""
        
        result_label.config(text=report)
        engine.say(f"Engine condition is {engine_condition}. {solution}")
        engine.runAndWait()
    except Exception as e:
        messagebox.showerror("Error", f"Evaluation failed: {str(e)}")

# GUI Setup
app = tk.Tk()
app.title("Vehicle Health Monitoring System")
app.geometry("1000x750")
app.config(bg="#e3f2fd")

header = tk.Label(app, text="🚗 COMPLETE Vehicle Health Monitor", font=("Arial", 28, "bold"), bg="#42a5f5", fg="white", padx=20, pady=20)
header.pack(fill=tk.X)

# Buttons
button_frame = tk.Frame(app, bg="#e3f2fd")
tk.Button(button_frame, text="📤 Upload CSV", command=upload_file, bg="#29b6f6", fg="white", font=("Arial", 14)).pack(side=tk.LEFT, padx=20)
tk.Button(button_frame, text="⚡ Train Model", command=train_model, bg="#66bb6a", fg="white", font=("Arial", 14)).pack(side=tk.LEFT, padx=20)
button_frame.pack(pady=20)

# Input Fields (all parameters included)
input_frame = tk.Frame(app, bg="#e3f2fd")
fields = [
    ("GPS Speed (km/h)", "gps_speed_entry"),
    ("OBD Speed (km/h)", "obd_speed_entry"),
    ("Engine RPM", "rpm_entry"),
    ("Throttle Position (%)", "throttle_entry"),
    ("Engine Load (%)", "load_entry"),
    ("Coolant Temp (°C)", "coolant_temp_entry")
]

for label_text, var_name in fields:
    frame = tk.Frame(input_frame, bg="#e3f2fd")
    tk.Label(frame, text=label_text, font=("Arial", 14), bg="#e3f2fd").pack(side=tk.LEFT)
    entry = tk.Entry(frame, font=("Arial", 14), width=15)
    entry.pack(side=tk.LEFT, padx=10)
    frame.pack(pady=8)
    globals()[var_name] = entry

input_frame.pack()

# Evaluate Button
tk.Button(app, text="Evaluate Health", command=evaluate, bg="#ffa726", fg="white", font=("Arial", 16, "bold"), width=20).pack(pady=30)

# Result Display
result_label = tk.Label(app, text="Upload CSV data and train model (10-20 sec training)", 
                       font=("Arial", 14), bg="#e3f2fd", justify="left", wraplength=800)
result_label.pack(pady=20)

app.mainloop()