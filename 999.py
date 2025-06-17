import cv2
import json
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import face_recognition
import os
import numpy as np
import pytesseract
import time
from sklearn.svm import SVC
from datetime import datetime

pytesseract.pytesseract.tesseract_cmd = r'../Tesseract-OCR/tesseract.exe'

with open("drivers.json", "r") as f:
    drivers_data = json.load(f)

with open("wanted.json", "r") as f:
    wanted_list = json.load(f)

def log_action(action):
    entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "action": action
    }

    try:
        
        if os.path.exists("history.json"):
            with open("history.json", "r", encoding="utf-8") as f:
                history = json.load(f)
        else:
            history = []

        history.append(entry)

        with open("history.json", "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2, ensure_ascii=False)

    except Exception as e:
        print(" ")

path = 'CPTF-TEAM-A/faces' #Le lien de dossie FACES qui contient les photo pour comparais avec le camera detection
images = []
labels = []

for folder_name in os.listdir(path):
    person_folder = os.path.join(path, folder_name)
    if os.path.isdir(person_folder):  
        print(f"Processing folder: {folder_name}")  
        for filename in os.listdir(person_folder):
            if filename.endswith('.jpeg'): 
                img_path = os.path.join(person_folder, filename)
                img = cv2.imread(img_path)
                if img is None:
                    print(f"Failed to load image: {img_path}")
                    continue
                rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                encodings = face_recognition.face_encodings(rgb_img)
                if encodings:
                    print(f"Found encoding for {folder_name}: {filename}")  
                    images.append(encodings[0])
                    labels.append(folder_name) 
                else:
                    print(f"No face encoding found for {folder_name}: {filename}")
                    
if not images:
    print("No images or encodings found. Please check the image files.")
else:
    svm_model = SVC(kernel='linear', probability=True)
    svm_model.fit(images, labels)
    print("Model trained successfully.")

class SurveillanceApp:
    def __init__(self, root):
        self.root = root
        self.root.title("AI Surveillance ")
        self.root.geometry("1200x700")
        self.root.configure(bg='#1e1e1e')
        #self.root.iconbitmap('../cptf logo.png') #Ce command pour l'icon de l'interface

        self.cap = None
        self.running = False
        self.shown_names = set()

        time.sleep(0.1)

        self.setup_ui()

    def setup_ui(self):
        self.video_label = tk.Label(self.root, bg='black')
        self.video_label.place(x=10, y=10, width=850, height=640)

        info_frame = tk.Frame(self.root, bg='#2e2e2e')
        info_frame.place(x=880, y=10, width=300, height=400)
        tk.Label(info_frame, text="Driver Info", fg='white', bg='#2e2e2e', font=("Arial", 14)).pack(pady=10)
        self.info_text = tk.Text(info_frame, height=15, width=35, bg="#1e1e1e", fg="white")
        self.info_text.pack(pady=5)

        log_frame = tk.Frame(self.root, bg='#2e2e2e')
        log_frame.place(x=880, y=420, width=300, height=230)
        tk.Label(log_frame, text="Detection Log", fg='white', bg='#2e2e2e', font=("Arial", 14)).pack(pady=5)
        self.log_box = tk.Text(log_frame, height=12, width=35, bg="#1e1e1e", fg="white")
        self.log_box.pack()

        btn_frame = tk.Frame(self.root, bg='#1e1e1e')
        btn_frame.place(x=10, y=660, width=1170, height=40)
        self.start_btn = ttk.Button(btn_frame, text="Start Camera", command=self.start_camera)
        self.start_btn.pack(side=tk.LEFT, padx=10)
        self.stop_btn = ttk.Button(btn_frame, text="Stop Camera", command=self.stop_camera)
        self.stop_btn.pack(side=tk.LEFT, padx=10)

    def start_camera(self):
        self.cap = cv2.VideoCapture(0)  
        self.running = True
        self.shown_names = set()
        self.update_frame()

    def stop_camera(self):
        self.running = False
        if self.cap:
            self.cap.release()
            self.cap = None
        log_action(f"Camer closed")
        self.video_label.config(image='')

    def update_frame(self):
        if not self.running or not self.cap or not self.cap.isOpened():
            return

        success, img = self.cap.read()
        if not success:
            self.root.after(10, self.update_frame)
            return

        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        faces = face_recognition.face_locations(rgb_img)
        encodings = face_recognition.face_encodings(rgb_img, faces)

        plate_text = pytesseract.image_to_string(img, config='--psm 6')
        plate_text = ''.join(e for e in plate_text if e.isalnum())
        
        plate_text = ""
        found_owner = "UNKOWN"
        if time.time() - self.last_ocr_time > 2 :
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            plate_text = pytesseract.image_to_string(img, config='--psm 6')
            plate_text = ''.join(e for e in plate_text if e.isalnum())
            self.last_ocr_time = time.time()

            if plate_text:
                found_owner = None
                for driver_name, driver_info in drivers_data.items():
                    plate_from_data = driver_info.get("plate", "").replace(" ", "").upper()
                    if plate_from_data == plate_text.upper():
                        found_owner = driver_name
                        break

                if found_owner:
                    print(f"Detected Plate: {plate_text}")
                    print(f"Original Owner: {found_owner}")
                    log_action(f"plat is {plate_text} \n plat owner : {found_owner}")
                    self.log_box.insert(tk.END, f"[LOG] Plate Owner: {found_owner} | Plate: {plate_text}\n")
                    self.log_box.see(tk.END)
                    
                else:
                    print(f"Detected Plate: {plate_text}")


        for face_encoding, face_location in zip(encodings, faces):
            distances = face_recognition.face_distance(images, face_encoding)

            if len(distances) == 0:
                name = "UNKNOWN"
            else:
                min_distance = np.min(distances)
                if min_distance > 0.5:
                    name = "UNKNOWN"
                else:
                    best_match_index = np.argmin(distances)
                    name = labels[best_match_index]

            y1, x2, y2, x1 = face_location

            plate = drivers_data.get(name, {}).get("plate", "UNKNOWN")
            id_card = drivers_data.get(name, {}).get("ID", "UNKNOWN")
            permit = drivers_data.get(name, {}).get("PERMIT", "UNKNOWN")

            if name == "UNKNOWN":
                status = "UNKNOWN"
                color = (0, 255, 255)
            elif permit == "nul" and name not in wanted_list:
                wanted_list.append(name)
                with open("wanted.json", "w") as f:
                    json.dump(wanted_list, f)
                status = "WANTED"
                color = (0, 0, 255)



            else:
                status = "CLEAR" if name not in wanted_list else "WANTED"
                color = (0, 255, 0) if status == "CLEAR" else (0, 0, 255)

            log_action(f"Personne Name :{name} status : {status} id : {id_card}")
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.rectangle(img, (x1, y1 - 55), (x2, y1), color, cv2.FILLED)
            cv2.putText(img, f"{name} | {plate} | {status}", (x1+6, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
            #cv2.putText(img, status, (x1+6, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

            if name not in self.shown_names:
                self.shown_names.add(name)
                self.info_text.delete(1.0, tk.END)
                self.info_text.insert(tk.END, f"Name: {name}\nPlate: {plate}\nID: {id_card}\nPermit: {permit}\nStatus: {status}\n")
                self.log_box.insert(tk.END, f"[LOG] {found_owner} | {plate_text} \n")
                self.log_box.see(tk.END)


    
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        imgtk = ImageTk.PhotoImage(image=img_pil)
        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)
        self.root.after(10, self.update_frame)

        


root = tk.Tk()
app = SurveillanceApp(root)
root.mainloop()
