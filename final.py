import tensorflow as tf
import cv2
import imutils
import numpy as np
import pyttsx3
from tkinter import *
from tkinter import filedialog, messagebox
from tensorflow.keras.models import load_model
from PIL import Image, ImageTk

r = Tk()
r.title("Sign Language Recognition System")
r.geometry("1100x650")
r.configure(bg="#f7f9fc")
r.resizable(False, False)

bg = None
predicted_word = ""

model = load_model("modelslr.h5")

labels = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

engine = pyttsx3.init()

def segment(image, threshold=25):
    global bg
    diff = cv2.absdiff(bg.astype("uint8"), image)
    thresholded = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)[1]
    cnts, _ = cv2.findContours(thresholded.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(cnts) == 0:
        return None
    segmented = max(cnts, key=cv2.contourArea)
    return thresholded, segmented

def run_avg(image, aWeight):
    global bg
    if bg is None:
        bg = image.copy().astype("float")
        return
    cv2.accumulateWeighted(image, bg, aWeight)

def preprocess(img):
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    img = cv2.resize(img, (64, 64))
    img = img / 255.0
    return np.expand_dims(img, axis=0)

def reset_word():
    global predicted_word
    predicted_word = ""
    word_label.config(text="Current Word: ")

def dtect_hand():
    try:
        global bg
        bg = None
        aWeight = 0.5
        camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        top, right, bottom, left = 10, 350, 225, 590
        num_frames = 0

        while True:
            _, frame = camera.read()
            frame = imutils.resize(frame, width=700)
            frame = cv2.flip(frame, 1)
            clone = frame.copy()

            roi = frame[top:bottom, right:left]
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (7, 7), 0)

            if num_frames < 30:
                run_avg(gray, aWeight)
            else:
                hand = segment(gray)
                if hand is not None:
                    _, segmented = hand
                    cv2.drawContours(clone, [segmented + (right, top)], -1, (0, 0, 255))

            cv2.rectangle(clone, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.imshow("Video Feed", clone)
            num_frames += 1

            key = cv2.waitKey(1) & 0xFF
            if key == 32 and num_frames > 30:
                cv2.imwrite("Clicked_image.png", gray)
                predict_image("Clicked_image.png")
                break
            if key == ord('q'):
                break

        camera.release()
        cv2.destroyAllWindows()

    except:
        messagebox.showwarning("WARNING", "PLEASE SHOW YOUR HAND INSIDE THE BOX!")
        camera.release()
        cv2.destroyAllWindows()

def predict_image(path):
    global predicted_word
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    processed = preprocess(img)
    pred = model.predict(processed)
    letter = labels[np.argmax(pred)]
    predicted_word += letter
    word_label.config(text=f"Current Word: {predicted_word}")
    engine.say(letter)
    engine.runAndWait()

def add_photo():
    file = filedialog.askopenfilename(filetypes=[("Images", "*.jpg *.png *.jpeg")])
    if file:
        img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
        processed = preprocess(img)
        pred = model.predict(processed)
        letter = labels[np.argmax(pred)]
        word_label.config(text=f"Current Word: {letter}")
        engine.say(letter)
        engine.runAndWait()

Label(
    r,
    text="SIGN LANGUAGE RECOGNITION SYSTEM",
    font=("Helvetica", 22, "bold"),
    bg="#f7f9fc",
    fg="#2c4aa3"
).pack(pady=20)

main_frame = Frame(r, bg="#f7f9fc")
main_frame.pack(fill=BOTH, expand=True, padx=20)

left_panel = Frame(main_frame, bg="#f7f9fc")
left_panel.pack(side=LEFT, padx=20)

btn_frame = Frame(left_panel, bg="#f7f9fc")
btn_frame.pack(pady=30)

Button(btn_frame, text="📷 OPEN CAMERA", width=22, bg="#ffd24d", command=dtect_hand).grid(row=0, column=0, padx=10)
Button(btn_frame, text="🖼 INPUT PHOTO", width=22, bg="#ffd24d", command=add_photo).grid(row=0, column=1, padx=10)

Button(btn_frame, text="🧹 CLEAR", width=22, bg="#4e73df", fg="white", command=reset_word).grid(row=1, column=0, pady=15)
Button(btn_frame, text="❌ CLOSE", width=22, bg="#4e73df", fg="white", command=r.destroy).grid(row=1, column=1, pady=15)

word_label = Label(
    left_panel,
    text="Current Word: ",
    font=("Helvetica", 16, "bold"),
    bg="white",
    width=45,
    anchor="w",
    padx=10,
    pady=10
)
word_label.pack(pady=30)

right_panel = Frame(main_frame, bg="#f7f9fc")
right_panel.pack(side=RIGHT, padx=20)

Label(
    right_panel,
    text="ASL Alphabet Reference",
    font=("Helvetica", 14, "bold"),
    bg="#f7f9fc"
).pack(pady=10)

asl_img = Image.open("backdown.jpg")
asl_img = asl_img.resize((420, 420))
asl_photo = ImageTk.PhotoImage(asl_img)

asl_label = Label(right_panel, image=asl_photo, bg="#f7f9fc")
asl_label.image = asl_photo
asl_label.pack(pady=10)

r.mainloop()
