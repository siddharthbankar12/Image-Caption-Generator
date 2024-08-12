import tkinter as tk
from tkinter import filedialog, Label, Button, Text, Scrollbar, Frame
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.applications.xception import Xception, preprocess_input
from keras.models import load_model
from pickle import load
import numpy as np
from PIL import Image, ImageTk

# Function to extract features from the image
def extract_features(filename, model):
    try:
        image = Image.open(filename)
        image = image.resize((299, 299))
        image = np.array(image)

        if image.shape[2] == 4:
            image = image[..., :3]

        image = np.expand_dims(image, axis=0)
        image = preprocess_input(image)
        feature = model.predict(image)
        feature = np.reshape(feature, feature.shape[1])
        return feature

    except Exception as e:
        print(f"ERROR: {e}")
        print("Couldn't open image! Make sure the image path and extension are correct.")
        return None

# Function to convert an integer to a word
def word_for_id(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

# Function to generate a description for the image
def generate_desc(model, tokenizer, photo, max_length):
    in_text = []
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([' '.join(in_text)])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        pred = model.predict([np.expand_dims(photo, axis=0), sequence], verbose=0)
        pred = np.argmax(pred)
        word = word_for_id(pred, tokenizer)
        if word is None or word == 'end':
            break
        in_text.append(word)
    return ' '.join(in_text)

# Load necessary models and tokenizer
max_length = 32
tokenizer = load(open("tokenizer.p", "rb"))
model = load_model('models/model_9.h5')
xception_model = Xception(include_top=False, pooling="avg")

# Function to upload an image and generate a caption
def upload_image():
    global img, photo
    img_path = filedialog.askopenfilename()
    photo = extract_features(img_path, xception_model)
    img = Image.open(img_path)
    img.thumbnail((500, 500))
    img = ImageTk.PhotoImage(img)
    image_label.config(image=img)
    description = generate_desc(model, tokenizer, photo, max_length)
    result_text.delete(1.0, tk.END)
    result_text.insert(tk.END, description)
    result_text.tag_add("center", "1.0", "end")

# Function to refresh the text box
def refresh():
    result_text.delete(1.0, tk.END)
    image_label.config(image='')  # Remove the image

# GUI Setup
root = tk.Tk()
root.geometry("1000x700")
root.title("Image Caption Generator")

# Style Configuration
dark_off_white = "#E0E0E0"  # Dark off-white background color
root.configure(bg=dark_off_white)

# Header Frame
header_frame = Frame(root, bg="#D3D3D3", bd=2, relief="ridge")
header_frame.pack(fill="x", padx=20, pady=(20, 0))

header_label = Label(header_frame, text="Image Caption Generator", font=("Helvetica", 24, "bold"), fg="#000000", bg="#D3D3D3")
header_label.pack(pady=10)

# Refresh Button Frame
refresh_frame = Frame(root, bg=dark_off_white)
refresh_frame.pack(fill="x", padx=20, pady=(10, 20))  # Adjust pady for space between header and refresh frame

# Load the refresh icon
refresh_icon = Image.open("refresh_icon.png")  # Path to your icon file
refresh_icon = refresh_icon.resize((24, 24), Image.LANCZOS)  # Use Image.LANCZOS for high-quality downsampling
refresh_icon = ImageTk.PhotoImage(refresh_icon)

# Adding the refresh button with icon to the refresh frame
refresh_button = Button(refresh_frame, image=refresh_icon, command=refresh, bg=dark_off_white, relief="flat")
refresh_button.image = refresh_icon  # Keep a reference to avoid garbage collection
refresh_button.pack(side=tk.RIGHT, padx=10)

# Main Frame for Centering Content
main_frame = Frame(root, bg=dark_off_white)
main_frame.pack(pady=(10, 20), padx=20, fill="both", expand=True)

# Centering the widgets in the main_frame
main_frame.grid_rowconfigure(0, weight=1)
main_frame.grid_rowconfigure(1, weight=1)
main_frame.grid_rowconfigure(2, weight=1)
main_frame.grid_columnconfigure(0, weight=1)
main_frame.grid_columnconfigure(1, weight=1)

image_label = Label(main_frame, bg=dark_off_white)
image_label.grid(row=0, column=0, columnspan=2, padx=20, pady=20, sticky="nsew")

# Centering the button horizontally
upload_button = Button(main_frame, text="Upload Image", command=upload_image, font=("Times New Roman", 16), bg="#D3D3D3", fg="#000000", relief="flat", padx=10, width=20)
upload_button.grid(row=1, column=0, columnspan=2, pady=10, sticky="nsew")

result_frame = Frame(main_frame, bg=dark_off_white)
result_frame.grid(row=2, column=0, columnspan=2, pady=20, sticky="nsew")

result_text = Text(result_frame, height=10, width=60, font=("Helvetica", 14), wrap="word", bg="#FFFFFF", fg="#000000", relief="flat", padx=10, pady=10)
result_text.pack(side=tk.LEFT, fill="both", expand=True)

scroll = Scrollbar(result_frame, command=result_text.yview, bg=dark_off_white)
scroll.pack(side=tk.RIGHT, fill=tk.Y)

result_text.config(yscrollcommand=scroll.set)

# Center-align the text in the Text widget
result_text.tag_configure("center", justify="center")

footer_frame = Frame(root, bg=dark_off_white)
footer_frame.pack(fill="x", padx=20, pady=20)

footer_label = Label(footer_frame, text="Developed by Siddharth Bankar", font=("Helvetica", 12), fg="#000000", bg=dark_off_white)
footer_label.pack(pady=10)

root.mainloop()
