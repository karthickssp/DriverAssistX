#Dark Theme
import tkinter as tk
from tkinter import filedialog, Label, Button, Frame
from PIL import Image, ImageTk

def init_gui():
    global top, label, sign_image, classify_button
    
    # Main window settings
    top = tk.Tk()
    top.geometry('1200x800')
    top.title('AI - Based Traffic Sign Recognition and Hand Gesture Recognition System')
    top.configure(background='#1B2631') 
    
    # Header Frame
    header_frame = Frame(top,bg='#2E4053', pady=20)
    header_frame.pack(fill=tk.X)
    
    # Title
    title = Label(header_frame, 
                 text="🚦 Traffic Sign & ✋ Gesture Recognition 🤖 ", 
                 font=('Helvetica', 28, 'bold'),
                 bg='#2E4053', 
                 fg='#F4D03F')
    title.pack()
    
    # Subtitle
    subtitle = Label(header_frame,
                    text="Upload an image for AI-powered recognition",
                    font=('Helvetica', 14),
                    bg='#2E4053', 
                    fg='#F4D03F')
    subtitle.pack(pady=5)
    
    # Main Content Frame
    content_frame = Frame(top, bg="#1B2631")
    content_frame.pack(expand=True, fill=tk.BOTH, pady=20)
    
    # Image Display Area
    image_frame = Frame(content_frame, bg='#2E4053', padx=80, pady=80)
    image_frame.pack(expand=True)
    
    sign_image = Label(image_frame, bg='#2E4053',)
    sign_image.pack()
    
    # Status Label
    label = Label(content_frame,
                 text="📤 Upload an image to begin analysis",
                 font=('Helvetica', 16),
                 bg='#1B2631', 
                 fg='#F7F9F9')
    label.pack(pady=20)
    
    # Button Frame
    button_frame = Frame(top, bg='#1B2631', pady=30)
    button_frame.pack()
    
    # Button styles
    button_style = {
        'font': ('Helvetica', 16, 'bold'),
        'padx': 20,
        'pady': 10,
        'cursor': 'hand2',
        'relief':'raised'
    }
    
    # Upload Button
    upload = Button(button_frame,
                   text="📂 Upload Image",
                   command=upload_image,
                   bg='#28B463', fg='white',
                   **button_style)
    upload.pack(side=tk.LEFT, padx=20)
    
    # Classify Button
    classify_button = Button(button_frame,
                       text="🔍 Classify Image",
                       bg='#F39C12', 
                       fg='white', 
                       state=tk.DISABLED,
                       **button_style)
    classify_button.pack(side=tk.LEFT, padx=10)
    
    # Exit Button
    exit_btn = Button(button_frame,
                     text="❌ Exit",
                     command=top.destroy,
                     bg='#E74C3C',
                     fg='white',
                     **button_style)
    exit_btn.pack(side=tk.LEFT, padx=10)
    
    return top, label, sign_image, classify_button

def show_classify_button(file_path):
    classify_button.config(command=lambda: classify(file_path), state=tk.NORMAL)

def upload_image():
    try:
        file_path = filedialog.askopenfilename()
        if not file_path:
            return
        uploaded = Image.open(file_path)
        uploaded.thumbnail((400, 400))  # Resize image
        im = ImageTk.PhotoImage(uploaded)
        sign_image.configure(image=im)
        sign_image.image = im
        label.configure(text='✅ Image Loaded Successfully', fg='#28B463', font=('Helvetica', 14, 'bold'))
        show_classify_button(file_path)
    except Exception as e:
        label.configure(text=f"Error: {str(e)}", fg='#f44336')
        print("Error:", e)

def classify(file_path):
    label.configure(text=f"🔍 Classified: {file_path.split('/')[-1]}", fg='#F39C12', font=('Helvetica', 14, 'bold'))

if __name__ == "__main__":
    top, label, sign_image, classify_button = init_gui()
    top.mainloop()

#White Theme

"""
import tkinter as tk
from tkinter import filedialog, Label, Button, Frame
from PIL import Image, ImageTk

def init_gui():
    global top, label, sign_image, classify_button
    
    # Main window settings
    top = tk.Tk()
    top.geometry('1200x800')
    top.title('AI - Based Traffic Sign Recognition and Hand Gesture Recognition System')
    top.configure(background='#F8F9FA') 
    
    # Header Frame
    header_frame = Frame(top, bg='#E9ECEF', pady=20)
    header_frame.pack(fill=tk.X)
    
    # Title
    title = Label(header_frame, 
                 text="🚦 Traffic Sign & ✋ Gesture Recognition 🤖 ", 
                 font=('Helvetica', 28, 'bold'),
                 bg='#E9ECEF', 
                 fg='#212529')
    title.pack()
    
    # Subtitle
    subtitle = Label(header_frame,
                    text="Upload an image for AI-powered recognition",
                    font=('Helvetica', 14),
                    bg='#E9ECEF', 
                    fg='#212529')
    subtitle.pack(pady=5)
    
    # Main Content Frame
    content_frame = Frame(top, bg="#F8F9FA")
    content_frame.pack(expand=True, fill=tk.BOTH, pady=20)
    
    # Image Display Area
    image_frame = Frame(content_frame, bg='#DEE2E6', padx=80, pady=80)
    image_frame.pack(expand=True)
    
    sign_image = Label(image_frame, bg='#DEE2E6',)
    sign_image.pack()
    
    # Status Label
    label = Label(content_frame,
                 text="📤 Upload an image to begin analysis",
                 font=('Helvetica', 16),
                 bg='#F8F9FA', 
                 fg='#343A40')
    label.pack(pady=20)
    
    # Button Frame
    button_frame = Frame(top, bg='#F8F9FA', pady=30)
    button_frame.pack()
    
    # Button styles
    button_style = {
        'font': ('Helvetica', 16, 'bold'),
        'padx': 20,
        'pady': 10,
        'cursor': 'hand2',
        'relief':'raised'
    }
    
    # Upload Button
    upload = Button(button_frame,
                   text="📂 Upload Image",
                   command=upload_image,
                   bg='#0D6EFD', fg='white',
                   **button_style)
    upload.pack(side=tk.LEFT, padx=20)
    
    # Classify Button
    classify_button = Button(button_frame,
                       text="🔍 Classify Image",
                       bg='#FFC107', 
                       fg='black', 
                       state=tk.DISABLED,
                       **button_style)
    classify_button.pack(side=tk.LEFT, padx=10)
    
    # Exit Button
    exit_btn = Button(button_frame,
                     text="❌ Exit",
                     command=top.destroy,
                     bg='#DC3545',
                     fg='white',
                     **button_style)
    exit_btn.pack(side=tk.LEFT, padx=10)
    
    return top, label, sign_image, classify_button

def show_classify_button(file_path):
    classify_button.config(command=lambda: classify(file_path), state=tk.NORMAL)

def upload_image():
    try:
        file_path = filedialog.askopenfilename()
        if not file_path:
            return
        uploaded = Image.open(file_path)
        uploaded.thumbnail((400, 400))  # Resize image
        im = ImageTk.PhotoImage(uploaded)
        sign_image.configure(image=im)
        sign_image.image = im
        label.configure(text='✅ Image Loaded Successfully', fg='#198754', font=('Helvetica', 16, 'bold'))
        show_classify_button(file_path)
    except Exception as e:
        label.configure(text=f"Error: {str(e)}", fg='#DC3545')
        print("Error:", e)

def classify(file_path):
    file_name = file_path.split('/')[-1].rsplit('.', 1)[0] 
    label.configure(text=f"🔍 Classified: {file_name}", fg='#212529', font=('Helvetica', 16, 'bold'))

if __name__ == "__main__":
    top, label, sign_image, classify_button = init_gui()
    top.mainloop()


"""