from ast import Lambda
from cProfile import label
import multiprocessing
from pickle import TRUE
from struct import pack
import tkinter as tk
from tkinter import HORIZONTAL, PhotoImage, filedialog, messagebox, ttk
from typing import Text
from matplotlib.pyplot import fill
import pandas as pd 
import GUI
import time
from multiprocessing import Process, Queue
import threading

root = tk.Tk()
root.geometry("900x600")
root.title("Motaz GUI")
root.config(background="#345")
root.pack_propagate(False)
root.resizable(0,0)

img = PhotoImage(file=r"C:\Users\97258\Desktop\download.png")
labl= tk.Label(root,image=img)
labl.place(x=530,y=400)

# the upper box
frame1 = tk.LabelFrame(root, text="Excel Data",background="#e2ebf9")
frame1.place(height=300, width=700)

pb = ttk.Progressbar(root,orient=HORIZONTAL, length=500,mode= "determinate")
pb.place(rely=0.52,relx=0)


txt = tk.Label(root,text = '0%',bg = '#345',fg = '#fff')
txt.place(relx=0.57,rely=0.52)
#the under box
file_frame = tk.LabelFrame(root, text="Open file")
file_frame.place(height=200, width=450, rely=0.6, relx=0)





#right button 
button1= tk.Button(file_frame, text="Browse A file",background="#93c4fc",command=lambda: File_dialog())
button1.place(rely=0.8,relx=0.5)

button2 = tk.Button(file_frame,bg="#7bfdc5", text="Load File",command=lambda: Load_excel_data())
button2.place(rely=0.8,relx=0.3)

button3 = tk.Button(file_frame, text = "chose directory",background="#93c4fc", command=lambda: Folder_dialog())
button3.place(rely=0.8,relx=0.7)

button4 = tk. Button(root, text="Calculate",background="#8ed49f",command=lambda:Testing_model())
button4.config(width=20,height=3)
button4.place(relx=0.8,rely=0.5)
Label_file = tk.Label(file_frame, text="No File Selected")
Label_file.place(rely=0, relx=0)

Label_folder = tk.Label(file_frame, text="No such Folder selected")
Label_folder.place(rely=0.1,relx=0)

tv1 = ttk.Treeview(frame1)
tv1.place(relheight=1, relwidth=1)

treeScrolly = tk.Scrollbar(frame1, orient="vertical", command=tv1.yview)
treeScrollx = tk.Scrollbar(frame1, orient="horizontal", command=tv1.xview)

tv1.configure(xscrollcommand=treeScrollx, yscrollcommand=treeScrolly)
treeScrolly.pack(side="right",fill="y")
treeScrollx.pack(side="bottom",fill="x")



def File_dialog():
    filename = filedialog.askopenfilename(initialdir='/', title = "Select a file", filetypes=(("xlsx files","*.xlsx"),))
    Label_file["text"] = filename
    pass


def Folder_dialog():
    folder_name = filedialog.askdirectory(initialdir='/', title="select a folder")
    Label_folder["text"] = folder_name
    pass

def Load_excel_data():
    file_path = Label_file["text"]
    try:
        excel_file = r"{}".format(file_path)
        df = pd.read_excel(excel_file)
    except ValueError:
        messagebox.showerror("Information", "the file you have choseen invalid")
        return None
    except FileNotFoundError:
        messagebox.showerror("Information","no such file like this")
        return None
    clear_data()
    tv1["column"] = list(df.columns)
    tv1["show"] = "headings"

    for colum in tv1["column"]:
        tv1.heading(colum, text = colum)
    df_rows = df.to_numpy().tolist()
    for row in df_rows:
        tv1.insert("","end", values=row)
    return None            



def clear_data():
    tv1.delete(*tv1.get_children())
    pass

def Testing_model():
    excelpath= r"{}".format( Label_file["text"])
    data_path= r"{}".format(Label_folder["text"])
    excel= GUI.testing_excel(excelpath,data_path)
    add_to_excel(excel=excel)
def add_to_excel(excel):
    df = pd.read_csv(excel)
    clear_data()
    tv1["column"] = list(df.columns)
    tv1["show"] = "headings"

    for colum in tv1["column"]:
        tv1.heading(colum, text = colum)
    df_rows = df.to_numpy().tolist()
    for row in df_rows:
        tv1.insert("","end", values=row)



def step():
    value= GUI.step_excel()
    pb['value'] += (0.6*100)
    time.sleep(1)
    txt['text']=pb['value'],'%'


# def mainfunc():
#     DontExit=True
#     p1=multiprocessing.Process(target=Testing_model)
#     p1.start()
    
#     p2=multiprocessing.Process(target=step)
#     p2.start()
#     p1.join()
#     DontExit=False
#     add_to_excel("final.csv")

root.mainloop()