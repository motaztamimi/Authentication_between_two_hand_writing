import multiprocessing
from time import sleep
import tkinter as tk
from tkinter import (
    HORIZONTAL,
    Frame,
    Label,
    PhotoImage,
    StringVar,
    filedialog,
    messagebox,
    ttk,
    Radiobutton,
)
import pandas as pd
from multiprocessing import Queue, freeze_support
import threading
from Generic_GUI import testing_excel
from PIL import Image, ImageTk
from matplotlib import pyplot as plt
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import skimage
import time
lang = ["Arabic", "Hebrew", "English"]
mode = ["CrossEntropy", "Triplet"]


class MainGUI(Frame):
    
    def __init__(self, root):
        Frame.__init__(self, root)
        self.root = root
        self.root.geometry("900x600")
        self.root.title("Author Verification Based On Hand Writing Analysis")
        self.root_img = tk.PhotoImage(file=r"..\images\root_bg.png")
        self.root_label = tk.Label(root, image=self.root_img)
        self.root_label.pack()
        self.root.pack_propagate(False)
        self.root.resizable(1, 0)
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.z=1
        # # ------------------------------ #
        # #          logo imagee           #
        # # ------------------------------ #
        self.logo_img = Image.open(r"..\images\logo.png")
        self.logo_resize_image = self.logo_img.resize((330, 100))
        self.logo_img = ImageTk.PhotoImage(self.logo_resize_image)
        self.logo_label = tk.Label(self.root, image=self.logo_img)
        self.logo_label.place(rely=0.76, relx=0.63)
        self.data_loading_label = tk.Label(root,background="white",foreground="black",font=(30))
        if 1:
            # # ------------------------------ #
            # #          File frame           #
            # # ------------------------------ #
            self.file_choosee = tk.LabelFrame(self.root, background="white")
            self.file_choosee.place(height=200, width=200, rely=0.6, relx=0.0)
            self.file_hidden_button = tk.Button(
                self.file_choosee,
                background="white",
                borderwidth=0,
                command=lambda: self.File_dialog(),
                cursor="dot",
            )
            self.file_hidden_button.place(height=200, width=200)
            self.file_label = tk.Label(
                self.file_choosee,
                text="Excel File",
                background="#0b5394",
                foreground="white",
                font=(20),
            )
            self.file_label.place(relx=0, rely=0, relwidth=1)

            self.file_Label_desc = tk.Label(
                self.file_choosee,
                text="Click to Choose excel file",
                background="white",
                foreground="blue",
            )
            self.file_Label_desc.place(relx=0, rely=0.7, relwidth=1)
            self.file_img = Image.open(r"..\images\filee.png")
            self.file_resize_image = self.file_img.resize((70, 70))
            self.file_img = ImageTk.PhotoImage(self.file_resize_image)
            self.file_button = tk.Button(
                self.file_choosee,
                image=self.file_img,
                borderwidth=0,
                cursor="dot",
                command=lambda: self.File_dialog(),
            )
            self.file_button.place(rely=0.3, relx=0.35)
            self.file_path = ""
            # # ------------------------------ #
            # #          folder frame          #
            # # ------------------------------ #
            self.folder_choosee = tk.LabelFrame(self.root, background="white")
            self.folder_choosee.place(height=200, width=200, rely=0.6, relx=0.225)
            self.folder_hidden_button = tk.Button(
                self.folder_choosee,
                background="white",
                borderwidth=0,
                command=lambda: self.Folder_dialog(),
                cursor="dot",
            )
            self.folder_hidden_button.place(height=200, width=200)
            self.folder_label = tk.Label(
                self.folder_choosee,
                text="Data Folder",
                background="#0b5394",
                foreground="white",
                font=(20),
            )
            self.folder_label.place(relx=0, rely=0, relwidth=1)
            self.folder_Label_desc = tk.Label(
                self.folder_choosee,
                text="Click to Choose data folder",
                background="white",
                foreground="blue",
            )
            self.folder_Label_desc.place(relx=0, rely=0.7, relwidth=1)
            self.folder_img = Image.open(r"..\images\folder.jpg")
            self.folder_resize_image = self.folder_img.resize((70, 70))
            self.foler_img = ImageTk.PhotoImage(self.folder_resize_image)
            self.folder_button = tk.Button(
                self.folder_choosee,
                image=self.foler_img,
                borderwidth=0,
                cursor="dot",
                command=lambda: self.Folder_dialog(),
            )
            self.folder_button.place(rely=0.3, relx=0.35)
            self.folder_path = ""
        # # ------------------------------ #
        # #          Language frame        #
        # # ------------------------------ #
        if 1:
            self.lang = StringVar()
            self.lang.set("Arabic")
            self.lang_choosee = tk.LabelFrame(self.root, background="white")
            self.lang_choosee.place(height=200, width=150, rely=0.6, relx=0.452)
            self.lang_label = tk.Label(
                self.lang_choosee,
                text="Language",
                background="#0b5394",
                foreground="white",
                font=(10),
            )
            self.lang_label.place(relx=0, rely=0, relwidth=1)
            chkbtn1 = Radiobutton(
                self.lang_choosee,
                text="Arabic",
                variable=self.lang,
                value="Arabic",
                background="white",
            )
            chkbtn1.place(x=0, y=50)
            chkbtn2 = Radiobutton(
                self.lang_choosee,
                text="Hebrew",
                variable=self.lang,
                value="Hebrew",
                background="white",
            )
            chkbtn2.place(x=0, y=80)
            chkbtn3 = Radiobutton(
                self.lang_choosee,
                text="English",
                variable=self.lang,
                value="English",
                background="white",
            )
            chkbtn3.place(x=0, y=110)
            self.lang_Label_desc = tk.Label(
                self.lang_choosee,
                text="Choose a Language",
                background="white",
                foreground="blue",
            )
            self.lang_Label_desc.place(relx=0, rely=0.7, relwidth=1)
            # Progress bar
            self.Progress_Bar = ttk.Progressbar(
                self.root, orient=HORIZONTAL, length=500, mode="determinate"
            )
            self.Progress_Bar.place(rely=0.52, relx=0)
            self.txt = tk.Label(self.root, text="0%", bg="#345", fg="#fff")
            self.txt.place(relx=0.57, rely=0.52)

        if 1:
            # ------------------------------ #
            #       Calculate button         #
            # ------------------------------ #
            self.calculate_button = tk.Button(
                root,
                text="Calculate",
                background="#6aa84f",
                foreground="white",
                font=(10),
                border="1",
                borderwidth=3,
                command=lambda: self.Testing_model(),
            )
            self.calculate_button.config(width=15, height=2)
            self.calculate_button.place(relx=0.63, rely=0.6)
            # ------------------------------ #
            #          Save button           #
            # ------------------------------ #
            self.button5 = tk.Button(
                root,
                text="Save",
                command=lambda: self.Save_file(),
                font=(10),
                background="#EB4255",
                foreground="white",
                border="1",
                borderwidth=3,
            )
            self.button5.config(width=15, height=2)
            self.button5.place(relx=0.82, rely=0.6)
            # ------------------------------ #
            #          Combo Box mode        #
            # ------------------------------ #
            self.mode_label = tk.Label(root, text="please select a mode", bg="#93c4fc")
            # self.mode_label.place(relx=0.8, rely=0.15)
            self.mode_list = ttk.Combobox(root, values=mode)
            self.mode_list.current(0)
            # self.mode_list.place(relx=0.8, rely=0.6)
            # EXCEL TABLE
            self.Excel_Box = tk.LabelFrame(
                self.root, text="Excel Data", background="#e2ebf9"
            )
            self.Excel_Box.place(height=300, width=500)
            style = ttk.Style(self)
            aktualTheme = style.theme_use()
            style.theme_create("dummy", parent=aktualTheme)
            style.theme_use("dummy")  # style.theme_use("vista")
            self.tv1 = ttk.Treeview(self.Excel_Box)
            self.tv1.tag_configure("Same", background="lightskyblue")
            self.tv1.tag_configure("diffrent",background="red")
            self.tv1.tag_configure("white",background="white")

            self.tv1.place(relheight=1, relwidth=1, width=1)
            self.treeScrolly = tk.Scrollbar(
                self.Excel_Box, orient="vertical", command=self.tv1.yview
            )
            self.tv1.configure(yscrollcommand=self.treeScrolly)
            self.treeScrolly.pack(side="right", fill="y")

            self.excel_path = ""
            self.data_path = ""
            self.isrunning = False
            self.start = False
            self.queue = Queue()
        
            self.size = 100
            self.queue2 = Queue()
            self.arabic_model = r"..\images\models\model_arabic_epoch_10.pt"
            self.hebrew_model = r"..\images\models\model_hebrew_epoch_10.pt"
            self.english_model = r"..\images\models\model_english_epoch_7.pt"
            self.model_bylang = {
                "Arabic": self.arabic_model,
                "Hebrew": self.hebrew_model,
                "English": self.english_model,
            }
            self.mode_loss = {"CrossEntropy": False, "Triplet": True}
            self.credit = Label(
                self.root,
                text="Developed By Motaz Tamimi & Mustafa Abu Ghanam (2022), All right reserved.",
                font=(20),
                background="#1d2028",
                fg="white",
            )
            self.credit.place(
                rely=0.95,
                relx=0,
            )
        # # ------------------------------ #
        # #          chart frame           #
        # # ------------------------------ #
        self.same = 1
        self.difrrent = 0
        self.chart_frame = tk.LabelFrame(self.root, borderwidth=0, background="black")
        self.fig = plt.figure(
            figsize=(3, 2.5),
            dpi=100,
        )
        self.label = ["same", "diff"]
        self.sizes = [10, 15]
        self.color = ["gold", "red"]
        self.ex = (0, 0.2)
        plt.pie(self.sizes, explode=self.ex, labels=self.label, colors=self.color)
        plt.axis("equal")
        can = FigureCanvasTkAgg(self.fig, self.chart_frame)
        can.draw()
        self.vv = 0
        can.get_tk_widget().place(relx=0, rely=0)

    def plot_values(self):
        self.chart_frame = tk.LabelFrame(self.root, borderwidth=3, background="white")
        self.chart_frame.place(relx=0.56, rely=0, height=300, width=400)
        self.fig = plt.figure(figsize=(3.1, 2.5), dpi=100, facecolor="white")
        self.label = ["diffrent", "Same"]
        self.sizes = [self.difrrent, self.same]
        self.color = ["red", "lightskyblue"]
        self.ex = (0, 0.2)
        plt.pie(
            self.sizes,
            explode=self.ex,
            labels=self.label,
            colors=self.color,
            shadow=True,
            startangle=140,
            autopct="%1.1f%%",
        )
        ax = plt.axis("equal")
        chart = FigureCanvasTkAgg(self.fig, self.chart_frame)
        chart.draw()
        chart.get_tk_widget().place(relx=0, rely=0)
        return

    def Save_file(self):
        if not self.start:
            messagebox.showerror("information", "still not start")
            return
        if not self.vv or not self.vv == 100:
            messagebox.showerror("information", "still not finish")
            return
        savefile = filedialog.asksaveasfilename(
            filetypes=(("Excel files", "*.xlsx"), ("All files", "*.*"))
        )
        filename = "../final_result"
        filename += ".xlsx"
        with pd.ExcelWriter(savefile + ".xlsx") as writer:
            data = pd.read_excel(filename)
            data.to_excel(writer, header=True)

    def File_dialog(self):
        filename = filedialog.askopenfilename(
            initialdir="/",
            title="Select a file",
            filetypes=(("xlsx files", "*.xlsx"), ("csv files", "*.csv")),
        )
        if filename:
            messagebox.showinfo("information", f"excel file uploaded /n {filename}")
            self.file_path = filename
            self.Load_excel_data()
        pass

    def Folder_dialog(self):
        folder_name = filedialog.askdirectory(initialdir="/", title="select a folder")
        if folder_name:
            messagebox.showinfo("information", f"data folder uploaded /n {folder_name}")
            self.folder_path = folder_name
        pass

    def Load_excel_data(self):
        file_path = self.file_path
        try:
            excel_file = r"{}".format(file_path)
            df = pd.read_excel(excel_file)
        except ValueError:
            messagebox.showerror("Information", "the file you have choseen invalid")
            return None
        except FileNotFoundError:
            messagebox.showerror("Information", "no such file like this")
            return None
        self.clear_data()
        self.tv1["column"] = list(df.columns)
        self.tv1.column("first", anchor="center", width=70)
        self.tv1.column("second", anchor="center", width=70)
        self.tv1["show"] = "headings"
        for colum in self.tv1["column"]:
            self.tv1.heading(colum, text=colum)
        df_rows = df.to_numpy().tolist()
        for row in df_rows:
            self.tv1.insert("", "end", values=row,tags="white")
        return None

    def clear_data(self):
        self.tv1.delete(*self.tv1.get_children())
        pass

    def keyy(self):
        t = threading.currentThread()


        while self.isrunning and getattr(t, "do_run", True):

            if self.queue:
                out = self.queue.get()
                self.vv = int((out / self.size) * 100)
                self.txt["text"] = f"{self.vv} %"
                self.Progress_Bar["value"] = ((out / self.size)) * 100
                if self.z ==1:
                    self.data_loading_label["text"] = "Finish"
                    self.data_loading_label["text"] = ""
                    self.data_loading_label.after(1000, self.data_loading_label.destroy())
                    self.z=0

            if self.queue2:
                out1 = self.queue2.get()
                tag = ""
                if out1[3] == "Same":
                    tag = "Same"
                    self.same += 1
                else:
                    tag = "diffrent"
                    self.difrrent += 1
                print(tag)
                self.plot_values()
                self.tv1.insert("", 0, values=out1, tags=(f"{tag}"))
        print("finishing keyy function")
        return

    def work(self):
        print(" in test")
        self.clear_data()
        self.tv1["column"] = ["first", "second", "Result", "simple_result"]
        self.tv1["show"] = "headings"
        self.tv1.column("first", anchor="center", width=70)
        self.tv1.column("second", anchor="center", width=70)
        self.tv1.column("Result", anchor="center", width=70)
        self.tv1.column("simple_result", anchor="center", width=90)

        for colum in self.tv1["column"]:
            self.tv1.heading(colum, text=colum)
        reading_for_Size = pd.read_excel(self.excel_path)
        self.size = reading_for_Size.shape[0]
        self.p = multiprocessing.Process(
            target=testing_excel,
            args=(
                self.excel_path,
                self.data_path,
                self.model_bylang[self.lang.get()],
                self.queue,
                self.queue2,
                self.mode_loss[self.mode_list.get()],
              
            ),
        )
        self.p.start()
        self.q = threading.Thread(target=self.keyy, daemon=True)
        self.q.start()
        print("in work")
        return self.q

    def Testing_model(self):
        self.start = True
        if self.isrunning:
            return
        if not self.file_path or not self.folder_path:
            messagebox.showerror("erorr", "must upload excel file and data folder")
            return 
        self.isrunning = True
        self.data_loading_label["text"] = "Loading data ..."
        self.data_loading_label.place(relx=0.7,rely=0.52)
  
        excelpath = r"{}".format(self.file_path)
        data_path = r"{}".format(self.folder_path)
        self.excel_path = excelpath
        self.data_path = data_path
        self.work()
        return

    def on_closing(self):
        if messagebox.askokcancel("Quit", "Do you want to quit?"):
            self.isrunning =False
            if  getattr(self, "p", None):
                self.p.kill()
                self.p.terminate()
                self.p.join()
            if getattr(self, 'q',None):
                self.q.do_run = False
            root.destroy()

if __name__ == "__main__":
    freeze_support()
    root = tk.Tk()
    MainGUI(root)
    root.mainloop()
