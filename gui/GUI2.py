import multiprocessing
import tkinter as tk
from tkinter import HORIZONTAL, Frame, Label, PhotoImage, filedialog, messagebox, ttk
from turtle import width
import pandas as pd
from multiprocessing import Queue, freeze_support
import threading
from Generic_GUI import testing_excel

lang = ["Arabic", "Hebrew", "English"]
mode = ["CrossEntropy", "Triplet"]
class MainGUI(Frame):
    

    def __init__(self ,root):
        Frame.__init__(self, root)
        self.root=root
        self.root.geometry("900x600")
        self.root.title("Author Verification Based On Hand Writing Analysis")
        self.root.config(background="#345")
        self.root.pack_propagate(False)
        self.root.resizable(0,0)
        #LOGO IMG
        self.Logo_Image = PhotoImage(file=r"..\images\download.png")
        self.Logo_label = tk.Label(self.root,image=self.Logo_Image)
        self.Logo_label.place(x=530,y=400)
        # EXCEL TABLE
        self.Excel_Box = tk.LabelFrame(self.root, text="Excel Data",background="#e2ebf9")
        self.Excel_Box.place(height=300, width=500)
        #Progress bar
        self.Progress_Bar =  ttk.Progressbar(self.root,orient=HORIZONTAL, length=500,mode= "determinate")
        self.Progress_Bar.place(rely=0.52,relx=0)
        # Down_Box
        self.txt = tk.Label(self.root,text = '0%',bg = '#345',fg = '#fff')
        self.txt.place(relx=0.57,rely=0.52)
        #Down_box_tilte
        self.box_title = tk.LabelFrame(self.root, text="choose file and folder")
        self.box_title.place(height=200, width=450, rely=0.6, relx=0)
        # ------------------------------ #
        #          File button           #
        # ------------------------------ #
        self.file_button = tk.Button(self.box_title, text="Browse A file",background="#93c4fc",command=lambda: self.File_dialog())
        self.file_button.place(rely=0.8,relx=0.5)
        # ------------------------------ #
        #     directory button           #
        # ------------------------------ #
        #directory button
        self.folder_button = tk.Button(self.box_title, text = "chose directory",background="#93c4fc",command=lambda: self.Folder_dialog())
        self.folder_button.place(rely=0.8,relx=0.7)
        # ------------------------------ #
        #     load file button           #
        # ------------------------------ #
        self.load_excel_button = tk.Button(self.box_title,bg="#7bfdc5", text="Load File",command=lambda: self.Load_excel_data())
        self.load_excel_button.place(rely=0.8,relx=0.3)
        # ------------------------------ #
        #       Calculate button         #
        # ------------------------------ #
        self.button4 = tk.Button(root, text="Calculate",background="#8ed49f",command=lambda:self.Testing_model())
        self.button4.config(width=20,height=3)
        self.button4.place(relx=0.8,rely=0.4)
        # ------------------------------ #
        #          Save button           #
        # ------------------------------ #
        self.button5 = tk.Button(root, text="Save",background="#8ed49f",command=lambda:self.Save_file())
        self.button5.config(width=20,height=3)
        self.button5.place(relx=0.8,rely=0.3)
        # ------------------------------ #
        #          Combo Box             #
        # ------------------------------ #
        self.lang_label = tk.Label(root, text= "please select a language", bg="#93c4fc")
        self.lang_label.place(relx=0.8, rely=0)
        self.lang_list = ttk.Combobox(root, values=lang)
        self.lang_list.current(0)
        self.lang_list.place(relx=0.8, rely=0.05)
        # ------------------------------ #
        #          Combo Box mode        #
        # ------------------------------ #
        self.mode_label = tk.Label(root, text= "please select a mode", bg="#93c4fc")
        self.mode_label.place(relx=0.8, rely=0.15)
        self.mode_list = ttk.Combobox(root, values=mode)
        self.mode_list.current(0)
        self.mode_list.place(relx=0.8, rely=0.2)
        

        self.Label_file = tk.Label(self.box_title, text="No File Selected")
        self.Label_file.place(rely=0, relx=0)

        self.Label_folder = tk.Label(self.box_title, text="No such Folder selected")
        self.Label_folder.place(rely=0.1,relx=0)


        style = ttk.Style(self)
        aktualTheme = style.theme_use()
        style.theme_create("dummy", parent=aktualTheme)
        style.theme_use("dummy")        # style.theme_use("vista")
        self.tv1 = ttk.Treeview(self.Excel_Box)
        self.tv1.tag_configure('Same', background='yellow')
        self.tv1.place(relheight=1, relwidth=1,width=1)
        self.treeScrolly = tk.Scrollbar(self.Excel_Box, orient="vertical", command=self.tv1.yview)
        self.tv1.configure(yscrollcommand=self.treeScrolly)
        self.treeScrolly.pack(side="right",fill="y")


        self.excel_path=""
        self.data_path = ""
        self.isrunning = False
        self.start = False
        self.queue =Queue()
        self.size =100
        self.queue2 = Queue()
        self.arabic_model = r"..\images\model_0_epoch_12.pt"
        self.hebrew_model = r"..\images\model_0_epoch_11.pt"
        self.model_bylang = {"Arabic" : self.arabic_model ,"Hebrew" :self.hebrew_model}
        self.mode_loss =  {"CrossEntropy" : False, "Triplet": True}
        self.credit = Label(self.root, text="Developed By Motaz Tamimi & Mustafa Abu Ghanam (2022), All right reserved.",font=(20), background="gray",fg="white")
        self.credit.place(rely=0.95,relx=0,height=40)
        
    def Save_file (self):
        if  not self.start:
            messagebox.showerror("information","still not start")
            return
        if self.isrunning:
            messagebox.showerror("information","still not finish")
            return
        savefile = filedialog.asksaveasfilename(filetypes=(("Excel files", "*.xlsx"),
                                                    ("All files", "*.*") ))     
        filename = "../final_result"   
        filename+=".xlsx"     
        with pd.ExcelWriter(savefile+".xlsx") as writer:
            data = pd.read_excel(filename)
            data.to_excel(writer ,header=True)   

    def File_dialog(self):
        filename = filedialog.askopenfilename(initialdir='/', title = "Select a file", filetypes=(("xlsx files","*.xlsx"),("csv files","*.csv")))
        self.Label_file["text"] = filename
        pass


    def Folder_dialog(self):
        folder_name = filedialog.askdirectory(initialdir='/', title="select a folder")
        self.Label_folder["text"] = folder_name
        pass


    def Load_excel_data(self):

        file_path = self.Label_file["text"]
        try:
            excel_file = r"{}".format(file_path)
            df = pd.read_excel(excel_file)
        except ValueError:
            messagebox.showerror("Information", "the file you have choseen invalid")
            return None
        except FileNotFoundError:
            messagebox.showerror("Information","no such file like this")
            return None
        self.clear_data()
        self.tv1["column"] = list(df.columns)
        self.tv1.column("first",anchor="center",width=70)
        self.tv1.column("second",anchor="center",width=70)
        self.tv1["show"] = "headings"
        for colum in self.tv1["column"]:
            self.tv1.heading(colum, text = colum)
        df_rows = df.to_numpy().tolist()
        for row in df_rows:
            self.tv1.insert("","end", values=row)
        return None            


    def clear_data(self):
        self.tv1.delete(*self.tv1.get_children())
        pass


    def keyy(self):
        while self.isrunning:
            if self.queue:
                out= self.queue.get()
                print(out)
                vv = int((out/self.size)*100)
                self.txt["text"] = f"{vv} %"
                self.Progress_Bar["value"] = ((out/self.size))*100
        
            if self.queue2:
                out1 = self.queue2.get()
                tag = ""
                if out1[3] == "Same":
                    tag = "Same"
                else:
                    tag = "diffrent"
                print(tag)
                self.tv1.insert("",0,values=out1, tags=(f"{tag}"))

        return
 

    def work(self):
        print(" in test")
        self.clear_data()
        self.tv1["column"] = ["first","second","Result","simple__result"]
        self.tv1["show"] = "headings"
        self.tv1.column("first",anchor="center",width=70)
        self.tv1.column("second",anchor="center",width=70)
        self.tv1.column("Result",anchor="center",width=70)
        self.tv1.column("simple__result",anchor="center",width=90)

        for colum in self.tv1["column"]:
            self.tv1.heading(colum, text = colum)
        reading_for_Size = pd.read_excel(self.excel_path )
        self.size=reading_for_Size.shape[0]
        p = multiprocessing.Process(target= testing_excel,args=(self.excel_path ,self.data_path, self.model_bylang[self.lang_list.get()] ,self.queue,self.queue2,self.mode_loss[self.mode_list.get()]))
        p.start()
        q = threading.Thread(target=self.keyy )
        q.start()
        print("in work")
        p.join()
        p.terminate()
        self.isrunning =False
        return q
        

    def Testing_model(self):
        self.start = True
        if self.isrunning:
            return
        if self.Label_file["text"] is None or self.Label_folder["text"] is None:
            messagebox.showerror("Information", "file or folder not loaded")
            return

        self.isrunning= True
        excelpath= r"{}".format(self.Label_file["text"])
        data_path= r"{}".format(self.Label_folder["text"])
        self.excel_path=excelpath
        self.data_path=data_path
        t_worker = threading.Thread(target=self.work)
        t_worker.start()
        print("im here in testing")
        if not self.isrunning:
            print("imhere")
            t_worker.join()
        return 


    def add_to_excel(self,excel):
        df = pd.read_csv(excel)
        self.clear_data()
        self.tv1["column"] = list(df.columns)
        self.tv1["show"] = "headings"

        for colum in self.tv1["column"]:
            self.tv1.heading(colum, text = colum)
        df_rows = df.to_numpy().tolist()
        for row in df_rows:
            self.tv1.insert("","end", values=row)
        self.isrunning=False
        return


if __name__ == '__main__':
    freeze_support()
    root = tk.Tk()
    MainGUI(root)
    root.mainloop()