import multiprocessing
import tkinter as tk
from tkinter import HORIZONTAL, Frame, PhotoImage, filedialog, messagebox, ttk
import pandas as pd 
import GUI_triplet
from multiprocessing import Queue
import threading

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
        self.Logo_Image = PhotoImage(file=r"C:\Users\FinalProject\Desktop\download.png")
        self.Logo_label = tk.Label(self.root,image=self.Logo_Image)
        self.Logo_label.place(x=530,y=400)
        # EXCEL TABLE
        self.Excel_Box = tk.LabelFrame(self.root, text="Excel Data",background="#e2ebf9")
        self.Excel_Box.place(height=300, width=700)
        #Progress bar
        self.Progress_Bar =  ttk.Progressbar(self.root,orient=HORIZONTAL, length=500,mode= "determinate")
        self.Progress_Bar.place(rely=0.52,relx=0)
        # Down_Box
        self.txt = tk.Label(self.root,text = '0%',bg = '#345',fg = '#fff')
        self.txt.place(relx=0.57,rely=0.52)
        #Down_box_tilte
        self.box_title = tk.LabelFrame(self.root, text="choose file and folder")
        self.box_title.place(height=200, width=450, rely=0.6, relx=0)
        #Buttons
        #file button
        self.file_button = tk.Button(self.box_title, text="Browse A file",background="#93c4fc",command=lambda: self.File_dialog())
        self.file_button.place(rely=0.8,relx=0.5)

        #directory button
        self.folder_button = tk.Button(self.box_title, text = "chose directory",background="#93c4fc",command=lambda: self.Folder_dialog())
        self.folder_button.place(rely=0.8,relx=0.7)
        
        #load_excel_button
        self.load_excel_button = tk.Button(self.box_title,bg="#7bfdc5", text="Load File",command=lambda: self.Load_excel_data())
        self.load_excel_button.place(rely=0.8,relx=0.3)

        self.button4 = tk.Button(root, text="Calculate",background="#8ed49f",command=lambda:self.Testing_model())
        self.button4.config(width=20,height=3)
        self.button4.place(relx=0.8,rely=0.5)


        self.Label_file = tk.Label(self.box_title, text="No File Selected")
        self.Label_file.place(rely=0, relx=0)

        self.Label_folder = tk.Label(self.box_title, text="No such Folder selected")
        self.Label_folder.place(rely=0.1,relx=0)

        self.tv1 = ttk.Treeview(self.Excel_Box)
        self.tv1.place(relheight=1, relwidth=1)

        self.treeScrolly = tk.Scrollbar(self.Excel_Box, orient="vertical", command=self.tv1.yview)
        self.treeScrollx = tk.Scrollbar(self.Excel_Box, orient="horizontal", command=self.tv1.xview)

        self.tv1.configure(xscrollcommand=self.treeScrollx, yscrollcommand=self.treeScrolly)
        self.treeScrolly.pack(side="right",fill="y")
        self.treeScrollx.pack(side="bottom",fill="x")
        self.excel_path=""
        self.data_path = ""
        self.isrunning = False
        self.queue =Queue()
        self.size =100
        self.queue2 = Queue()
    

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
                self.txt["text"] = ((out/self.size))*100
                self.Progress_Bar["value"] = ((out/self.size))*100
        
            if self.queue2:
                out1 = self.queue2.get()
                self.tv1.insert("","end",values=out1)
        
        return


    def work(self):
        print(" in test")
        self.clear_data()
        self.tv1["column"] = ["first","second","pfirst","pfs","psecond"]
        self.tv1["show"] = "headings"
        for colum in self.tv1["column"]:
            self.tv1.heading(colum, text = colum)
 
        p = multiprocessing.Process(target= GUI_triplet.testing_excel,args=(self.excel_path ,self.data_path, self.queue,self.queue2))
        p.start()
        q = threading.Thread(target=self.keyy )
        q.start()
        print("in work")

        p.join()

        self.isrunning =False
        return p
        

    def Testing_model(self):
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
        # t_worker.join()
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
    root = tk.Tk()
    MainGUI(root)
    root.mainloop()