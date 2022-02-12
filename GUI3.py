from tkinter import *
from tkinter.ttk import Progressbar
import time


def step():
    for i in range(5):
        ws.update_idletasks()
        pb['value'] += 20
        time.sleep(1)
        txt['text']=pb['value'],'%'

ws = Tk()
ws.title('PythonGuides')
ws.geometry('200x150')
ws.config(bg='#345')


pb = Progressbar(
    ws,
    orient = HORIZONTAL,
    length = 200,
    mode = 'determinate'
    )

pb.place(x=40, y=20)

txt = Label(
    ws,
    text = '0%',
    bg = '#345',
    fg = '#fff'

)

txt.place(x=150 ,y=20 )

Button(
    ws,
    text='Start',
    command=step
).place(x=40, y=50)

ws.mainloop()