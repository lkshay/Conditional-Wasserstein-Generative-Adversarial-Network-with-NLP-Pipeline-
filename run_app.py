from tkinter import *
import tkinter as tk
from subprocess import Popen, PIPE
# import ImageTk

i = 0

def draw_it():
	global i,file
	text = e1.get()
	print(text)

	process = Popen(['./demo.sh', text], stdout=PIPE, stderr=PIPE)
	process.wait()


	file = "results.png".format(i)

	image = tk.PhotoImage(file=file)
	# label = tk.Label(image=image)
	# label.grid(row=4)

	# img2 = ImageTk.PhotoImage(Image.open(file))
	panel.configure(image=image)
	panel.image = image

	i = i+1
	if i==10:
		i=0



if __name__=='__main__':
	master = Tk()
	Label(master, text="What do you wanna see today? :").grid(row=0, pady=10, padx=7)

	e1 = Entry(master)

	e1.grid(row=0, column=1, padx=10)

	Button(master, text='Draw it for me!', command=draw_it).grid(row=2, column=1, sticky=E, pady=10, padx=10)
	Button(master, text='I\'m done for the day', command=master.quit).grid(row=2, column=0, sticky=W, pady=10, padx=10)

	panel = tk.Label(master, image=None)
	# panel.pack(side="bottom", fill="both", expand="yes")
	panel.grid(row=3, column=0, padx = 10, pady=20)

	master.mainloop()