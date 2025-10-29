import tkinter as tk
from tkinter import filedialog
from recorder import Recorder

r = Recorder()


def load_sound():
	global sound,sound_data
	for sound_display in frame.winfo_children():
		sound_display.destroy()
	
	sound_data = filedialog.askopenfilename(initialdir="/home/safat/", title="upload a sound", filetypes=(("all files", "*.*"),("mp3 files", "*.mp3")))
	print(sound_data)
	file_name = sound_data.split('/')
	panel = tk.Label(frame, text = str(file_name[-1]).upper()).pack()
	
#len(file_name)-1]


def play_loaded_sound():
	try:
		r.play(sound_data)
	except NameError:
		pass


switch = False

def record_audio():
		r.record(5, output='temp.wav')
		#switch = True
		 
def play_recorded_sound():
		#if switch == True:
		r.play('temp.wav')
		#else:
		#	pass

def classify_sound():
	pass



gui = tk.Tk()
gui.configure(background="light blue")
gui.title("Sound Classifier GUI")
gui.geometry("640x640")
gui.iconbitmap("@/home/safat/python_code/tkinterTry/clipping_sound.xbm")
gui.resizable(0, 0)

#tk.Label(gui, text = 'It\'s resizable').pack(side = tk.TOP, pady = 10)
title = tk.Label(gui, text="UrbanSound8K Classifier", padx=25, pady=6, font=("", 12)).pack()

canvas = tk.Canvas(gui, height=500, width=500, bg='grey', bd=3)
canvas.pack(side = tk.TOP)

button_rec = tk.Button(canvas, text='Record Audio for 5 seconds', fg='white', bg= 'green', command = record_audio)
button_rec.pack(side=tk.TOP)

play_record_button = tk.Button(gui, text = 'Play Recorded Sound', fg = 'white', bg = 'black', command = play_recorded_sound).pack(side=tk.BOTTOM)


frame = tk.Frame(gui, bg='white')
frame.place(relwidth=.8, relheight=0.8, relx=.1, rely=0.1)

b = tk.Button(gui,text='Choose Sound', fg='white', bg = 'black',command=load_sound)
b.pack(side=tk.LEFT)

classify_sound_button = tk.Button(gui,text="Classify Sound", fg="white", bg="grey", command=classify_sound)
classify_sound_button.pack(side=tk.RIGHT)

play_sound_button = tk.Button(gui,text="Play Loaded Sound", fg="white", bg="grey", command = play_loaded_sound)
play_sound_button.pack(side=tk.BOTTOM)


gui.mainloop()

