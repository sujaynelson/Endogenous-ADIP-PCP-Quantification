import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

class Disregard(tk.Tk):
    def __init__(self, image_feeder):
        super().__init__()
        self.title('Disregard')
        self.geometry('400x300')

        self.image_feeder = image_feeder

        self.all_ids = self.image_feeder.get_all_ids()

        self.current_id_index = 0

        self.should_be_disregarded = [False for _ in range(len(self.all_ids))]

        self.main_frame = tk.Frame(self)
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        # Make controls frame
        self.controls_frame = tk.Frame(self.main_frame)
        self.controls_frame.pack(side=tk.TOP, fill=tk.X)

        self.label = tk.Label(self.controls_frame, text='Number 0')
        self.label.pack(side=tk.LEFT)

        self.button = tk.Button(self.controls_frame, text='Disregard', command=self.disregard)
        self.button.pack(side=tk.LEFT)

        self.next_button = tk.Button(self.controls_frame, text='Next', command=self.next)
        self.next_button.pack(side=tk.RIGHT)

        # Make image frame
        self.image_frame = tk.Frame(self.main_frame)
        self.image_frame.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)

        # Add first figure from image_feeder
        self.figure = plt.Figure(figsize=(5, 5), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.image_frame)
        self.canvas.draw()
        # Add to image frame
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.start()


    def next(self):
        if not self.current_id_index + 1 == len(self.all_ids):
            self.current_id_index += 1

        print(self.current_id_index)
        
        # update label
        self.label.config(text=f'Number {self.current_id_index}')

        pilimg = self.image_feeder.get_img_from_id(self.all_ids[self.current_id_index])
        self.figure.clear()
        plt.imshow(pilimg)
        self.canvas.draw()

        



    def disregard(self):
        self.should_be_disregarded[self.current_id_index] = True


    def start(self):
        self.mainloop()