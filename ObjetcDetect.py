import tkinter as tk
from tkinter import filedialog as fd
from tkinter.ttk import *
import tkinter.font as font
from PIL import ImageTk, Image
import cv2

root = tk.Tk()  # definir globale frame
root.iconbitmap("logo.ico")  # add logo
root.title('Object Detection')  # add title
root.resizable(False, False)  # set resizable false


def select_file():  # upload image
    filetypes = (
        [('image files', '*.jpg *.png *.jpeg *.tif')]  # type d'image
    )
    filename = fd.askopenfilename(  # get path
        title='Select Image',
        initialdir='/',
        filetypes=filetypes)
    global path
    path = filename
    for widget in frame.winfo_children():  # supprimer l'ancienne image
        widget.destroy()
    imageBefore = Image.open(filename)
    img = imageBefore.resize((560, 560))
    imageFinale = ImageTk.PhotoImage(img)
    label = Label(frame, image=imageFinale)
    label.image = imageFinale
    label.pack()


def detectObject():  # detect object
    img = cv2.imread(path)  # lire image
    classNames = []  # array pour stocker les classe
    classFile = 'objects.names'

    with open(classFile, 'r') as f:  # stocker les class coco dans le array
        classNames = f.read().split('\n')

    # fichier est un modèle pré-entraîné sur l'ensemble de données COCO
    # coco c'est un ensemble de données d'image vous pouvez utiliser ses données pour entainer des modeles d'apprentissage et de reconaissance des objets
    # text format
    configPath = 'ssd_mobilenet_v3.pbtxt'
    # c'est graphe pré entraine qui ne peut plus étrenne
    # binary format
    weightpath = 'frozen_inference_graph.pb'

    # on va creer une network pour detecter les objets
    net = cv2.dnn_DetectionModel(weightpath, configPath)
    net.setInputSize(320, 230)
    net.setInputScale(1.0 / 127.5)
    # Définir la valeur moyenne pour le cadre.
    net.setInputMean((127.5, 127.5, 127.5))
    net.setInputSwapRB(True)

    # detect objet get id et confidence et posistion de chaque objet
    classIds, confs, bbox = net.detect(img, confThreshold=0.5)
    # boucler sur les données en meme temps fait la boucle dans les trois array avec zip
    for classId, confidence, box in zip(classIds, confs, bbox):
        cv2.rectangle(img, box, color=(153, 102, 102),
                      thickness=1)  # tracer le cadre
        cv2.putText(img, classNames[classId-1] + " : " + str('{:.2f}%'.format(confidence * 100)), (box[0] + 10, box[1] + 20),
                    cv2.QT_FONT_NORMAL, 0.8, (0, 180, 0), thickness=1)  # add text

    cv2.imshow('Output', img)  # show image
    cv2.waitKey(0)


canvas = tk.Canvas(root, height=700, width=700)  # le conteneur des objet
canvas.pack()

buttonFont = font.Font(family='Helvetica', size=13,
                       weight='bold')  # definir style button

ButtonUpload = tk.Button(root, text="Upload Image",
                         bg='#3366ff', fg='white', font=buttonFont, command=select_file)  # creer button
ButtonUpload.place(x=292, y=10)


frame = tk.Frame(root, highlightbackground="#3366ff",
                 highlightthickness=2)  # definir cadre pour l'image
frame.place(relwidth=0.8, relheight=0.8, relx=0.1, rely=0.1)


ButtonDetect = tk.Button(root, text="Detect Object",
                         bg='#3366ff', fg='white', font=buttonFont, command=detectObject)  # creer button
ButtonDetect.place(x=292, y=650)

root.mainloop()  # lancer le frame
