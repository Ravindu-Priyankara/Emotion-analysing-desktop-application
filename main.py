#! /usr/bin/python3

import os
import time
import tkinter as tk
from tkinter import *
from PIL import Image, ImageTk
import tensorflow as tf
from tensorflow import keras
import numpy as np
import cv2
from keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
import mysql.connector
import customtkinter 
import tensorflow as tf
from tensorflow import keras
from keras.models import load_model
from datetime import date
import pandas
import plotly.express as px
import matplotlib.pyplot as plt
import random
import datetime


class Splash(tk.Toplevel):
    def __init__(self,parent):
        tk.Toplevel.__init__(self,parent)

        #inside splash screen components
        self.geometry("800x600")
        self.title("Splash Screen")

        #image resize aand add to window
        self.image = Image.open("img/test.jpg")
        self.img = self.image.resize((800,600))
        self.my_img = ImageTk.PhotoImage(self.img)

        #set image to label
        self.label = Label(self,image = self.my_img)
        self.label.pack()

        self.update()



class Face_id(tk.Toplevel):
    def __init__(self,parent):
        tk.Toplevel.__init__(self,parent)

        cap = cv2.VideoCapture(1)
        recognizer = cv2.face.LBPHFaceRecognizer_create()

        cascadePath = "models/haarcascade_frontalface_default.xml"
        faceCascade = cv2.CascadeClassifier(cascadePath);
        font = cv2.FONT_HERSHEY_SIMPLEX

        path = 'trained/'

        for i in os.listdir(path):
            recognizer.read(path + i)
            print(i)

        def close():
            cap.release()
            cv2.destroyAllWindows()
            self.destroy()
            logged()

        while True:
            flag, im = cap.read()
            gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            faces = faceCascade.detectMultiScale(gray, 1.2, 5)

            for (x, y, w, h) in faces:
                cv2.circle(im, (x+int(w/2), y+int(h/2)), 100, (0, 0, 255), 0)
                Id, confidence = recognizer.predict(gray[y:y + h, x:x + w])

                Id = "User {0:.2f}%".format(round(100 - confidence, 2))

                if float("{0:.2f}".format(round(100 - confidence, 2))) > 50:
                    cv2.putText(im, str(Id)+'\tUser Matched', (x, y - 40), font, 1, (255, 255, 255), 3)
                    break
                else:
                    cv2.putText(im, "Not found!", (0, 30), font, 1, (255, 0, 0), 2)

            cv2.imshow('im', im)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

            close()

        cap.release()
        cv2.destroyAllWindows()

class Face_Recog(tk.Toplevel):
    def __init__(self,parent):
        tk.Toplevel.__init__(self,parent)

        global count
        count = 0
        cap = cv2.VideoCapture(0)
        global button_path

        def detect_face(img):
            face_cascade = cv2.CascadeClassifier('models/haarcascade_frontalface_default.xml')
            faces = face_cascade.detectMultiScale(img, 1.2, 5)

            if faces == ():
                return False
            else:
                for (x, y, w, h) in faces:
                    img = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

        def take_pic():
            global count

            face_cascade = cv2.CascadeClassifier('models/haarcascade_frontalface_default.xml')

            while (True):
                ret, frame = cap.read()
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.2, 5)

                if faces == ():
                    label1.config(text="Can't identify your face. Please get closer and make sure there is red rectangular araund your face.")
                    return False
                else:
                    for (x, y, w, h) in faces:

                        crop_img = gray[y:y+h, x:x+w]
                        count += 1
                        cv2.imwrite(newpath+"/User_" + str(id_input) + '.' + str(count) + ".jpg", crop_img)
                        print(count)

                time.sleep(0)
                if count == 50:
                    label1.config(text="Please smile and capture.")
                    count += 1
                    return False
                elif count == 100:
                    label1.config(text="Please take of glasses if it is exist and capture.")
                    count += 1
                    return False
                elif count == 150:
                    label1.config(text="Please try to draw circle with your head")
                    count += 1
                    return False
                elif count > 200:
                    label1.config(text="Done! You can close the program.")
                    train_faces()
                    self.destroy()
                    return False

        def train_faces():
            recognizer = cv2.face.LBPHFaceRecognizer_create()
            detector = cv2.CascadeClassifier('models/haarcascade_frontalface_default.xml')

            get_imagePath = [os.path.join(newpath,f) for f in os.listdir(newpath)]
            faceSamples = []
            ids = []

            for imagePath in get_imagePath:

                PIL_img = Image.open(imagePath).convert('L')
                img_numpy = np.array(PIL_img,'uint8')
                id = int(os.path.split(imagePath)[-1].split(".")[1])
                faces = detector.detectMultiScale(img_numpy)

                for (x,y,w,h) in faces:

                    faceSamples.append(img_numpy[y:y+h,x:x+w])

                    ids.append(id)

            recognizer.train(faceSamples, np.array(ids))
            recognizer.save('trained/'+id_input+'.yml')

        def show_frame():
            ret, frame = cap.read()
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            detect_face(rgb)

            prevImg = Image.fromarray(rgb)
            imgtk = ImageTk.PhotoImage(image=prevImg)
            lmain.imgtk = imgtk
            lmain.configure(image=imgtk)
            lmain.after(10, show_frame)

        def create_path():
            global newpath, id_input, label1, button

            if not os.path.exists('trained/') and not os.path.exists('face_data/'):
                os.makedirs('trained/')
                os.makedirs('face_data/')

            button_path.pack_forget()
            label1 = tk.Label(self, text="Please capture when you ready!", relief=tk.GROOVE)
            button = tk.Button(self, text="Capture", command=take_pic)
            label1.pack(side="left")
            button.pack(side="right")
            id_input = id_path.get()              #input
            newpath = r'face_data/' + id_input    #face_data/input
            if not os.path.exists(newpath):
                os.makedirs(newpath)

        self.title = ("Camera Capture")
        self.resizable(width=False, height=False)
        self.bind('<Escape>', lambda e: self.quit())

        id_path = tk.StringVar()

        lmain = Label(self, compound=tk.CENTER, anchor=tk.CENTER, relief=tk.RAISED)
        button_path = Button(self, text="Get new id", command=create_path)
        entry = Entry(self, textvariable = id_path)

        entry.pack()
        button_path.pack()
        lmain.pack()

        show_frame()
        self.mainloop()

class Sign_up(tk.Toplevel):
    def __init__(self,parent):
        tk.Toplevel.__init__(self,parent)

        mycursor = mydb.cursor()

        #inside main screen components
        self.title("Login")
        self.geometry("800x650")
        self.resizable(False,False)

        #image resize and add to windows
        self.image = Image.open("img/login.jpg")
        self.img = self.image.resize(((800,650)))
        self.my_img = ImageTk.PhotoImage(self.img)

        #place the label for image placing
        self.login_img_label = Label(self,image = self.my_img)
        self.login_img_label.pack() 

        #label for login 
        self.login_msg_label = Label(self,text = "Signup Panel",font =10, bg = "green2", foreground='black')
        self.login_msg_label.place(x=350,y = 150)

        #user name and password labels
        self.username_label = Label(self,text = "Username",font = 50, bg= "magenta2")
        self.username_label.place(x=220,y = 300)

        self.password_label = Label(self,text = "Password",font = 50,bg = "magenta2")
        self.password_label.place(x=220,y = 360)

        self.password_label = Label(self,text = "Retype Password",font = 50,bg = "magenta2")
        self.password_label.place(x=200,y = 420)

        #variables for username and password

        Username = StringVar()
        Password = StringVar()
        
        #add entry for gather password and username
        
        self.userEntry = Entry(self, width = 20)
        self.userEntry.place(x=325,y= 298)

        self.passEntry = Entry(self, show = "*", width = 20)
        self.passEntry.place(x=325,y =358)

        self.repassEntry = Entry(self, show = "*", width = 20)
        self.repassEntry.place(x=325,y =418)

        def face_recog():
            face = Face_Recog(self)
        
        

        #set data to variables
        def set_data():

            Username = self.userEntry.get()
            Password = self.passEntry.get()
            retypePassword = self.repassEntry.get()

            self.login_notifi = Label(self,text = "",font = 50,bg = "magenta2")
            self.login_notifi.place(x=320,y = 120)

            if Password == retypePassword:
                id = random.randint(0,1000)
                sql = "INSERT INTO login (id, username, password) VALUES (%s, %s, %s)"
                val = (int(id),Username, Password)
                mycursor.execute(sql, val)
                mydb.commit()
                print(mycursor.rowcount,"record inserted.")
                self.login_notifi.config(text = "SignUp Success!")
                face_recog()
                self.destroy()
            else:
                self.login_notifi.config(text = "SignUp Failed!")
                


        #add login button to this windows
        self.login_button = Button(self,text = "Sign_up",command = set_data)
        self.login_button.place(x=350,y= 478)
        

        #bind a keyevent for exit the window
        self.bind('esc',lambda x: self.destroy())
        
class login(tk.Toplevel):
    def __init__(self,parent):
        tk.Toplevel.__init__(self,parent)

        mycursor = mydb.cursor()

        #inside main screen components
        self.title("Login")
        self.geometry("800x650")
        self.resizable(False,False)

        #image resize and add to windows
        self.image = Image.open("img/login.jpg")
        self.img = self.image.resize(((800,650)))
        self.my_img = ImageTk.PhotoImage(self.img)

        #place the label for image placing
        self.login_img_label = Label(self,image = self.my_img)
        self.login_img_label.pack() 

        #label for login 
        self.login_msg_label = Label(self,text = "Login Panel",font =10, bg = "green2", foreground='black')
        self.login_msg_label.place(x=350,y = 150)

        #user name and password labels
        self.username_label = Label(self,text = "Username",font = 50, bg= "magenta2")
        self.username_label.place(x=220,y = 300)

        self.password_label = Label(self,text = "Password",font = 50,bg = "magenta2")
        self.password_label.place(x=220,y = 360)


        #variables for username and password

        Username = StringVar()
        Password = StringVar()
        
        #add entry for gather password and username
        
        self.userEntry = Entry(self, width = 20)
        self.userEntry.place(x=325,y= 298)

        self.passEntry = Entry(self, show = "*", width = 20)
        self.passEntry.place(x=325,y =358)

        def signup():
            sign = Sign_up(self)
        
        def faceid():
            face = Face_id(self)

        #signup
        signup_button = customtkinter.CTkButton(master=self,text="Sign up" ,corner_radius=10, command=signup)
        signup_button.place(x=650,y= 10)

        #face id use 

        face_id_button = customtkinter.CTkButton(master=self,text="Face Id" ,corner_radius=10, command=faceid)
        face_id_button.place(x=500,y= 10)

        

        #set data to variables
        def set_data():

            Username = self.userEntry.get()
            Password = self.passEntry.get()

            self.login_notifi = Label(self,text = "",font = 50,bg = "magenta2")
            self.login_notifi.place(x=320,y = 120)

            if Username == '':
                self.login_notifi.config(text = "All fields are required!")

            else :
                mycursor.execute('SELECT * FROM login WHERE username = %s AND password = %s', (Username, Password,))
                account = mycursor.fetchone()
                if len(account) > 0:
                    self.login_notifi.config(text = "Login Success!")
                    logged()

                


        #add login button to this windows
        self.login_button = Button(self,text = "Login",command = set_data)
        self.login_button.place(x=350,y= 430)
        

        #bind a keyevent for exit the window
        self.bind('esc',lambda x: self.destroy())

class Main_Window(tk.Tk):
    def __init__(self):
        tk.Tk.__init__(self)

        global log_insert

        self.title("Emotion Detection")
        self.geometry('1400x750')
        self.resizable(False,False)

        frame_left = tk.Frame(self, width=240, height=750, bg="gray22", borderwidth=1, relief=tk.RIDGE)
        frame_left.place(x=0, y=0)

        frame_top = tk.Frame(self,width=1000,height = 100,bg="gray31",borderwidth=1, relief=tk.RIDGE)
        frame_top.place(x=240, y=0)

        frame_right = tk.Frame(self,width=160,height=750,bg="gray21",borderwidth=1, relief=tk.RIDGE)
        frame_right.place(x=1240,y=0)


        label_user_dashboard = customtkinter.CTkLabel(master=self,text="Sentimental Analysing User Dashboard",fg_color=( "red"),width=320,height=25,corner_radius=8)
        label_user_dashboard.place(x = 550,y=15)
        
        label_user_dashboard_slogan = customtkinter.CTkLabel(master=self,text="Unleash the power of emotions with our cutting-edge detection software.",fg_color=( "green2"),width=520,height=25,corner_radius=8)
        label_user_dashboard_slogan.place(x = 450,y=55)

        #get date
        today = date.today()

        #split data
        year = str(today.year)
        month = str(today.month)
        day = str(today.day)

        def monitor():
            os.system("python Emotion_Detection.py")
            log_insert("Start Monitoring.............................................")
        
        def run_command():
            input = terminal.get("1.0", "end-1c")
            out = os.popen(input)
            terminal_output.insert(END,out.read())
            terminal.delete("1.0", "end-1c")
            log_insert('Running command.............................../')

        def log_insert(data):
            log.insert(END,data)
        
        def clean_output():
            terminal_output.delete("1.0", "end-1c")
            log_insert("cleaning inputs................................")

        def det_info():
            x = np.array(['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Suprise'])
            y = np.array([0,1,2,3,4,5,6])

            plt.title("Emotions Data")
            plt.xlabel("Emotion")
            plt.ylabel("Value")

            plt.plot(x, y)

            plt.grid()

            plt.show()

        def analyse1():
            df = pandas.read_csv(year+'/'+month+'/'+day+'/'+str(today)+'.csv')

            fig = px.line(df, x = 'ProbabilityValue', y = 'Name', title='Today your emotions')
            fig.show()
            log_insert("showing analyse...........................................")

        def today_analysed():
            today = date.today()
            year = str(today.year)
            month = str(today.month)
            day = str(today.day)

            #emotions

            emotions = ['Sad','Happy','Disgust','Neutral','Angry','Surprise','Fear']

            sad = 0
            happy = 0
            disguest = 0
            neutral = 0
            angry = 0
            suprise = 0
            fear = 0

            df = pandas.read_csv(year+'/'+month+'/'+day+'/'+str(today)+'.csv')

            names = df['Name'].tolist()


            value = len(names)

            for x in range(value):
                if names[x] == emotions[0]:
                    sad += 1
                elif names[x] == emotions[1]:
                    happy += 1
                elif names[x] == emotions[2]:
                    disguest += 1
                elif names[x] == emotions[3]:
                    neutral += 1
                elif names[x] == emotions[4]:
                    angry += 1
                elif names[x] == emotions[5]:
                    suprise += 1
                elif names[x] == emotions[6]:
                    fear += 1

            y = [sad,happy,disguest,neutral,angry,suprise,fear]
            myexplode = [0.2, 0, 0, 0,0,0,0]
            plt.pie(y, labels = emotions, explode = myexplode, shadow = True)
            plt.show()

            log_insert('Showing informations............................................/')

        def update_Data():

            today = date.today()
            year = str(today.year)
            month = str(today.month)
            day = str(today.day)

            #emotions

            emotions = ['Sad','Happy','Disgust','Neutral','Angry','Surprise','Fear']

            sad = 0
            happy = 0
            disguest = 0
            neutral = 0
            angry = 0
            suprise = 0
            fear = 0

            df = pandas.read_csv(year+'/'+month+'/'+day+'/'+str(today)+'.csv')

            names = df['Name'].tolist()


            value = len(names)

            for x in range(value):
                if names[x] == emotions[0]:
                    sad += 1
                elif names[x] == emotions[1]:
                    happy += 1
                elif names[x] == emotions[2]:
                    disguest += 1
                elif names[x] == emotions[3]:
                    neutral += 1
                elif names[x] == emotions[4]:
                    angry += 1
                elif names[x] == emotions[5]:
                    suprise += 1
                elif names[x] == emotions[6]:
                    fear += 1

            id = random.randint(0,1000)
            mycursor = mydb.cursor()
            sql = "INSERT INTO data VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s)"
            val = (int(id),today,int(sad),int(angry),int(happy),int(disguest),int(neutral),int(fear),int(suprise))
            mycursor.execute(sql, val)

            mydb.commit()

            log_insert("data updated")

        def weekly_data():
            day = str(today.day)
            mycursor = mydb.cursor()

            for i in range (7):
                dates = (datetime.datetime.now() - datetime.timedelta(days=1)).date()
            
                mycursor.execute('SELECT * FROM data WHERE today = %s', (dates))
                DATA = mycursor.fetchone()
                log_insert(DATA)
        
        def monthly_data():
            month = today.month
            new_month = month -1

            mycursor = mydb.cursor()
            mycursor.execute('SELECT * FROM data WHERE today = %s', ('2023-'+str(new_month)+'-30'))
            DATA = mycursor.fetch()
            log_insert(DATA)

        
        #monitor button
        button = customtkinter.CTkButton(master=self,text="Start Monitoring" ,corner_radius=10, command=monitor)
        button.place(x=350,y=180)

        #log
        log = Text(self, height = 27,width = 142,bg = "grey1",fg="white")
        log.place(x=240,y=420)

        #shell text
        shell_text = customtkinter.CTkLabel(master=self,text="Terminal",fg_color=( "green2"),width=100,height=25,corner_radius=8)
        shell_text.place(x=50,y = 100)

        #shell output
        shell_text = customtkinter.CTkLabel(master=self,text="Terminal output",fg_color=( "green2"),width=150,height=25,corner_radius=8)
        shell_text.place(x=30,y = 400)

        #terminal
        terminal = Text(self, height = 10,width = 30,bg = "slate gray",fg="green2")
        terminal.place(x=0,y=135)

        #terminal output
        terminal_output = Text(self, height = 20,width = 30,bg = "slate gray",fg="green2")
        terminal_output.place(x=0,y=435)

        #terminal run
        terminal_run = customtkinter.CTkButton(master=self,text="run command" ,corner_radius=10, command=run_command)
        terminal_run.place(x=50,y=290)

        #terminal out clean
        terminal_out_clean = customtkinter.CTkButton(master=self,text="clean output" ,corner_radius=10, command=clean_output)
        terminal_out_clean.place(x=50,y=330)

        #testing data
        detection_info = customtkinter.CTkButton(master=self,text="detection info" ,corner_radius=10, command=det_info)
        detection_info.place(x=350,y=250)

        #analysed data
        analysed = customtkinter.CTkButton(master=self,text="analyse data" ,corner_radius=10, command=analyse1)
        analysed.place(x=350,y= 330)

        today_analyse = customtkinter.CTkButton(master=self,text="Today data" ,corner_radius=10, command=today_analysed)
        today_analyse.place(x=550,y= 250)

        weekly_analyse = customtkinter.CTkButton(master=self,text="weekly data" ,corner_radius=10, command=weekly_data)
        weekly_analyse.place(x=750,y= 190)

        monthly_analyse = customtkinter.CTkButton(master=self,text="monthly data" ,corner_radius=10, command=monthly_data)
        monthly_analyse.place(x=750,y= 310)

        update_data = customtkinter.CTkButton(master=self,text="update data" ,corner_radius=10, command=update_Data)
        update_data.place(x=950,y= 250)



        

class Main_App(tk.Tk):
    def __init__(self):
        tk.Tk.__init__(self)

        global logged,mydb

        mydb = mysql.connector.connect(
            host="localhost",
            user="root",
            password="rAvi628$",
            database="Sentiment"
        )

        self.withdraw()
        
        splash = Splash(self)

        #set time count to destroy splash screeen
        time.sleep(5)
        splash.destroy()
        

        login_and_signup = login(self)
        
        def logged():
            time.sleep(5)
            login_and_signup.destroy()

            Main_win = Main_Window()
        


if __name__ == "__main__":
    mainAppLogin = Main_App()
    mainAppLogin.mainloop()