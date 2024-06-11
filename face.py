import cv2
import face_recognition
import os
import glob
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk
from tkinter import messagebox

class SimpleFacerec:
    def __init__(self):
        self.known_face_encodings = []
        self.known_face_names = []
        self.frame_resizing = 0.5

    def load_encoding_images(self, images_path):
        images_path = glob.glob(os.path.join(images_path, "*.*"))
        print("{} encoding images found.".format(len(images_path)))

        for img_path in images_path:
            img = face_recognition.load_image_file(img_path)
            img_encoding = face_recognition.face_encodings(img)[0]

            basename = os.path.basename(img_path)
            (filename, ext) = os.path.splitext(basename)
            self.known_face_encodings.append(img_encoding)
            self.known_face_names.append(filename)
        print("Encoding images loaded")

    def detect_known_faces(self, frame):
        small_frame = cv2.resize(frame, (0, 0), fx=self.frame_resizing, fy=self.frame_resizing)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
            name = "Unknown"

            face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = self.known_face_names[best_match_index]
            face_names.append(name)

        face_locations = np.array(face_locations)
        face_locations = face_locations / self.frame_resizing
        return face_locations.astype(int), face_names


class App:
    def __init__(self, root):
        self.root = root
        self.sfr = SimpleFacerec()
        self.video_capture = None
        self.canvas = None
        self.start_button = None
        self.capture_button = None
        self.stop_button = None

        self.sfr.load_encoding_images(r"D:\Face recognition\images")

        self.setup_ui()

    def setup_ui(self):
        self.root.title("Face Recognition")
        self.root.geometry("800x600")

        self.canvas = tk.Canvas(self.root, width=640, height=480)
        self.canvas.pack()

        self.start_button = tk.Button(self.root, text="Start", command=self.start_video)
        self.start_button.pack(side=tk.LEFT, padx=10)

        self.capture_button = tk.Button(self.root, text="Capture", command=self.capture_image)
        self.capture_button.pack(side=tk.LEFT, padx=10)

        self.stop_button = tk.Button(self.root, text="Stop", command=self.stop_video)
        self.stop_button.pack(side=tk.LEFT, padx=10)

    def start_video(self):
        self.video_capture = cv2.VideoCapture(0)
        self.show_frame()

    def show_frame(self):
        ret, frame = self.video_capture.read()

        # Detect Faces
        face_locations, face_names = self.sfr.detect_known_faces(frame)
        for face_loc, name in zip(face_locations, face_names):
            y1, x2, y2, x1 = face_loc[0], face_loc[1], face_loc[2], face_loc[3]

            cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 200), 2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 200), 4)

        # Display the frame in the UI
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        image = ImageTk.PhotoImage(image)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=image)
        self.canvas.image = image

        self.root.after(15, self.show_frame)

    def capture_image(self):
        if self.video_capture is not None:
            ret, frame = self.video_capture.read()
            if ret:
                # Detect Faces
                face_locations, face_names = self.sfr.detect_known_faces(frame)
                for face_loc, name in zip(face_locations, face_names):
                    y1, x2, y2, x1 = face_loc[0], face_loc[1], face_loc[2], face_loc[3]
                    cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 200), 2)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 200), 4)

                cv2.imwrite("captured_image.png", frame)
                messagebox.showinfo("Image Captured", "Image saved as captured_image.png")
                print("Image captured.")

    def stop_video(self):
        if self.video_capture is not None:
            self.video_capture.release()
            self.video_capture = None


if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()
