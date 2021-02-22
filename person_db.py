#!/usr/bin/env python3

import os
import cv2
import imutils
import shutil
import face_recognition
import numpy as np
import time
import pickle


class Face():
    key = "face_encoding"

    def __init__(self, filename, image, face_encoding):
        self.filename = filename
        self.image = image
        self.encoding = face_encoding
        
        self.time = time.localtime()


    @classmethod
    def get_encoding(cls, image):
        rgb = image[:, :, ::-1]
        boxes = face_recognition.face_locations(rgb, model="hog")
        if not boxes:
            height, width, channels = image.shape
            top = int(height/3)
            bottom = int(top*2) 
            left = int(width/3)
            right = int(left*2)
            box = (top, right, bottom, left)
        else:
            box = boxes[0]
        return face_recognition.face_encodings(image, [box])[0]


class Person():
    _last_id = 0

    def __init__(self, name=None):
        if name is None:
            Person._last_id += 1
            self.name = "person_%02d" % Person._last_id
        else:
            self.name = name
            if name.startswith("person_") and name[7:].isdigit():
                id = int(name[7:])
                if id > Person._last_id:
                    Person._last_id = id
        self.encoding = None
        self.faces = []
        self.count = 0

    def add_face(self, face):
        self.faces.append(face)

    def calculate_average_encoding(self):
        if len(self.faces) == 0:
            self.encoding = None
        else:
            encodings = [face.encoding for face in self.faces]
            self.encoding = np.average(encodings, axis=0)

    def distance_statistics(self):
        encodings = [face.encoding for face in self.faces]
        distances = face_recognition.face_distance(encodings, self.encoding)
        return min(distances), np.mean(distances), max(distances)

    def last_time(self):
        return self.faces[-1].time.tm_min + 0.01 * self.faces[-1].time.tm_sec

    @classmethod
    def load(cls, pathname, face_encodings):
        basename = os.path.basename(pathname)
        person = Person(basename)
        for face_filename in os.listdir(pathname):
            face_pathname = os.path.join(pathname, face_filename)
            image = cv2.imread(face_pathname)
            if image.size == 0:
                continue
            if face_filename in face_encodings:
                face_encoding = face_encodings[face_filename]
            else:
                print(pathname, face_filename, "calculate encoding")
                face_encoding = Face.get_encoding(image)
            if face_encoding is None:
                print(pathname, face_filename, "drop face")

        person.calculate_average_encoding()
        return person

    def set_encoding(self, face_encoding):
        self.encoding = face_encoding

class PersonDB():
    def __init__(self):
        self.persons = []
        self.knowns = []
        self.unknown_dir = "unknowns"
        self.encoding_file = "face_encodings"
        self.unknown = Person(self.unknown_dir)

    def load_db(self, dir_name):
        if not os.path.isdir(dir_name):
            return
        print("Start loading persons in the directory '%s'" % dir_name)
        start_time = time.time()

        # read face_encodings
        pathname = os.path.join(dir_name, self.encoding_file)
        try:
            with open(pathname, "rb") as f:
                face_encodings = pickle.load(f)

                print(len(face_encodings), "face_encodings in", pathname)
        except:
            face_encodings = {}

        # read persons
        for key in face_encodings:
            person = Person(key)
            person.set_encoding(face_encodings[key])
            self.persons.append(person)
        
        # load knowns
        knownsdir = 'knowns'
        files = os.listdir(knownsdir)
        for filename in files:
            name, ext = os.path.splitext(filename)
            if ext == '.jpg' or ext == '.png':
                person = Person(name)
                pathname = os.path.join(knownsdir, filename)
                img = face_recognition.load_image_file(pathname)
                face_encodings = face_recognition.face_encodings(img)[0]
                person.set_encoding(face_encodings)
                self.knowns.append(person)

        print("Successfully loading knowns in", knownsdir)

        elapsed_time = time.time() - start_time
        print("Loading persons finished in %.3f sec." % elapsed_time)
        

    def save_encodings(self, dir_name):
        face_encodings = {}
        for person in self.persons:
            face_encodings[person.name] = person.encoding
        '''
        for face in self.unknown.faces:
            face_encodings[face.filename] = face.encoding
        '''
        pathname = os.path.join(dir_name, self.encoding_file)
        with open(pathname, "wb") as f:
            pickle.dump(face_encodings, f)
        print(pathname, "saved")
  

    def save_results(self, start_hour):
        # save using txt
        print('start save_results')
        persons = sorted(self.persons, key=lambda obj : obj.name)
        knowns = sorted(self.knowns, key=lambda obj : obj.name)

        now = time.localtime()
        s = "%04d.%02d.%02d.%02d-%02d.txt" % (now.tm_year, now.tm_mon, now.tm_mday, start_hour, now.tm_hour)
        f = open(s, 'w')

        total_counts = 0
        total_visitors = 0

        for known in knowns:
            data = "{} : {}\n".format(known.name, known.count)
            f.write(data)
            known.count = 0
        for person in persons:
            if person.count > 0:
                total_visitors += 1
                total_counts += person.count
            data = "{:10} : {}\n".format(person.name, person.count)
            f.write(data)
            person.count = 0
        f.write("total number of unknown faces : {}\n". format(total_counts))
        f.write("total number of visitors : {}". format(total_visitors))
        f.close()


    def save_db(self, dir_name):
        print("Start saving persons in the directory '%s'" % dir_name)
        start_time = time.time()
        try:
            shutil.rmtree(dir_name)
        except OSError as e:
            pass
        os.mkdir(dir_name)

        self.save_encodings(dir_name)

        elapsed_time = time.time() - start_time
        print("Saving persons finished in %.3f sec." % elapsed_time)


    def __repr__(self):
        s = "%d persons" % len(self.persons)
        #num_known_faces = sum(len(person.faces) for person in self.persons)
        #s += ", %d known faces" % num_known_faces
        #s += ", %d unknown faces" % len(self.unknown.faces)
        return s


    def print_persons(self):
        print(self)
        persons = sorted(self.persons, key=lambda obj : obj.name)
        knowns = sorted(self.knowns, key=lambda obj : obj.name)

        for known in knowns:
            s = "{:10} ->".format(known.name)
            s += " %d faces" % (known.count)
            print(s)

        for person in persons:
            s = "{:10} ->".format(person.name)
            s += " %d faces" % (person.count)
            print(s)


if __name__ == '__main__':
    dir_name = "result"
    pdb = PersonDB()
    pdb.load_db(dir_name)
    pdb.print_persons()
    pdb.save_encodings(dir_name)