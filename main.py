import os
import random
import timeit
import cv2
import face_recognition as fr
import numpy as np
import time


def get_face_name(face_path):
    image_names = os.listdir(face_path)
    image_names = [image_name.replace('.jpg', '') for image_name in image_names]
    return image_names


def get_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def get_random_color():
    return random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)


def get_face_encoding(image):
    return fr.face_encodings(image)[0]


def main(use_web_cam=True, face_ref_path=None, test_path=None):
    if face_ref_path is None:
        face_ref_path = os.path.join('data', 'faces')
    faces = get_face_name(face_ref_path)
    images = [get_image(os.path.join(face_ref_path, image_path)) for image_path in os.listdir(face_ref_path)]
    start_time = timeit.default_timer()
    face_encodings_gt = [get_face_encoding(image) for image in images]
    print(f'Encoding completed in {round(timeit.default_timer() - start_time, 3)}s')
    if use_web_cam:
        cap = cv2.VideoCapture(0)
        while True:
            success, img = cap.read()
            img_s = cv2.resize(img, (0, 0), None, 0.25, 0.25)
            img_s = cv2.cvtColor(img_s, cv2.COLOR_BGR2RGB)
            faces_loc_in_frame = fr.face_locations(img_s)
            face_encodings_in_frame = fr.face_encodings(img_s, known_face_locations=faces_loc_in_frame)
            for face_encoding, face_loc in zip(face_encodings_in_frame, faces_loc_in_frame):
                matches = fr.compare_faces(face_encodings_gt, face_encoding)
                face_distance = fr.face_distance(face_encodings_gt, face_encoding)
                match_index = np.argmin(face_distance)
                if matches[match_index]:
                    name = faces[match_index].upper()
                    y1, x2, y2, x1 = map(lambda x: x * 4, face_loc)
                    color = get_random_color()
                    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                    cv2.rectangle(img, (x1, y2 - 35), (x2, y2), color, cv2.FILLED)
                    cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
            cv2.imshow('web_cam', img)
            cv2.waitKey(1)
    else:
        print('Loading files from disc')
        if test_path is None:
            test_path = os.path.join('data', 'test')
        for test_image_path in os.listdir(test_path):
            test_image = get_image(os.path.join(test_path, test_image_path))
            faces_loc_in_frame = fr.face_locations(test_image)
            face_encodings_in_frame = fr.face_encodings(test_image, known_face_locations=faces_loc_in_frame)
            for face_encoding, face_loc in zip(face_encodings_in_frame, faces_loc_in_frame):
                matches = fr.compare_faces(face_encodings_gt, face_encoding)
                face_distance = fr.face_distance(face_encodings_gt, face_encoding)
                match_index = np.argmin(face_distance)
                if matches[match_index]:
                    name = faces[match_index].upper()
                    y1, x2, y2, x1 = face_loc
                    color = get_random_color()
                    cv2.rectangle(test_image, (x1, y1), (x2, y2), color, 2)
                    cv2.rectangle(test_image, (x1, y2 - 35), (x2, y2), color, cv2.FILLED)
                    cv2.putText(test_image, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
            cv2.imshow('test_image', cv2.cvtColor(test_image, cv2.COLOR_RGB2BGR))
            cv2.waitKey(1)
            time.sleep(1)

    print()


if __name__ == '__main__':
    main(use_web_cam=False)
