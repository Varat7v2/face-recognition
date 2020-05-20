import numpy as np
import pickle
import cv2
import glob
import FaceRecognition as FR

def serialize_dict(filename):
    mydict = dict()
    # Load the embeddings of known people or check if there are any new people need to add in the list
    for image in glob.glob('known_persons/*'):
        name = image.split('/')[-1].split('.')[0]
        image = FR.load_image_file(image)
        encoding = FR.face_encodings(image)[0]
        mydict[name] = encoding

    # Serialize dictionary in binary format
    with open(filename, 'wb') as handle:
        pickle.dump(mydict, handle, protocol=pickle.HIGHEST_PROTOCOL)

def main():
    cam = cv2.VideoCapture(0)
    frame_width = int(cam.get(3))
    frame_height = int(cam.get(4))

    name = input("Enter your name: ")

    while True:
        ret, frame = cam.read()
        frame_show = frame.copy()

        if ret == False:
            break

        cv2.putText(frame_show, 'Hi! {}, press q to save your image.'.format(name), 
            (10, 20), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 255, 0), 1)
        
        # Display the resulting image
        cv2.imshow('Camera', frame_show)

        # Hit 'q' on the keyboard to quit!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.imwrite('known_persons/'+name+'.jpg', frame)
            serialize_dict('data/known_persons.pkl')
            break

    # Release handle to the webcam
    cam.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()