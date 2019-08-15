:main
set /P var="Photo number "
python recognize.py --detector face_detection_model --image images/%var%.jpg -m open.t7 --recognizer output/recognizer.pickle --le output/le.pickle
goto main
