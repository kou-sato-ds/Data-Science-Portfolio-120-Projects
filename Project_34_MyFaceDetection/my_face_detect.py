# [1] Import: 必要なライブラリ
import cv2
import sys

# [2] Read (Load): モデルの読み込み
cascPath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)

# [3] Features (Setup): カメラ設定
video_capture = cv2.VideoCapture(0)

# [4] Split/Process: リアルタイム処理ループ
while True:
    ret, frame = video_capture.read()
    # 処理しやすいようにグレースケール化
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # [5] Fit (Detect): 顔検知の実行
    faces = faceCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    # [6] Predict (Output): 結果の描画
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.imshow('Video', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()