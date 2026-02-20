# ==========================================
# Project 34: Real-Time Face Detection
# Goal: 実装力の証明 (黄金の6ステップ)
# ==========================================

# [1] Import
import cv2
import sys

# [2] Read (Load Model)
cascPath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)

# [3] Features (Setup Input)
video_capture = cv2.VideoCapture(0)

# [4] Split/Process (Looping through data)
while True:
    # データを1フレームずつ分割して読み込み
    ret, frame = video_capture.read()
    
    # 特徴抽出を容易にするためのグレースケール処理
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # [5] Fit (Detection)
    # モデルを現在のデータ(gray)に適合させ、顔の位置を特定
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    # [6] Predict/Output
    # 推論結果（座標）を元に、画面にフィードバック
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # 結果の表示
    cv2.imshow('Video', frame)

    # 終了判定
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 後片付け
video_capture.release()
cv2.destroyAllWindows()