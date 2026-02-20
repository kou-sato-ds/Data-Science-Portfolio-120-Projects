# Project 34: Real-time Face Detection

## 1. 概要
OpenCVを活用し、PCの内蔵カメラからリアルタイムで顔を検知するアプリケーションです。

## 2. 実装の型 (黄金の6ステップ)
本プロジェクトは、データサイエンス実装の標準的な流れに沿って構築しています。
1. **Import**: OpenCV (`cv2`) の導入
2. **Read**: 学習済みモデル (Haar Cascades) の読み込み
3. **Features**: カメラデバイスのセットアップとグレースケール変換
4. **Split/Process**: リアルタイムフレームの処理ループ
5. **Fit (Detect)**: `detectMultiScale` による顔検知の実行
6. **Predict/Output**: 検知結果の描画と画面出力

## 3. セットアップ
```bash
pip install opencv-python
python my_face_detect.py