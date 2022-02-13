# object_detection
画像をアップロードすると、物体検出された(バウンディボックス描画された)が画像が返ってくる

## 開発環境

- Ubuntu 20.04LTS on WSL
- Python 3.7

## インストール

```
# pipを最新に
pip install --upgrade pip

# ライブラリのインストール
pip install -r requirements.txt

# TensorFlow Object Detection APIのインストール
git clone --depth 1 https://github.com/tensorflow/models
cd models/research/
protoc object_detection/protos/*.proto --python_out=.
cp object_detection/packages/tf2/setup.py .
python -m pip install .

# モデルのインストール
cd ../..
wget http://download.tensorflow.org/models/object_detection/tf2/20200713/centernet_hg104_512x512_coco17_tpu-8.tar.gz
tar -xf centernet_hg104_512x512_coco17_tpu-8.tar.gz
rm centernet_hg104_512x512_coco17_tpu-8.tar.gz
```

## 実行

```
./app.py
```

## 使用したシステムの解説

ウェブブラウザで http://localhost:8000/ にアクセスすると表示される画面に入力された画像に対して、TensorFlow Object Detection API により物体検出モデルでの推論を行い、http://localhost:8000/uploads/[ファイル名] に保存する

上記URLにリダイレクトされることにより、物体検出された(バウンディボックス描画された)が画像が返ってくる画面が表示される

## 参考にしたサイト
1. [Flaskで画像ファイルをアップロード](https://qiita.com/keimoriyama/items/7c935c91e95d857714fb)
2. [TensorFlow Object Detection API のつかいかた（推論。Colabサンプル付き）](https://qiita.com/john-rocky/items/7d9176d16617c66fb3f1)