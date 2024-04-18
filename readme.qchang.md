2019.8.14. Bugs 修正

## 追加点

- Epoch総数を減らさないように修正した。


2019.8.9. Bugs 修正

## 追加点

- MNISTセットを学習できるように修正した。
- tf.check_numeric関数中の'name'を修正　

2019.8.9. Chain-Job 機能追加Ver.

## 追加点

- 学習を再開する時に、オプション　'--config_file=params_XX.json'　を使用可能にした。
  - 'XX'は再開ジョブのインデックス(00, 01,02,..)。
  - params_0*.jsonファイルの中にすべてのハイパーパラメータが保存される。
  - 学習を再開するごとに、下記のパラメータが更新される
    - n_processed: total of trained images
    - n_images: the number of loaded images
    - n_epoch_processsed: 処理されたエポックの数
    - n_restore: 学習再開の回数　
    - restore_path: モデルファイルのパス
    - trainとtestログファイルのファイル名
    

使用例：

1. ゼロから学習する時、いつものどおりにコマンドを入力する

        mpiexec -n 2 python train.py --problem original --epochs 1000 --image_size 256 .......
      　　
2. 学習データ保存時に以下のファイルが生成される。
   - モデルファイル
   - 設定ファイル(params_01.json)
   - ログファイル(train_00.txt, test_00.txt)

3. 学習を再開する時は以下のコマンドを入力する。

        mpiexec -n 2 python train.py --config_file params_01.json
          
4. その後、モデルファイルとともに、params_02.jsonファイルが生成される。
