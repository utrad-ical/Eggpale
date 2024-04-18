
# 2019.7.29. Input is not invertible 回避Ver.

## 追加点

- オプション `--flow_permutation 3` を使用可能にした。`Input is not invertible`エラーを回避できるはず。**学習で使用した時は推論時にも必須なので注意。**
- 速度低下が見込まれるので注意。
- 演算の(モデル学習および推論の)結果が本質的に変わりうるのでそこも注意。



# 2019.5.21. モノクロ追加Ver.

## 追加点

- オプション `--image_color_num 1` を追加。これを入れておくと、**`--problem original`と併用したときのみ**、モノクロームで(たとえば、512x512x1で)学習や推論が行われます。**推論時にも必須なので注意。** 
- また、イメージサイズは `--image_size 512`などと明示してください。これは推論時は省略できると思いますが一応つけておくことを推奨。



# 2019.5.7 花岡変更Ver.

## 追加点

- 目的関数 ln p(x) を出力できるようにしました
	- 推論時 (--inference を指定したとき) 
	- ログディレクトリに`objective.npy`として出力されます
- オリジナルデータセットのリーダを追加しました
	- 詳細はdata_loaders\get_original.pyを参照
	- コマンドラインでのオプションとして、`--problem original --data_dir /lustre/gh68/g44001/glow/test.json`のように指定します
		- (/lustre/gh68/g44001/glow/test.jsonにデータセット定義ファイルがある場合) 
	- データセット定義ファイルtest.jsonの内容サンプルは

```

{
  "training": {
    "list": "/lustre/gh68/g44001/glow/lists/lists_unsupervised_pa/training_even.txt",
    "image": "/lustre/gh68/share/nih-cxp-dataset/images_256x256"
  },
  "test": {
    "list": "/lustre/gh68/g44001/glow/lists/lists_unsupervised_pa/test_even.txt",
    "image": "/lustre/gh68/share/nih-cxp-dataset/images_256x256"
  }
}

```

	- 学習時には"training"がわのデータセットで学習し、"test"がわのデータセットでvalidationします
	- 推論時には"test"がわのデータセットで推論します。"training"がわのデータセットは使わないので"test"と同じものでOKです
	- "list"はテキストで、画像ファイル名を列挙したものです
	- "image"のディレクトリに実際の画像ファイルを置いてください

- 与えられた各症例について、擬似正常症例を作成し、それとの補間or補外を作成します

	- オプション `--q_path` を使って実現します
	- 正常例セットを学習させ、ログディレクトリに出力された`z.npy`を、ツール `qr.py`に食べさせると、`q.npy`が生成されます
	- このq.npyを、推論時に`--q_path q.npy`のように指定してください
	- mixing ratioを、`--ratioOriginalZ` で指定できます
		- たとえば `--q_path q.npy --ratioOriginalZ 1.1` とすると、10%だけ異常強調された出力画像が`xprime.npy`に出力されます

## 注意点

- 現状では、推論モード `--inference` のときにマルチタスクにすると動作がおかしいようです
	- 推論モード `--inference` のときは `mpiexec -n 1 python...` としてください(遅いですが)

- ハイパーパラメータの与え方ですが、
	- `--image_size 256`
		- イメージサイズは明示して与えてください、2のべき乗である必要があるようです
	- `--learntop` は外してください
		- objectiveの計算がおかしくなる可能性があるので外してください。演算結果はほとんど変わらないようです
	- `--seed 0` は指定してください
		- 再現性を保つため必ず指定しましょう
	- `--n_bits_x 8`
		- 画像が8bitですので、このように指定する必要があるようです


## 花岡の環境におけるコマンドライン例 (gpu4枚のとき)

```

mpiexec -n 4 python /lustre/gh68/g44001/glow/train.py --problem original --data_dir /lustre/gh68/g44001/glow/train.json --image_size 256 --n_level 6 --depth 32 --flow_permutation 2 --flow_coupling 1 --seed 0 --lr 0.001 --n_bits_x 8 --n_batch_train 128 --epochs_full_valid 50 --epochs_full_sample 10

```

モノクロ512x512のとき(メインメモリ容量注意)

```

mpiexec -n 4 python /lustre/gh68/g44001/glow/train.py --problem original --data_dir /lustre/gh68/g44001/glow/train.json --image_size 512 --n_level 7 --depth 32 --flow_permutation 2 --flow_coupling 1 --seed 0 --lr 0.001 --n_bits_x 8 --n_batch_train 128 --epochs_full_valid 50 --epochs_full_sample 10 --image_color_num 1

```



