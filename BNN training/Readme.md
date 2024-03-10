# BNN (IR-net)
這個檔案是利用bnn(IR-net)來做訓練，參考論文IR-net可以打開來看，另外我有附兩個我寫的propagation的筆記

**Remind**

這個檔案訓練方式IR-net是以訓練CIFAR-10為主，也有mnist，可以自己按照我底下的方式改。

**VGG-small**

我使用的模型是VGG-small

    conv - bn -
    conv - pool - bn - hardtanh -
    conv - bn - hardtanh -
    conv - pool - bn - hardtanh -
    conv - bn - hardtanh -
    conv - pool - bn - hardtanh - fc

**AlexNet**

我使用的模型是AlexNet

    conv - bn - pool - hardtanh -
    conv - bn - pool - hardtanh -
    conv - bn - hardtanh -
    conv - bn - hardtanh -
    conv - bn - pool - hardtanh - fc

**Command**

如果今天想要訓練CIFAR10:

     main.py 和 vgg.py 的程式自己改一下
        CIFAR10 = True 
        MNIST = False

    並執行檔案後面加上命令
      --save-dir cifar10_train_model --epochs 200 --save-every 5 --half

    下列看自己需求調整
        -b (batch size)
        --half (如果要用半精度訓練)
        -p (print frequency)
        --warm_up True (如果要用warm up可以加快前期訓練)

如果今天想要訓練MNIST:

    main.py 和 vgg.py 的程式自己改一下
        CIFAR10 = False 
        MNIST = True

    並執行檔案後面加上命令
      --save-dir mnist_train_model --epochs 10 --save-every 5 -b 100 -p 100

如果想要把之前暫停訓練從新開始繼續訓練可以多加下面的命令:

    --resume cifar10_train_model/vgg_small_checkpoint.th (這個是指到你要繼續練習的checkpoint檔)

如果想要評估一個最佳模型可以多加下面的命令:

    -e --model cifar10_train_model/vgg_small.th (這個是指到你要評估的模型檔)


**Accuracy:**

CIFAR-10(IR-net) (first layer mixed precision):

| Topology  | Bit-Width (W/A) | Accuracy (%) |
| --------- | --------------- | ------------ |
| ResNet-20 | 1 / 1           | 86.5         |
| ResNet-20 | 1 / 32          | 90.8         |
| VGG-Small | 1 / 1           | 90.4         |

CIFAR-10(IR-net) (first layer binary, bn layer => offset):

| Topology        | Bit-Width (W/A) | Accuracy (%) |
| --------------- | --------------- | ------------ |
| ResNet-20       | 1 / 1           | 68.4         |
| VGG-Small(half) | 1 / 1           | 64.8         |

CIFAR-10(IR-net) (first layer binary, bn layer => offset, BinarizeLinear):

| Topology        | Bit-Width (W/A) | Accuracy (%) |
| --------------- | --------------- | ------------ |
| Alex-net        | 1 / 1           | ????         |

CIFAR-10(IE-net):

| Topology  | Bit-Width (W/A) | Accuracy (%) |
| --------- | --------------- | ------------ |
| ResNet-18 | 1 / 1           | 92.9         |
| ResNet-20 | 1 / 1           | 88.5         |
| VGG-Small | 1 / 1           | 92.0         |


**Model**

我有附我已經訓練完的一些模型，如果你們要訓練cifar-10要花比較長時間，可以自己架設一個環境來訓練。

**File**

可以先跑VGG-Small/main.py 然後去看modules/ir_1w1a.py和modules/binaryfunction.py 裡面的propagation就是在實現論文中的方法

如果要測試其他神經網路可以跑跑看ResNet20/1w1a/trainer.py 這個檔案裡面我沒有加mnist訓練。

**Remark**

現在我有嘗試把fc layer (Linear)替換成binarlize(輸入和權重全為1bit)，但還在測試中，原本跑一個epoch要花6分鐘，現在優化程式碼後一個epoch也要花2分鐘，但是準確率還是很低，我還在測試中。

**Future**

1. ShiftIRConv2d的測試。
2. 輸入image的預處理(3 channel => 9 channel)。