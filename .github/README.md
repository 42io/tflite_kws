# TensorFlow Lite Keyword Spotting
Native C/C++. Suitable for embedded devices.

    ~$ git clone --recursive --depth 1 https://github.com/42io/tflite_kws.git

### Inference
Default models pre-trained on 0-9 words: zero one two three four five six seven eight nine.

    ~$ arecord -f S16_LE -c1 -r16000 -d1 test.wav
    ~$ aplay test.wav
    ~$ dataset/dataset/google_speech_commands/src/features/build.sh
    ~$ src/brain/build.sh
    ~$ alias fe=dataset/dataset/google_speech_commands/bin/fe
    ~$ fe test.wav | bin/guess models/dcnn.tflite
    ~$ fe test.wav | head -48 | tail -47 | bin/guess models/dcnn47.tflite
    ~$ cat <(fe test.wav) <(seq 52) | bin/guess models/dcnn13.tflite | tail -1

Delay for streaming model 5 layers is `5 * 13 = 65`.

### Real Time
Microphone quality is very important. You should probably think about how to remove fan noise from the mic... Using headset seems like a good idea :)

    ~$ argmax() { mawk -Winteractive '{m=$1;j=1;for(i=j;i<=NF;i++)if($i>m){m=$i;j=i;}print j-1}'; }
    ~$ stable() { mawk -Winteractive -v u=$1 '{if(x!=$1){c=0;x=$1}else if(++c==u&&y!=x)print y=x}'; }
    ~$ ignore() { mawk -Winteractive -v t=$1 '{if($1<t)print $1}'; }

Simple non-streaming mode. Model receives the whole input sequence and then returns the classification result:

    ~$ arecord -f S16_LE -c1 -r16000 -t raw | fe | \
       bin/ring 47 | bin/guess models/dcnn47.tflite | argmax | stable 10 | ignore 10

[Streaming](https://arxiv.org/abs/2005.06720) mode is more CPU friendly as it reduces MAC operations in neural
network. Model receives portion of the input sequence and classifies it incrementally:

    ~$ arecord -f S16_LE -c1 -r16000 -t raw | fe | \
       bin/guess models/dcnn13.tflite | argmax | stable 10 | ignore 10

### Training
Jupyter Notebooks [MLP](jupyter/mlp.ipynb) | [CNN](jupyter/cnn.ipynb) | [RNN](jupyter/rnn.ipynb) | [DCNN](jupyter/dcnn.ipynb) | [DCNN47](jupyter/dcnn47.ipynb) | [DCNN13](jupyter/dcnn13.ipynb) | [EDCNN47](jupyter/edcnn47.ipynb) | [ECNN47](jupyter/ecnn47.ipynb) | [2ECNN47](jupyter/2ecnn47_tpu.ipynb).

Each notebook generates model file. To evaluate model accuracy:

    ~$ apt install gcc lrzip wget
    ~$ wget https://github.com/42io/dataset/releases/download/v1.0/0-9up.lrz -O /tmp/0-9up.lrz
    ~$ lrunzip /tmp/0-9up.lrz -o /tmp/0-9up.data # md5 87fc2460c7b6cd3dcca6807e9de78833
    ~$ dataset/matrix.sh /tmp/0-9up.data

Confusion matrix for pre-trained modeles:

    MLP confusion matrix...
    zero   0.93 0.00 0.03 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.03 0.00 | 603
    one    0.00 0.85 0.00 0.00 0.01 0.01 0.00 0.00 0.00 0.05 0.06 0.01 | 575
    two    0.03 0.00 0.86 0.02 0.02 0.00 0.00 0.01 0.01 0.00 0.04 0.01 | 564
    three  0.00 0.00 0.01 0.90 0.00 0.01 0.01 0.01 0.04 0.01 0.01 0.01 | 548
    four   0.00 0.01 0.01 0.00 0.90 0.01 0.00 0.00 0.00 0.00 0.05 0.01 | 605
    five   0.00 0.01 0.00 0.01 0.01 0.80 0.01 0.03 0.01 0.03 0.09 0.01 | 607
    six    0.00 0.00 0.00 0.00 0.00 0.00 0.96 0.00 0.00 0.00 0.02 0.01 | 462
    seven  0.01 0.00 0.03 0.01 0.00 0.00 0.01 0.90 0.00 0.00 0.03 0.01 | 574
    eight  0.00 0.00 0.01 0.07 0.00 0.00 0.03 0.00 0.84 0.01 0.03 0.01 | 547
    nine   0.00 0.04 0.00 0.01 0.00 0.01 0.00 0.01 0.00 0.86 0.06 0.01 | 596
    #unk#  0.02 0.03 0.03 0.05 0.06 0.07 0.02 0.03 0.02 0.07 0.58 0.02 | 730
    #pub#  0.00 0.00 0.01 0.00 0.00 0.01 0.01 0.00 0.00 0.00 0.00 0.96 | 730
    MLP guessed wrong 1029...

    CNN confusion matrix...
    zero   0.97 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.02 0.00 | 603
    one    0.00 0.93 0.00 0.00 0.00 0.01 0.00 0.00 0.00 0.01 0.05 0.00 | 575
    two    0.01 0.00 0.95 0.00 0.00 0.00 0.00 0.01 0.00 0.00 0.03 0.00 | 564
    three  0.00 0.00 0.00 0.91 0.00 0.00 0.01 0.01 0.01 0.00 0.06 0.00 | 548
    four   0.00 0.00 0.00 0.00 0.90 0.00 0.00 0.00 0.00 0.00 0.09 0.00 | 605
    five   0.00 0.00 0.00 0.00 0.00 0.93 0.00 0.00 0.01 0.01 0.06 0.00 | 607
    six    0.00 0.00 0.00 0.00 0.00 0.00 0.99 0.00 0.00 0.00 0.01 0.00 | 462
    seven  0.00 0.00 0.00 0.00 0.00 0.00 0.01 0.97 0.00 0.00 0.02 0.00 | 574
    eight  0.00 0.00 0.01 0.01 0.00 0.01 0.01 0.00 0.93 0.00 0.03 0.00 | 547
    nine   0.00 0.01 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.93 0.06 0.00 | 596
    #unk#  0.01 0.01 0.00 0.02 0.02 0.00 0.00 0.00 0.00 0.01 0.92 0.01 | 730
    #pub#  0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.01 0.98 | 730
    CNN guessed wrong 427...

    RNN confusion matrix...
    zero   0.98 0.00 0.01 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 | 603
    one    0.00 0.95 0.00 0.00 0.00 0.01 0.00 0.00 0.00 0.01 0.02 0.00 | 575
    two    0.00 0.00 0.98 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.01 0.00 | 564
    three  0.00 0.00 0.00 0.97 0.00 0.00 0.01 0.00 0.01 0.00 0.01 0.00 | 548
    four   0.00 0.00 0.00 0.00 0.97 0.00 0.00 0.00 0.00 0.00 0.02 0.00 | 605
    five   0.00 0.00 0.00 0.00 0.01 0.98 0.00 0.00 0.00 0.00 0.01 0.00 | 607
    six    0.00 0.00 0.00 0.00 0.00 0.00 0.99 0.00 0.00 0.00 0.00 0.00 | 462
    seven  0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.98 0.00 0.00 0.01 0.00 | 574
    eight  0.00 0.00 0.00 0.01 0.00 0.00 0.00 0.00 0.97 0.00 0.01 0.00 | 547
    nine   0.00 0.01 0.00 0.00 0.00 0.01 0.00 0.00 0.00 0.97 0.02 0.00 | 596
    #unk#  0.00 0.01 0.00 0.01 0.02 0.02 0.00 0.00 0.01 0.02 0.91 0.00 | 730
    #pub#  0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.99 | 730
    RNN guessed wrong 220...

    DCNN confusion matrix...
    zero   0.98 0.00 0.00 0.00 0.00 0.00 0.00 0.01 0.00 0.00 0.00 0.00 | 603
    one    0.00 0.98 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.01 0.01 0.00 | 575
    two    0.01 0.00 0.98 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.01 0.00 | 564
    three  0.00 0.00 0.00 0.97 0.00 0.00 0.01 0.00 0.01 0.00 0.00 0.00 | 548
    four   0.00 0.00 0.00 0.00 0.98 0.00 0.00 0.00 0.00 0.00 0.01 0.00 | 605
    five   0.00 0.00 0.00 0.00 0.00 0.98 0.00 0.00 0.00 0.00 0.01 0.00 | 607
    six    0.00 0.00 0.00 0.00 0.00 0.00 0.99 0.00 0.00 0.00 0.00 0.00 | 462
    seven  0.00 0.00 0.00 0.00 0.00 0.00 0.00 1.00 0.00 0.00 0.00 0.00 | 574
    eight  0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.99 0.00 0.01 0.00 | 547
    nine   0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.98 0.01 0.00 | 596
    #unk#  0.00 0.01 0.01 0.01 0.01 0.00 0.00 0.00 0.00 0.00 0.94 0.00 | 730
    #pub#  0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 1.00 | 730
    DCNN guessed wrong 143...

    DCNN47 confusion matrix...
    zero   0.99 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 | 603
    one    0.00 0.98 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.01 0.01 0.00 | 575
    two    0.00 0.00 0.99 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 | 564
    three  0.00 0.00 0.01 0.97 0.00 0.00 0.00 0.00 0.00 0.00 0.01 0.00 | 548
    four   0.00 0.00 0.00 0.00 0.99 0.00 0.00 0.00 0.00 0.00 0.01 0.00 | 605
    five   0.00 0.00 0.00 0.00 0.00 0.99 0.00 0.00 0.00 0.00 0.00 0.00 | 607
    six    0.00 0.00 0.00 0.00 0.00 0.00 1.00 0.00 0.00 0.00 0.00 0.00 | 462
    seven  0.00 0.00 0.00 0.00 0.00 0.00 0.00 1.00 0.00 0.00 0.00 0.00 | 574
    eight  0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.99 0.00 0.00 0.00 | 547
    nine   0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.99 0.01 0.00 | 596
    #unk#  0.00 0.01 0.00 0.00 0.01 0.00 0.00 0.00 0.00 0.00 0.97 0.00 | 730
    #pub#  0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 1.00 | 730
    DCNN47 guessed wrong 88...

    DCNN13 confusion matrix...
    zero   1.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 | 603
    one    0.00 0.98 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.01 0.01 0.00 | 575
    two    0.00 0.00 0.99 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.01 0.00 | 564
    three  0.00 0.00 0.00 0.98 0.00 0.00 0.01 0.00 0.00 0.00 0.01 0.00 | 548
    four   0.00 0.00 0.00 0.00 0.99 0.00 0.00 0.00 0.00 0.00 0.01 0.00 | 605
    five   0.00 0.00 0.00 0.00 0.00 0.99 0.00 0.00 0.00 0.00 0.01 0.00 | 607
    six    0.00 0.00 0.00 0.00 0.00 0.00 1.00 0.00 0.00 0.00 0.00 0.00 | 462
    seven  0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.99 0.00 0.00 0.00 0.00 | 574
    eight  0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.99 0.00 0.01 0.00 | 547
    nine   0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.99 0.01 0.00 | 596
    #unk#  0.00 0.01 0.00 0.00 0.01 0.00 0.00 0.00 0.00 0.00 0.97 0.00 | 730
    #pub#  0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 1.00 | 730
    DCNN13 guessed wrong 82...

    EDCNN47 confusion matrix...
    zero   0.98 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.01 0.00 | 603
    one    0.00 0.98 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.01 0.02 0.00 | 575
    two    0.00 0.00 0.98 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.02 0.00 | 564
    three  0.00 0.00 0.00 0.97 0.00 0.00 0.01 0.00 0.00 0.00 0.03 0.00 | 548
    four   0.00 0.00 0.00 0.00 0.97 0.00 0.00 0.00 0.00 0.00 0.03 0.00 | 605
    five   0.00 0.00 0.00 0.00 0.00 0.98 0.00 0.00 0.00 0.00 0.01 0.00 | 607
    six    0.00 0.00 0.00 0.00 0.00 0.00 0.99 0.00 0.00 0.00 0.01 0.00 | 462
    seven  0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.99 0.00 0.00 0.01 0.00 | 574
    eight  0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 1.00 0.00 0.00 0.00 | 547
    nine   0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.99 0.01 0.00 | 596
    #unk#  0.00 0.01 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.98 0.00 | 730
    #pub#  0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 1.00 | 730
    EDCNN47 guessed wrong 116...

    ECNN47 confusion matrix...
    zero   1.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 | 603
    one    0.00 0.98 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.01 0.01 0.00 | 575
    two    0.00 0.00 0.99 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.01 0.00 | 564
    three  0.00 0.00 0.00 0.98 0.00 0.00 0.01 0.00 0.00 0.00 0.01 0.00 | 548
    four   0.00 0.00 0.00 0.00 0.99 0.00 0.00 0.00 0.00 0.00 0.01 0.00 | 605
    five   0.00 0.00 0.00 0.00 0.00 1.00 0.00 0.00 0.00 0.00 0.00 0.00 | 607
    six    0.00 0.00 0.00 0.00 0.00 0.00 1.00 0.00 0.00 0.00 0.00 0.00 | 462
    seven  0.00 0.00 0.00 0.00 0.00 0.00 0.00 1.00 0.00 0.00 0.00 0.00 | 574
    eight  0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 1.00 0.00 0.00 0.00 | 547
    nine   0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.99 0.00 0.00 | 596
    #unk#  0.00 0.01 0.00 0.01 0.00 0.00 0.00 0.00 0.00 0.00 0.98 0.00 | 730
    #pub#  0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 1.00 | 730
    ECNN47 guessed wrong 63...

    2ECNN47 confusion matrix...
    zero   1.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 | 603
    one    0.00 0.98 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.01 0.01 0.00 | 575
    two    0.00 0.00 0.99 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.01 0.00 | 564
    three  0.00 0.00 0.00 0.97 0.00 0.00 0.01 0.00 0.00 0.00 0.02 0.00 | 548
    four   0.00 0.00 0.00 0.00 0.99 0.00 0.00 0.00 0.00 0.00 0.01 0.00 | 605
    five   0.00 0.00 0.00 0.00 0.00 0.98 0.00 0.00 0.00 0.00 0.01 0.00 | 607
    six    0.00 0.00 0.00 0.00 0.00 0.00 1.00 0.00 0.00 0.00 0.00 0.00 | 462
    seven  0.00 0.00 0.00 0.00 0.00 0.00 0.00 1.00 0.00 0.00 0.00 0.00 | 574
    eight  0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.99 0.00 0.01 0.00 | 547
    nine   0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.99 0.01 0.00 | 596
    #unk#  0.00 0.01 0.00 0.01 0.00 0.00 0.00 0.00 0.00 0.00 0.97 0.00 | 730
    #pub#  0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 1.00 | 730
    2ECNN47 guessed wrong 93...

Evaluate false positives:

    ~$ wget https://data.deepai.org/timit.zip -O /tmp/timit.zip
    ~$ unzip -q /tmp/timit.zip -d /tmp/timit # md5 5b736303c55cf4970926bb9978b655fe
    ~$ dataset/false.sh /tmp/timit 100

A false positive error, or false positive, is a result that indicates a given condition exists when it does not.

    2ECNN47   48 | 11191
    EDCNN47 2042 | 11191
    ECNN47  4494 | 11191
    DCNN13  4787 | 11191
    DCNN47  4517 | 11191
    MLP     5091 | 10991
    CNN     4958 | 10991
    RNN     4527 | 10991

### Heap Memory Usage
Some magic numbers to know before stepping into embedded world.

    ~$ valgrind dataset/dataset/google_speech_commands/bin/fe test.wav # 638,336 bytes allocated
    ~$ seq 637 | valgrind bin/guess models/mlp.tflite                  # 152,302 bytes allocated
    ~$ seq 637 | valgrind bin/guess models/cnn.tflite                  # 896,442 bytes allocated
    ~$ seq 637 | valgrind bin/guess models/rnn.tflite                  # 2,379,574 bytes allocated
    ~$ seq 637 | valgrind bin/guess models/dcnn.tflite                 # 459,306 bytes allocated
    ~$ seq 611 | valgrind bin/guess models/dcnn47.tflite               # 974,946 bytes allocated
    ~$ seq 13  | valgrind bin/guess models/dcnn13.tflite               # 683,750 bytes allocated
    ~$ seq 611 | valgrind bin/guess models/edcnn47.tflite              # 1,664,188 bytes allocated
    ~$ seq 611 | valgrind bin/guess models/ecnn47.tflite               # 8,625,734 bytes allocated
    ~$ seq 611 | valgrind bin/guess models/2ecnn47.tflite              # 22,938,842 bytes allocated

### Play
Let's consider voice control for led bulb.

    ~$ bigram() { mawk -Winteractive '{if(s)print prev,$0; prev=$0; s=1}'; }
    ~$ intent() { mawk -Winteractive '
        /0 6/{system("./on.sh")}
        /0 7/{system("./off.sh")}
        /0 8/{system("./yellow.sh")}
        /0 9/{system("./white.sh")}
        '; }

There are 4 commands here - turn on, off, change color. When we speak words `zero six`, script `./on.sh` will be executed e.t.c.

    ~$ ./mic.sh models/2ecnn47.tflite | bigram | intent