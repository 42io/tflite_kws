# TensorFlow Lite Keyword Spotting
Native C/C++. Suitable for embedded devices.

    ~$ git clone --recursive --depth 1 https://github.com/42io/tflite_kws.git
    ~$ cd tflite_kws
    ~$ apt install cmake gcc g++ unzip lrzip wget

### Inference
Default models pre-trained on 0-9 words: zero one two three four five six seven eight nine.

    ~$ arecord -f S16_LE -c1 -r16000 -d1 test.wav
    ~$ aplay test.wav
    ~$ src/features/build.sh
    ~$ src/brain/build.sh
    ~$ bin/fe test.wav | head -48 | tail -47 | bin/guess models/dcnn47.tflite
    ~$ cat <(bin/fe test.wav) <(seq 52) | bin/guess models/dcnn13.tflite | tail -1

Delay for streaming model 5 layers is `5 * 13 = 65`.

### Real Time
Microphone quality is very important. You should probably think about how to remove fan noise from the mic... Using headset seems like a good idea :)

    ~$ argmax() { mawk -Winteractive '{m=$1;j=1;for(i=j;i<=NF;i++)if($i>m){m=$i;j=i;}print j-1}'; }
    ~$ stable() { mawk -Winteractive -v u=$1 '{if(x!=$1){c=0;x=$1}else if(++c==u&&y!=x)print y=x}'; }
    ~$ ignore() { mawk -Winteractive -v t=$1 '{if($1<t)print $1}'; }

Simple non-streaming mode. Model receives the whole input sequence and then returns the classification result:

    ~$ arecord -f S16_LE -c1 -r16000 -t raw | bin/fe | bin/ring 47 | \
       bin/guess models/dcnn47.tflite | argmax | stable 10 | ignore 10

[Streaming](https://arxiv.org/abs/2005.06720) mode is more CPU friendly as it reduces MAC operations in neural
network. Model receives portion of the input sequence and classifies it incrementally:

    ~$ arecord -f S16_LE -c1 -r16000 -t raw | bin/fe | \
       bin/guess models/dcnn13.tflite | argmax | stable 10 | ignore 10

### Training
[DCNN47](jupyter/dcnn47.ipynb) | [2ECNN47](jupyter/2ecnn47_tpu.ipynb) | [ECNN47](jupyter/ecnn47.ipynb) | [S2CNN47](jupyter/s2cnn47_tpu.ipynb) \
[DCNN13](jupyter/dcnn13.ipynb) | [2ECNN13](jupyter/2ecnn13.ipynb) | [S2CNN13](jupyter/s2cnn13_tpu.ipynb)

Each notebook generates model file. To evaluate model accuracy:

    ~$ wget https://github.com/42io/dataset/releases/download/v1.0/0-9up.lrz -O /tmp/0-9up.lrz
    ~$ lrunzip /tmp/0-9up.lrz -o /tmp/0-9up.data # md5 87fc2460c7b6cd3dcca6807e9de78833
    ~$ dataset/matrix.sh /tmp/0-9up.data

Confusion matrix for pre-trained modeles:

    DCNN47 confusion matrix...
    zero   0.99  .    .    .    .    .    .    .    .    .    .    .   | 603
    one     .   0.98  .    .    .    .    .    .    .   0.01 0.01  .   | 575
    two     .    .   0.99  .    .    .    .    .    .    .    .    .   | 564
    three   .    .   0.01 0.97  .    .    .    .    .    .   0.01  .   | 548
    four    .    .    .    .   0.99  .    .    .    .    .   0.01  .   | 605
    five    .    .    .    .    .   0.99  .    .    .    .    .    .   | 607
    six     .    .    .    .    .    .   1.00  .    .    .    .    .   | 462
    seven   .    .    .    .    .    .    .   1.00  .    .    .    .   | 574
    eight   .    .    .    .    .    .    .    .   0.99  .    .    .   | 547
    nine    .    .    .    .    .    .    .    .    .   0.99 0.01  .   | 596
    #unk#   .   0.01  .    .   0.01  .    .    .    .    .   0.97  .   | 730
    #pub#   .    .    .    .    .    .    .    .    .    .    .   1.00 | 730
    DCNN47 guessed wrong 88...

    DCNN13 confusion matrix...
    zero   1.00  .    .    .    .    .    .    .    .    .    .    .   | 603
    one     .   0.98  .    .    .    .    .    .    .   0.01 0.01  .   | 575
    two     .    .   0.99  .    .    .    .    .    .    .   0.01  .   | 564
    three   .    .    .   0.98  .    .   0.01  .    .    .   0.01  .   | 548
    four    .    .    .    .   0.99  .    .    .    .    .   0.01  .   | 605
    five    .    .    .    .    .   0.99  .    .    .    .   0.01  .   | 607
    six     .    .    .    .    .    .   1.00  .    .    .    .    .   | 462
    seven   .    .    .    .    .    .    .   0.99  .    .    .    .   | 574
    eight   .    .    .    .    .    .    .    .   0.99  .   0.01  .   | 547
    nine    .    .    .    .    .    .    .    .    .   0.99 0.01  .   | 596
    #unk#   .   0.01  .    .   0.01  .    .    .    .    .   0.97  .   | 730
    #pub#   .    .    .    .    .    .    .    .    .    .    .   1.00 | 730
    DCNN13 guessed wrong 82...

    ECNN47 confusion matrix...
    zero   1.00  .    .    .    .    .    .    .    .    .    .    .   | 603
    one     .   0.98  .    .    .    .    .    .    .   0.01 0.01  .   | 575
    two     .    .   0.99  .    .    .    .    .    .    .   0.01  .   | 564
    three   .    .    .   0.98  .    .   0.01  .    .    .   0.01  .   | 548
    four    .    .    .    .   0.99  .    .    .    .    .   0.01  .   | 605
    five    .    .    .    .    .   1.00  .    .    .    .    .    .   | 607
    six     .    .    .    .    .    .   1.00  .    .    .    .    .   | 462
    seven   .    .    .    .    .    .    .   1.00  .    .    .    .   | 574
    eight   .    .    .    .    .    .    .    .   1.00  .    .    .   | 547
    nine    .    .    .    .    .    .    .    .    .   0.99  .    .   | 596
    #unk#   .   0.01  .   0.01  .    .    .    .    .    .   0.98  .   | 730
    #pub#   .    .    .    .    .    .    .    .    .    .    .   1.00 | 730
    ECNN47 guessed wrong 63...

    2ECNN47 confusion matrix...
    zero   1.00  .    .    .    .    .    .    .    .    .    .    .   | 603
    one     .   0.98  .    .    .    .    .    .    .   0.01 0.01  .   | 575
    two     .    .   0.99  .    .    .    .    .    .    .   0.01  .   | 564
    three   .    .    .   0.97  .    .   0.01  .    .    .   0.02  .   | 548
    four    .    .    .    .   0.99  .    .    .    .    .   0.01  .   | 605
    five    .    .    .    .    .   0.98  .    .    .    .   0.01  .   | 607
    six     .    .    .    .    .    .   1.00  .    .    .    .    .   | 462
    seven   .    .    .    .    .    .    .   1.00  .    .    .    .   | 574
    eight   .    .    .    .    .    .    .    .   0.99  .   0.01  .   | 547
    nine    .    .    .    .    .    .    .    .    .   0.99 0.01  .   | 596
    #unk#   .   0.01  .   0.01  .    .    .    .    .    .   0.97  .   | 730
    #pub#   .    .    .    .    .    .    .    .    .    .    .   1.00 | 730
    2ECNN47 guessed wrong 93...

    2ECNN13 confusion matrix...
    zero   1.00  .    .    .    .    .    .    .    .    .    .    .   | 603
    one     .   0.97  .    .    .    .    .    .    .   0.01 0.02  .   | 575
    two     .    .   0.99  .    .    .    .    .    .    .   0.01  .   | 564
    three   .    .    .   0.97  .    .    .    .    .    .   0.02  .   | 548
    four    .    .    .    .   0.98  .    .    .    .    .   0.01  .   | 605
    five    .    .    .    .    .   0.98  .    .    .    .   0.01  .   | 607
    six     .    .    .    .    .    .   1.00  .    .    .    .    .   | 462
    seven   .    .    .    .    .    .    .   1.00  .    .    .    .   | 574
    eight   .    .    .    .    .    .    .    .   1.00  .    .    .   | 547
    nine    .    .    .    .    .    .    .    .    .   0.98 0.01  .   | 596
    #unk#   .   0.01  .   0.01  .    .    .    .    .    .   0.98  .   | 730
    #pub#   .    .    .    .    .    .    .    .    .    .    .   1.00 | 730
    2ECNN13 guessed wrong 92...

    S2CNN47 confusion matrix...
    zero   1.00  .    .    .    .    .    .    .    .    .    .    .   | 603
    one     .   0.99  .    .    .    .    .    .    .    .   0.01  .   | 575
    two     .    .   1.00  .    .    .    .    .    .    .    .    .   | 564
    three   .    .    .   0.99  .    .    .    .    .    .   0.01  .   | 548
    four    .    .    .    .   1.00  .    .    .    .    .    .    .   | 605
    five    .    .    .    .    .   1.00  .    .    .    .    .    .   | 607
    six     .    .    .    .    .    .   1.00  .    .    .    .    .   | 462
    seven   .    .    .    .    .    .    .   1.00  .    .    .    .   | 574
    eight   .    .    .    .    .    .    .    .   1.00  .    .    .   | 547
    nine    .    .    .    .    .    .    .    .    .   0.99 0.01  .   | 596
    #unk#   .    .    .    .    .    .    .    .    .    .   0.99  .   | 730
    #pub#   .    .    .    .    .    .    .    .    .    .    .   1.00 | 730
    S2CNN47 guessed wrong 38...

    S2CNN13 confusion matrix...
    zero    1.00  .    .    .    .    .    .    .    .    .    .    .   | 603
    one      .   0.99  .    .    .    .    .    .    .    .   0.01  .   | 575
    two      .    .   1.00  .    .    .    .    .    .    .    .    .   | 564
    three    .    .    .   0.99  .    .    .    .    .    .   0.01  .   | 548
    four     .    .    .    .   1.00  .    .    .    .    .    .    .   | 605
    five     .    .    .    .    .   1.00  .    .    .    .    .    .   | 607
    six      .    .    .    .    .    .   1.00  .    .    .    .    .   | 462
    seven    .    .    .    .    .    .    .   1.00  .    .    .    .   | 574
    eight    .    .    .    .    .    .    .    .   0.99  .    .    .   | 547
    nine     .    .    .    .    .    .    .    .    .   0.99 0.01  .   | 596
    #unk#    .    .    .   0.01  .    .    .    .    .    .   0.98  .   | 730
    #pub#    .    .    .    .    .    .    .    .    .    .    .   1.00 | 730
    S2CNN13 guessed wrong 45...

Evaluate false positives:

    ~$ wget https://data.deepai.org/timit.zip -O /tmp/timit.zip
    ~$ unzip -q /tmp/timit.zip -d /tmp/timit # md5 5b736303c55cf4970926bb9978b655fe
    ~$ dataset/false.sh /tmp/timit 100

A false positive error, or false positive, is a result that indicates a given condition exists when it does not.

    S2CNN47   12 | 11191
    S2CNN13   22 | 11191
    2ECNN13   83 | 11191
    2ECNN47   48 | 11191
    ECNN47  4494 | 11191
    DCNN13  4787 | 11191
    DCNN47  4517 | 11191

### Heap Memory Usage
Some magic numbers to know before stepping into embedded world.

    ~$ head /dev/zero -c32000 | valgrind bin/fe           # 1,136,764 bytes allocated
    ~$ seq 611 | valgrind bin/guess models/dcnn47.tflite  # 981,583 bytes allocated
    ~$ seq 13  | valgrind bin/guess models/dcnn13.tflite  # 689,566 bytes allocated
    ~$ seq 611 | valgrind bin/guess models/ecnn47.tflite  # 8,637,011 bytes allocated
    ~$ seq 611 | valgrind bin/guess models/2ecnn47.tflite # 22,956,483 bytes allocated
    ~$ seq 13  | valgrind bin/guess models/2ecnn13.tflite # 7,264,955 bytes allocated
    ~$ seq 611 | valgrind bin/guess models/s2cnn47.tflite # 28,190,401 bytes allocated
    ~$ seq 13  | valgrind bin/guess models/s2cnn13.tflite # 17,881,002 bytes allocated

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

    ~$ ./mic.sh models/s2cnn13.tflite | bigram | intent