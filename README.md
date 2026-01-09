# winograd-HE

## Environment
```
Linux 6.17.2-arch1-1
g++ (GCC) 15.2.1 20250813
Python 3.13.7
```

## (Optional) Install OpenFHE
```sh
git clone https://github.com/openfheorg/openfhe-development.git
cd openfhe-development/
mkdir build
cd build/
cmake .. -DCMAKE_INSTALL_PREFIX="/home/ubuntu/openfhe_opt" -DCMAKE_BUILD_TYPE=Release
make
make install
```

## build
```sh
git clone git@github.com:winograd-he/winograd-framework.git
pip install -e .
CMAKE_ARGS="-DOpenFHE_DIR=/home/ubuntu/openfhe_opt/lib/OpenFHE" pip install -e backend/openfhe/bindings/python
```

## example
1. (Optional) Train mnist
    ```sh
    cd clear/mnist
    python3 cnn_HerPN_avgpool_train.py
    ```
2. Pack plaintexts
    ```sh
    cd pack
    python3 mnist-winograd.py
    ```
3. Run evaluator
    ```sh
    cd runner
    LD_LIBRARY_PATH="/home/ubuntu/openfhe_opt/lib:$LD_LIBRARY_PATH" python3 mnist.py --mode winograd
    ```
