#/bin/sh

docker run --rm --gpus all --name kbartol-triangulation -it -v /home/dbojanic/pose/learnable-triangulation-pytorch/:/learnable-triangulation/ learnable-triangulation
