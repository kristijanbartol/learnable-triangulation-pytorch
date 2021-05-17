#!/bin/sh

if [[ "$(whoami)" == "kristijan" ]]; then
			REPO_DIR=/home/kristijan/phd/pose/learnable-triangulation-pytorch/
				else
								REPO_DIR=/home/dbojanic/pose/learnable-triangulation-pytorch/
fi

echo ${REPO_DIR}

docker run --rm --gpus all --name kbartol-learnable -it \
	-v ${REPO_DIR}:/learnable-triangulation \
	-v /media/kristijan/kristijan-hdd-ex/datasets/human36m/:/data/human36m kbartol-learnable
