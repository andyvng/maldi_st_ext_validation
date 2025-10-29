#!/bin/bash

run_id=$1
preprocessed_maldi=$2
checkpoints_dir=$3
test_ids_abs_path=$4
predict_out_dir=$5
test_classes=$6

if ! [[ "$run_id" =~ ^[0-9]+$ ]]; then
    echo "error: run_id must be an integer between 1 and 10" >&2
    exit 1
fi

if (( run_id < 1 || run_id > 10 )); then
    echo "error: run_id must be between 1 and 10 (got $run_id)" >&2
    exit 1
fi

docker run --rm \
            -v $(pwd):/maldist \
            -v ${preprocessed_maldi}:/preprocessed_MALDI \
            -v ${checkpoints_dir}:/checkpoints \
            -v ${test_ids_abs_path}:/data/test_ids.csv \
            maldist.outlier:latest \
            predict_out_dir=${predict_out_dir}/${run_id} \
            checkpoint.save_dir=/checkpoints/${run_id} \
            dataset.test_classes=${test_classes}
