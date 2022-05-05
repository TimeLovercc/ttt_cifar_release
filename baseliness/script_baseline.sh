export PYTHONPATH=$PYTHONPATH:$(pwd)

CUDA_VISIBLE_DEVICES=5 python main.py --group_norm 8\
			--nepoch 150 --milestone_1 75 --milestone_2 125 \
			--outf results/cifar10_none_gn
for i in $(seq 1 5);
do CUDA_VISIBLE_DEVICES=5 python script_test_c10.py $i none none gn
done
CUDA_VISIBLE_DEVICES=5 python script_test_c10.py 0 none none gn


