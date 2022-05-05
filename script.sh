export PYTHONPATH=$PYTHONPATH:$(pwd)

CUDA_VISIBLE_DEVICES=4 python main.py --shared layer2 --rotation_type expand --color_type expand \
			--group_norm 8 \
			--nepoch 150 --milestone_1 75 --milestone_2 125 \
			--outf results/cifar10_layer2_gn_expand

CUDA_VISIBLE_DEVICES=4 python script_test_c10.py 5 layer2 slow gn_expand
CUDA_VISIBLE_DEVICES=4 python script_test_c10.py 5 layer2 online gn_expand

CUDA_VISIBLE_DEVICES=4 python script_test_c10.py 4 layer2 slow gn_expand
CUDA_VISIBLE_DEVICES=4 python script_test_c10.py 4 layer2 online gn_expand

CUDA_VISIBLE_DEVICES=4 python script_test_c10.py 3 layer2 slow gn_expand
CUDA_VISIBLE_DEVICES=4 python script_test_c10.py 3 layer2 online gn_expand

CUDA_VISIBLE_DEVICES=4 python script_test_c10.py 2 layer2 slow gn_expand
CUDA_VISIBLE_DEVICES=4 python script_test_c10.py 2 layer2 online gn_expand

CUDA_VISIBLE_DEVICES=4 python script_test_c10.py 1 layer2 slow gn_expand
CUDA_VISIBLE_DEVICES=4 python script_test_c10.py 1 layer2 online gn_expand

CUDA_VISIBLE_DEVICES=4 python script_test_c10.py 0 layer2 slow gn_expand
CUDA_VISIBLE_DEVICES=4 python script_test_c10.py 0 layer2 online gn_expand

