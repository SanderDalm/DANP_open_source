Example usage:

# CIFAR 3 layers
python main.py --dataset cifar10 --algorithm np --hidden_sizes 1024 1024 1024 --epochs 50 --batch_size 1000 --lr 1e-5 --noise_std 0.001 --num_noise_iters 1 --num_seeds 3 --write_results_dir results/cifar10_fc
python main.py --dataset cifar10 --algorithm anp --hidden_sizes 1024 1024 1024 --epochs 50 --batch_size 1000 --lr 1e-5 --noise_std 0.001 --num_noise_iters 1 --num_seeds 3 --write_results_dir results/cifar10_fc
python main.py --dataset cifar10 --algorithm inp --hidden_sizes 1024 1024 1024 --epochs 50 --batch_size 1000 --lr 1e-5 --noise_std 0.001 --num_noise_iters 1 --num_seeds 3 --write_results_dir results/cifar10_fc

python main.py --dataset cifar10 --algorithm dnp --hidden_sizes 1024 1024 1024 --epochs 50 --batch_size 1000 --lr 1e-4 --noise_std 0.001 --num_noise_iters 1 --num_seeds 3 --write_results_dir results/cifar10_fc
python main.py --dataset cifar10 --algorithm danp --hidden_sizes 1024 1024 1024 --epochs 50 --batch_size 1000 --lr 1e-4 --noise_std 0.001 --num_noise_iters 1 --num_seeds 3 --write_results_dir results/cifar10_fc
python main.py --dataset cifar10 --algorithm dinp --hidden_sizes 1024 1024 1024 --epochs 50 --batch_size 1000 --lr 1e-4 --noise_std 0.001 --num_noise_iters 1 --num_seeds 3 --write_results_dir results/cifar10_fc

python main.py --dataset cifar10 --algorithm bp --hidden_sizes 1024 1024 1024 --epochs 50 --batch_size 1000 --lr 1e-4 --noise_std 0.001 --num_noise_iters 1 --num_seeds 3 --write_results_dir results/cifar10_fc
python main.py --dataset cifar10 --algorithm dbp --hidden_sizes 1024 1024 1024 --epochs 50 --batch_size 1000 --lr 1e-3 --noise_std 0.001 --num_noise_iters 1 --num_seeds 3 --write_results_dir results/cifar10_fc

python plot.py --results_folder results/cifar10_fc