# seed 0
python run.py --mode "train" --model_name "mlp" --epochs 10 --batch_size 64 --layers "784,392,102" --ds_name "cal_si" --note "full_0" --base_activation "silu" --scheduler "OneCycleLR" --seed 0 --norm_type "";
sleep 5s;
python run.py --mode "train" --model_name "mlp" --epochs 10 --batch_size 64 --layers "784,64,10" --ds_name "mnist" --note "full_0" --base_activation "silu" --scheduler "OneCycleLR" --seed 0 --norm_type "";
sleep 5s;
python run.py --mode "train" --model_name "mlp" --epochs 10 --batch_size 64 --layers "784,64,10" --ds_name "fashion_mnist" --note "full_0" --base_activation "silu" --scheduler "OneCycleLR" --seed 0 --norm_type "";
sleep 5s;
python run.py --mode "train" --model_name "mlp" --epochs 10 --batch_size 16 --layers "3072,64,10" --ds_name "cifar10" --note "full_0" --base_activation "silu" --scheduler "OneCycleLR" --seed 0 --norm_type "";
sleep 5s;
python run.py --mode "train" --model_name "mlp" --epochs 10 --batch_size 16 --layers "3072,256,100" --ds_name "cifar100" --note "full_0" --base_activation "silu" --scheduler "OneCycleLR" --seed 0 --norm_type "";
sleep 5s;

# seed 1
python run.py --mode "train" --model_name "mlp" --epochs 10 --batch_size 64 --layers "784,392,102" --ds_name "cal_si" --note "full_1" --base_activation "silu" --scheduler "OneCycleLR" --seed 1 --norm_type "";
sleep 5s;
python run.py --mode "train" --model_name "mlp" --epochs 10 --batch_size 64 --layers "784,64,10" --ds_name "mnist" --note "full_1" --base_activation "silu" --scheduler "OneCycleLR" --seed 1 --norm_type "";
sleep 5s;
python run.py --mode "train" --model_name "mlp" --epochs 10 --batch_size 64 --layers "784,64,10" --ds_name "fashion_mnist" --note "full_1" --base_activation "silu" --scheduler "OneCycleLR" --seed 1 --norm_type "";
sleep 5s;
python run.py --mode "train" --model_name "mlp" --epochs 10 --batch_size 16 --layers "3072,64,10" --ds_name "cifar10" --note "full_1" --base_activation "silu" --scheduler "OneCycleLR" --seed 1 --norm_type "";
sleep 5s;
python run.py --mode "train" --model_name "mlp" --epochs 10 --batch_size 16 --layers "3072,256,100" --ds_name "cifar100" --note "full_1" --base_activation "silu" --scheduler "OneCycleLR" --seed 1 --norm_type "";
sleep 5s;

# seed 2
python run.py --mode "train" --model_name "mlp" --epochs 10 --batch_size 64 --layers "784,392,102" --ds_name "cal_si" --note "full_2" --base_activation "silu" --scheduler "OneCycleLR" --seed 2 --norm_type "";
sleep 5s;
python run.py --mode "train" --model_name "mlp" --epochs 10 --batch_size 64 --layers "784,64,10" --ds_name "mnist" --note "full_2" --base_activation "silu" --scheduler "OneCycleLR" --seed 2 --norm_type "";
sleep 5s;
python run.py --mode "train" --model_name "mlp" --epochs 10 --batch_size 64 --layers "784,64,10" --ds_name "fashion_mnist" --note "full_2" --base_activation "silu" --scheduler "OneCycleLR" --seed 2 --norm_type "";
sleep 5s;
python run.py --mode "train" --model_name "mlp" --epochs 10 --batch_size 16 --layers "3072,64,10" --ds_name "cifar10" --note "full_2" --base_activation "silu" --scheduler "OneCycleLR" --seed 2 --norm_type "";
sleep 5s;
python run.py --mode "train" --model_name "mlp" --epochs 10 --batch_size 16 --layers "3072,256,100" --ds_name "cifar100" --note "full_2" --base_activation "silu" --scheduler "OneCycleLR" --seed 2 --norm_type "";
sleep 5s;

# seed 3
python run.py --mode "train" --model_name "mlp" --epochs 10 --batch_size 64 --layers "784,392,102" --ds_name "cal_si" --note "full_3" --base_activation "silu" --scheduler "OneCycleLR" --seed 3 --norm_type "";
sleep 5s;
python run.py --mode "train" --model_name "mlp" --epochs 10 --batch_size 64 --layers "784,64,10" --ds_name "mnist" --note "full_3" --base_activation "silu" --scheduler "OneCycleLR" --seed 3 --norm_type "";
sleep 5s;
python run.py --mode "train" --model_name "mlp" --epochs 10 --batch_size 64 --layers "784,64,10" --ds_name "fashion_mnist" --note "full_3" --base_activation "silu" --scheduler "OneCycleLR" --seed 3 --norm_type "";
sleep 5s;
python run.py --mode "train" --model_name "mlp" --epochs 10 --batch_size 16 --layers "3072,64,10" --ds_name "cifar10" --note "full_3" --base_activation "silu" --scheduler "OneCycleLR" --seed 3 --norm_type "";
sleep 5s;
python run.py --mode "train" --model_name "mlp" --epochs 10 --batch_size 16 --layers "3072,256,100" --ds_name "cifar100" --note "full_3" --base_activation "silu" --scheduler "OneCycleLR" --seed 3 --norm_type "";
sleep 5s;

# seed 42
python run.py --mode "train" --model_name "mlp" --epochs 10 --batch_size 64 --layers "784,392,102" --ds_name "cal_si" --note "full_4" --base_activation "silu" --scheduler "OneCycleLR" --seed 4 --norm_type "";
sleep 5s;
python run.py --mode "train" --model_name "mlp" --epochs 10 --batch_size 64 --layers "784,64,10" --ds_name "mnist" --note "full_4" --base_activation "silu" --scheduler "OneCycleLR" --seed 4 --norm_type "";
sleep 5s;
python run.py --mode "train" --model_name "mlp" --epochs 10 --batch_size 64 --layers "784,64,10" --ds_name "fashion_mnist" --note "full_4" --base_activation "silu" --scheduler "OneCycleLR" --seed 4 --norm_type "";
sleep 5s;
python run.py --mode "train" --model_name "mlp" --epochs 10 --batch_size 16 --layers "3072,64,10" --ds_name "cifar10" --note "full_4" --base_activation "silu" --scheduler "OneCycleLR" --seed 4 --norm_type "";
sleep 5s;
python run.py --mode "train" --model_name "mlp" --epochs 10 --batch_size 16 --layers "3072,256,100" --ds_name "cifar100" --note "full_4" --base_activation "silu" --scheduler "OneCycleLR" --seed 4 --norm_type "";
sleep 5s;