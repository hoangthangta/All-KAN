python run.py --mode "train" --model_name "sech_kan_cnn" --epochs 10 --batch_size 16 --num_grids 4 --middle_channel 8 --out_channel 16 --ds_name "cal_si" --note "mlpfull_0" --base_activation "silu" --scheduler "OneCycleLR" --hidden_size 64 --classifier_type "mlp" --seed 0; 
sleep 5s; 
python run.py --mode "train" --model_name "sech_kan_cnn" --epochs 10 --batch_size 16 --num_grids 4 --middle_channel 8 --out_channel 16 --ds_name "mnist" --note "mlpfull_0" --base_activation "silu" --scheduler "OneCycleLR" --hidden_size 64 --classifier_type "mlp" --seed 0; 
sleep 5s; 
python run.py --mode "train" --model_name "sech_kan_cnn" --epochs 10 --batch_size 16 --num_grids 4 --middle_channel 8 --out_channel 16 --ds_name "fashion_mnist" --note "mlpfull_0" --base_activation "silu" --scheduler "OneCycleLR" --hidden_size 64 --classifier_type "mlp" --seed 0; 
sleep 5s; 
python run.py --mode "train" --model_name "sech_kan_cnn" --epochs 10 --batch_size 16 --num_grids 8 --middle_channel 24 --out_channel 48 --ds_name "cifar10" --note "mlpfull_0" --base_activation "silu" --scheduler "OneCycleLR" --hidden_size 60 --classifier_type "mlp" --seed 0;
sleep 5s; 
python run.py --mode "train" --model_name "sech_kan_cnn" --epochs 10 --batch_size 16 --num_grids 8 --middle_channel 24 --out_channel 48 --ds_name "cifar100" --note "mlpfull_0" --base_activation "silu" --scheduler "OneCycleLR" --hidden_size 250 --classifier_type "mlp" --seed 0;
sleep 5s; 


python run.py --mode "train" --model_name "sech_kan_cnn" --epochs 10 --batch_size 16 --num_grids 4 --middle_channel 8 --out_channel 16 --ds_name "cal_si" --note "mlpfull_1" --base_activation "silu" --scheduler "OneCycleLR" --hidden_size 64 --classifier_type "mlp" --seed 1; 
sleep 5s; 
python run.py --mode "train" --model_name "sech_kan_cnn" --epochs 10 --batch_size 16 --num_grids 4 --middle_channel 8 --out_channel 16 --ds_name "mnist" --note "mlpfull_1" --base_activation "silu" --scheduler "OneCycleLR" --hidden_size 64 --classifier_type "mlp" --seed 1; 
sleep 5s; 
python run.py --mode "train" --model_name "sech_kan_cnn" --epochs 10 --batch_size 16 --num_grids 4 --middle_channel 8 --out_channel 16 --ds_name "fashion_mnist" --note "mlpfull_1" --base_activation "silu" --scheduler "OneCycleLR" --hidden_size 64 --classifier_type "mlp" --seed 1; 
sleep 5s; 
python run.py --mode "train" --model_name "sech_kan_cnn" --epochs 10 --batch_size 16 --num_grids 8 --middle_channel 24 --out_channel 48 --ds_name "cifar10" --note "mlpfull_1" --base_activation "silu" --scheduler "OneCycleLR" --hidden_size 60 --classifier_type "mlp" --seed 1;
sleep 5s; 
python run.py --mode "train" --model_name "sech_kan_cnn" --epochs 10 --batch_size 16 --num_grids 8 --middle_channel 24 --out_channel 48 --ds_name "cifar100" --note "mlpfull_1" --base_activation "silu" --scheduler "OneCycleLR" --hidden_size 250 --classifier_type "mlp" --seed 1;
sleep 5s; 


python run.py --mode "train" --model_name "sech_kan_cnn" --epochs 10 --batch_size 16 --num_grids 4 --middle_channel 8 --out_channel 16 --ds_name "cal_si" --note "mlpfull_2" --base_activation "silu" --scheduler "OneCycleLR" --hidden_size 64 --classifier_type "mlp" --seed 2; 
sleep 5s; 
python run.py --mode "train" --model_name "sech_kan_cnn" --epochs 10 --batch_size 16 --num_grids 4 --middle_channel 8 --out_channel 16 --ds_name "mnist" --note "mlpfull_2" --base_activation "silu" --scheduler "OneCycleLR" --hidden_size 64 --classifier_type "mlp" --seed 2; 
sleep 5s; 
python run.py --mode "train" --model_name "sech_kan_cnn" --epochs 10 --batch_size 16 --num_grids 4 --middle_channel 8 --out_channel 16 --ds_name "fashion_mnist" --note "mlpfull_2" --base_activation "silu" --scheduler "OneCycleLR" --hidden_size 64 --classifier_type "mlp" --seed 2; 
sleep 5s; 
python run.py --mode "train" --model_name "sech_kan_cnn" --epochs 10 --batch_size 16 --num_grids 8 --middle_channel 24 --out_channel 48 --ds_name "cifar10" --note "mlpfull_2" --base_activation "silu" --scheduler "OneCycleLR" --hidden_size 60 --classifier_type "mlp" --seed 2;
sleep 5s; 
python run.py --mode "train" --model_name "sech_kan_cnn" --epochs 10 --batch_size 16 --num_grids 8 --middle_channel 24 --out_channel 48 --ds_name "cifar100" --note "mlpfull_2" --base_activation "silu" --scheduler "OneCycleLR" --hidden_size 250 --classifier_type "mlp" --seed 2;
sleep 5s; 


python run.py --mode "train" --model_name "sech_kan_cnn" --epochs 10 --batch_size 16 --num_grids 4 --middle_channel 8 --out_channel 16 --ds_name "cal_si" --note "mlpfull_3" --base_activation "silu" --scheduler "OneCycleLR" --hidden_size 64 --classifier_type "mlp" --seed 3; 
sleep 5s; 
python run.py --mode "train" --model_name "sech_kan_cnn" --epochs 10 --batch_size 16 --num_grids 4 --middle_channel 8 --out_channel 16 --ds_name "mnist" --note "mlpfull_3" --base_activation "silu" --scheduler "OneCycleLR" --hidden_size 64 --classifier_type "mlp" --seed 3; 
sleep 5s; 
python run.py --mode "train" --model_name "sech_kan_cnn" --epochs 10 --batch_size 16 --num_grids 4 --middle_channel 8 --out_channel 16 --ds_name "fashion_mnist" --note "mlpfull_3" --base_activation "silu" --scheduler "OneCycleLR" --hidden_size 64 --classifier_type "mlp" --seed 3; 
sleep 5s; 
python run.py --mode "train" --model_name "sech_kan_cnn" --epochs 10 --batch_size 16 --num_grids 8 --middle_channel 24 --out_channel 48 --ds_name "cifar10" --note "mlpfull_3" --base_activation "silu" --scheduler "OneCycleLR" --hidden_size 60 --classifier_type "mlp" --seed 3;
sleep 5s; 
python run.py --mode "train" --model_name "sech_kan_cnn" --epochs 10 --batch_size 16 --num_grids 8 --middle_channel 24 --out_channel 48 --ds_name "cifar100" --note "mlpfull_3" --base_activation "silu" --scheduler "OneCycleLR" --hidden_size 250 --classifier_type "mlp" --seed 3;
sleep 5s; 


python run.py --mode "train" --model_name "sech_kan_cnn" --epochs 10 --batch_size 16 --num_grids 4 --middle_channel 8 --out_channel 16 --ds_name "cal_si" --note "mlpfull_4" --base_activation "silu" --scheduler "OneCycleLR" --hidden_size 64 --classifier_type "mlp" --seed 4; 
sleep 5s; 
python run.py --mode "train" --model_name "sech_kan_cnn" --epochs 10 --batch_size 16 --num_grids 4 --middle_channel 8 --out_channel 16 --ds_name "mnist" --note "mlpfull_4" --base_activation "silu" --scheduler "OneCycleLR" --hidden_size 64 --classifier_type "mlp" --seed 4; 
sleep 5s; 
python run.py --mode "train" --model_name "sech_kan_cnn" --epochs 10 --batch_size 16 --num_grids 4 --middle_channel 8 --out_channel 16 --ds_name "fashion_mnist" --note "mlpfull_4" --base_activation "silu" --scheduler "OneCycleLR" --hidden_size 64 --classifier_type "mlp" --seed 4; 
sleep 5s; 
python run.py --mode "train" --model_name "sech_kan_cnn" --epochs 10 --batch_size 16 --num_grids 8 --middle_channel 24 --out_channel 48 --ds_name "cifar10" --note "mlpfull_4" --base_activation "silu" --scheduler "OneCycleLR" --hidden_size 60 --classifier_type "mlp" --seed 4;
sleep 5s; 
python run.py --mode "train" --model_name "sech_kan_cnn" --epochs 10 --batch_size 16 --num_grids 8 --middle_channel 24 --out_channel 48 --ds_name "cifar100" --note "mlpfull_4" --base_activation "silu" --scheduler "OneCycleLR" --hidden_size 250 --classifier_type "mlp" --seed 4;
sleep 5s; 