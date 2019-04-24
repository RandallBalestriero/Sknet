screen -dmS a bash -c "tf3;sleep 2;export CUDA_VISIBLE_DEVICES='0';python quickstart_hessian.py 0.1 0";
screen -dmS b bash -c "tf3;sleep 2;export CUDA_VISIBLE_DEVICES='1';python quickstart_hessian.py 0.1 0.01";
screen -dmS c bash -c "tf3;sleep 2;export CUDA_VISIBLE_DEVICES='2';python quickstart_hessian.py 0.1 0.1";
screen -dmS d bash -c "tf3;sleep 2;export CUDA_VISIBLE_DEVICES='3';python quickstart_hessian.py 0.1 1";



screen -dmS e bash -c "tf3;sleep 2;export CUDA_VISIBLE_DEVICES='3';python quickstart_hessian.py 0.01 0";
screen -dmS f bash -c "tf3;sleep 2;export CUDA_VISIBLE_DEVICES='4';python quickstart_hessian.py 0.01 0.01";
screen -dmS g bash -c "tf3;sleep 2;export CUDA_VISIBLE_DEVICES='6';python quickstart_hessian.py 0.01 0.1";
screen -dmS h bash -c "tf3;sleep 2;export CUDA_VISIBLE_DEVICES='7';python quickstart_hessian.py 0.01 1";




