screen -dmS a bash -c "export CUDA_VISIBLE_DEVICES=0;
                        python quickstart_hessian.py 0 1 1";
screen -dmS b bash -c "export CUDA_VISIBLE_DEVICES=6;
                        python quickstart_hessian.py 1 1 1";

screen -dmS c bash -c "export CUDA_VISIBLE_DEVICES=1;
                        python quickstart_hessian.py 0 3 1";
screen -dmS d bash -c "export CUDA_VISIBLE_DEVICES=7;
                        python quickstart_hessian.py 1 3 1";

screen -dmS e bash -c "export CUDA_VISIBLE_DEVICES=2;
                        python quickstart_hessian.py 0 5 1";
screen -dmS f bash -c "export CUDA_VISIBLE_DEVICES=2;
                        python quickstart_hessian.py 1 5 1";

screen -dmS g bash -c "export CUDA_VISIBLE_DEVICES=3;
                        python quickstart_hessian.py 0 1 2";
screen -dmS h bash -c "export CUDA_VISIBLE_DEVICES=3;
                        python quickstart_hessian.py 1 1 2";

screen -dmS i bash -c "export CUDA_VISIBLE_DEVICES=4;
                        python quickstart_hessian.py 0 1 3";
screen -dmS j bash -c "export CUDA_VISIBLE_DEVICES=7;
                        python quickstart_hessian.py 1 1 3";





