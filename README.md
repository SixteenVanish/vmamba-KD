# cmd
cd classification
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nnodes=1 --node_rank=0 --nproc_per_node=2 --master_addr="127.0.0.1" \
--master_port=29600 main_KD.py \
--cfg path_of_config \
--batch-size 128 --data-path path_of_data \
--output path_of_output

path_of_data can be ‘/ibex/user/lix0k/code/mmpretrain/data/imagenet’

# configs of best results:
## B/C/dt ATloss
  classification/configs/kd_in100/0.1CLS_0.9KL_BCdt/BorCorDelta/vmamba_tiny-base_0.1CLS_0.9KL_S3-500B.yaml
  classification/configs/kd_in100/0.1CLS_0.9KL_BCdt/BorCorDelta/vmamba_tiny-base_0.1CLS_0.9KL_S3-500C.yaml
  classification/configs/kd_in100/0.1CLS_0.9KL_BCdt/BorCorDelta/vmamba_tiny-base_0.1CLS_0.9KL_S3-750dt.yaml
## B&C&dt ATloss
  classification/configs/kd_in100/0.1CLS_0.9KL_BCdt/B-C-dt/vmamba_tiny-base_0.1CLS_0.9KL_S3-100B-100C-100dt.yaml 

如果运行某些config文件失败的话，可能原因是：
1.需要设置数据集，在config文件中加上，BASE: ["configs/data_cfg/in100.yaml"]
2.后期为了不同的STAGE的loss weight，部分loss weight的设置是WEIGHT_STAGE[0.,0.,0.,0.,False]而不是单个浮点值
3.cd classification后，cmd中的config文件名：configs/...而不是classification/configs/...
