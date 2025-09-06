# train source
# for i in {0..11}; do
#     python3.11 ./SHOT_for_source_training/train_source.py --dset office-caltech --s $i --max_epoch 100 --trte val --gpu_id 0 --output ckps/source/uda
# done
# $i is the i'th task

# for i in {0..17}; do
#     python ./SHOT_for_source_training/train_source.py --dset domainnet  --s $i --max_epoch 100 --trte val --gpu_id 0 --output ckps/source/uda
# done
 
# clipart infograph painting quickdraw real sketch

run
for domain in clipart painting quickdraw real sketch; do
    for i in 0 1 2; do
        echo "######Running with dataset.domain=${domain}_${i}############"
        python test_HEnsemble.py dataset=domainnet model.name=resnet18 model.hidden_dim=256 seed=49999 output_dir=./logs dataset.domain=${domain}_${i} eval_batch_size=64 max_epochs=8 few_shot_num=4 checkpoint_dir=/2025May/ckps/source/uda val_sample_num=2000 if_use_shot_model=True
    done
done

# run
# for domain in amazon caltech dslr webcam; do
#     for i in 0 1 2; do
#         echo "######Running with dataset.domain=${domain}_${i}############"
#         python test_HEnsemble.py dataset=office-caltech model.name=resnet18 model.hidden_dim=256 seed=2023 output_dir=./logs dataset.domain=${domain}_${i} eval_batch_size=64 max_epochs=8 few_shot_num=8 checkpoint_dir=/2025May/ckps/source/uda val_sample_num=2000 if_use_shot_model=True
#     done
# done
awk 'NR % 2 == 1 {sum += $1; count++} END {if (count > 0) print "Mean =", sum/count}' results.txt
     