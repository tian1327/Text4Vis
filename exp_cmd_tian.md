
# Commands for running some experimentations using CLIP

1. Generate the textual embeddings for ActivityNet-v1.3 200 class labels
   * using the prompt text `a video of a person {CLASS}.`
   * `sh scripts/run_train_tian.sh  configs/k400/k400_train_rgb_vitl-14-f8.yaml`
   * 
   * `sh scripts/run_test_zeroshot_tian.sh  configs/anet/anet_zero_shot.yaml exp/k400/ViT-L/14/f8/last_model.pt`


2. Test the idea of `threshold max of softmax` and `thereshold entropy` using the pretrained model (trained on Kinetics400) for seen vs. unseen classification on Kinetics600 (test set)


```sh
# On Kinetics-600: manually calculating the mean accuracy with standard deviation of three splits.
sh scripts/run_test.sh  configs/k600/k600_zero_shot_split1.yaml exp/k400/ViT-L/14/f8/last_model.pt
sh scripts/run_test.sh  configs/k600/k600_zero_shot_split2.yaml exp/k400/ViT-L/14/f8/last_model.pt
sh scripts/run_test.sh  configs/k600/k600_zero_shot_split3.yaml exp/k400/ViT-L/14/f8/last_model.pt
```