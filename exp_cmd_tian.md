
# Commands for running some experimentations using CLIP

1. Generate the textual embeddings for ActivityNet-v1.3 200 class labels
   * using the prompt text `a video of a person {CLASS}.`
   * using CLIP pretrained model `ViT-L/14`
   * run `sh scripts/run_test_zeroshot_txtemd_tian.sh configs/anet/anet_zero_shot.yaml exp/k400/ViT-L/14/f8/last_model.pt` to get the textual embeddings of the 200 class labels
   * the generated textual embeddings is saved to `text_feats_anet_ViT-L14.pt`
   * run `python correlation_of_label_embeddings.py` to plot the correlation of the textual embeddings


2. Run k-way classification to get the zero-shot performance on the sampled ActivityNet dataset
   * `sh scripts/run_test_zeroshot_tian.sh configs/anet/anet_zero_shot.yaml ckpt/k400-vitl-14-f8.pt`
   
3. Test the idea of `threshold max of softmax` and `thereshold entropy` for seen vs. unseen
   * using the pretrained model (trained on Kinetics400) for seen vs. unseen classification on sampled ActivityNet dataset
   * 


```sh
# On Kinetics-600: manually calculating the mean accuracy with standard deviation of three splits.
sh scripts/run_test.sh  configs/k600/k600_zero_shot_split1.yaml exp/k400/ViT-L/14/f8/last_model.pt
sh scripts/run_test.sh  configs/k600/k600_zero_shot_split2.yaml exp/k400/ViT-L/14/f8/last_model.pt
sh scripts/run_test.sh  configs/k600/k600_zero_shot_split3.yaml exp/k400/ViT-L/14/f8/last_model.pt
```