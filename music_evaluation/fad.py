from frechet_audio_distance import FrechetAudioDistance
import sys

# Compute FAD distance between the ground truth dataset and sample dataset
# Pretty slow depends on the speed
# Save embeddings.npy for future fast usage
#
# Usage: python fad.py background_dir_path eval_dir_path
# Feel free the change the embedding path in the code
# More info about FrechetAudioDistance: https://github.com/gudgud96/frechet-audio-distance

if __name__ == "__main__":
    # to use `vggish`
    frechet = FrechetAudioDistance(
        model_name="vggish",
        use_pca=False,
        use_activation=False,
        verbose=False
    )
    # # to use `PANN`
    # frechet = FrechetAudioDistance(
    #     model_name="pann",
    #     use_pca=False,
    #     use_activation=False,
    #     verbose=False
    # )

    background_dir = sys.argv[1]
    eval_dir = sys.argv[2]

    background_embds_path = "./ground_truth_embeddings.npy"
    eval_embds_path = "./eval_embeddings.npy"

    fad_score = frechet.score(background_dir, eval_dir,
                              background_embds_path=background_embds_path,
                              eval_embds_path=eval_embds_path,dtype="float32")

    print(fad_score)