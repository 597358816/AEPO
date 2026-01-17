set -x

MODEL_PATH=Qwen/Qwen2.5-math-7B  # replace it with your local file path

FORMAT_PROMPT="""You FIRST think about the reasoning process as an internal monologue and then provide the final answer.
 The reasoning process MUST BE enclosed within <think> </think> tags. The final answer MUST BE put in \boxed{}."""

python3 -m verl.trainer.main \
    config=config.yaml \
    data.train_files=./train_dapo.parquet \
    data.val_files=hiyouga/math12k@test \
    data.format_prompt="${FORMAT_PROMPT}" \
    algorithm.disable_kl=true \
    worker.actor.model.model_path=${MODEL_PATH} \
    trainer.experiment_name=test \
    trainer.algorithm=DCPO \
    trainer.entropy_base=0.5 \
    trainer.n_gpus_per_node=8 \
    trainer.total_episodes=10

    #trainer.algorithm=GRPO #default is GRPO
    #trainer.entropy_base is the entropy threshold H.
    #trainer.reinforce_sample_num is the sample for entropy control, which should be set as small as possible when entropy is controllable.



