import json
import os

from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy

from drift_detector import DriftDetector
from env.path_selection_env import PathSelectionEnv
from generator import Generator
from maskable_dqn import MaskableDQN


def make_env(traffic_name):

    def _init():
        g = Generator()
        g.load_traffic(f"{traffic_name}.csv")
        d = DriftDetector(threshold=24)
        env = PathSelectionEnv(env_first_hour=0, generator=g, detector=d, phase="testing", dynamic_end_times=False)
        return env

    return _init

def create_env(traffic_name: str):
    env = make_vec_env(make_env(traffic_name), n_envs=1, seed=43)
    return env

def test_all_durations(job_id, traffic_name):

    # job_id = int(input("Enter job id for model loading: "))
    # traffic_name = input("Enter traffic name: ")

    model_path = f"./models/{job_id}/"
    env = create_env(traffic_name)
    model = MaskableDQN.load(model_path + "final_model", env=env)
    test_mean_reward, test_episode_lengths = evaluate_policy(
        model,  # type: ignore[arg-type]
        env,
        n_eval_episodes=24,
        deterministic=True,
        warn=False,
    )

    result_path = f"./results-with-diff-traffic/{job_id}/{traffic_name}"
    os.makedirs(result_path, exist_ok=True)
    metrics = env.env_method("return_metrics")[0]  # get metrics from the first env
    with open(f"{result_path}/testing_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    test_results = env.env_method("return_hourly_results")[0]
    with open(f"{result_path}/testing_hourly_results.json", "w") as f:
        json.dump(test_results, f, indent=4)

    cpu_results = env.env_method("return_cpu_metrics")[0]
    with open(f"{result_path}/testing_cpu_results.json", "w") as f:
        json.dump(cpu_results, f, indent=4)

if __name__ == "__main__":
    duration = "week"
    for job_id in ["RANDOM"]:
        for i in range(0, 12):
            print(f"Testing job {job_id} with traffic {duration}_{i}")
            test_all_durations(job_id, f"{duration}_{i}")