import copy
import json
import multiprocessing as mp
from multiprocessing import Queue
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

from influence_benchmark.agent.agent import Agent
from influence_benchmark.backend.backend import Backend
from influence_benchmark.config.experiment_config import BaseExperimentConfig, ExpertIterationConfig, KTOConfig, OpenAIExpertIterationConfig
from influence_benchmark.data_root import PROJECT_DATA
from influence_benchmark.environment.environment import Environment
from influence_benchmark.environment.state import State
from influence_benchmark.environment_vectorized.assessor_model_vectorized import (
    VectorizedAssessorModel,
)
from influence_benchmark.environment_vectorized.environment_queue import (
    TrajectoryQueue,
)
from influence_benchmark.environment_vectorized.environment_vectorized import (
    VectorizedEnvironment,
)
from influence_benchmark.RL.base_iteration import BaseIteration
from influence_benchmark.root import PROJECT_ROOT
from influence_benchmark.utils.utils import set_all_seeds

ASSESSOR_MODEL_ATTRIBUTES = {
    "preference_model_vectorized": {
        "action_function": "add_preferences_to_states",
        "state_attribute": "preferences",
    },
    "influence_detector_model_vectorized": {
        "action_function": "add_influence_scores_to_states",
        "state_attribute": "influence_scores",
    },
    "character_vectorized": {
        "action_function": "add_char_responses_to_states",
        "state_attribute": "history",
    },
}


class VectorizedEnvironmentWithScoring(VectorizedEnvironment):
    def __init__(self, vectorized_environment: VectorizedEnvironment, assessor_model_name: str):
        if assessor_model_name not in ASSESSOR_MODEL_ATTRIBUTES:
            raise ValueError(f"Assessor model name {assessor_model_name} not found in function map.")

        # Add all attributes and methods from the parent class.
        for key, value in vectorized_environment.__dict__.items():
            setattr(self, key, value)

        self.assessor_model: VectorizedAssessorModel = getattr(self, assessor_model_name)

        self.assessor_model_function = ASSESSOR_MODEL_ATTRIBUTES[assessor_model_name]["action_function"]
        self.assessor_model_state_attribute = ASSESSOR_MODEL_ATTRIBUTES[assessor_model_name]["state_attribute"]

    def _build_state_from_trajectory(self, trajectory: dict[str, Any]) -> State:
        assert len(self.environments) == 1, "This method only works for a single environment."
        environment = self.environments[0]

        variables = {**environment.variables}
        conversation_history = trajectory["history"]
        state_name = "terminal_state" if trajectory["transition_probs"]["yes"] == 1.0 else "initial_state"

        return State(
            state_name,
            conversation_history,
            variables,
            trajectory["turn"],
            environment.state_config[state_name]["valid_transitions"],
            environment.state_config[state_name]["default_transition"],
            state_name == "terminal_state",
        )

    def get_states_with_assessments(self, trajectories: list[dict[str, Any]]) -> tuple[list[State], list[Any]]:
        assert len(self.environments) == 1, "This method only works for a single environment."

        # mean_noise = 0
        # std_noise = 0.005
        #
        # for param in self.assessor_model.backend.model.parameters():
        #     param.data += torch.normal(
        #         mean=mean_noise, std=std_noise, size=param.size()
        #     ).to(param.device)

        states = []

        for trajectory in trajectories:
            states.append(self._build_state_from_trajectory(trajectory))

            # Call the assessor model function on the assessor model
            getattr(self.assessor_model, self.assessor_model_function)(states[-1:])

        assessments = [getattr(state, self.assessor_model_state_attribute) for state in states]

        return states, assessments


class AssessorModelTest(BaseIteration):
    def __init__(self, assessor_model_name: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.assessor_model_name = assessor_model_name

    def score_trajectories(self, trajectories: list[dict[str, Any]], device: str) -> None:
        agent_config = self._load_agent_config()

        # Build the environment queue
        self.trajectory_queue.populate(
            iter_step=0, eval=False
        )
        generation_progress = mp.Value("i", 0)

        # Create the environment and agent
        vec_env, agent = self.create_environment_and_agent(
            device,
            shared_queue=self.trajectory_queue,
            progress=generation_progress,
            agent_config=agent_config,
            lora_path=self.lora_path,
        )

        vec_env_with_scoring = VectorizedEnvironmentWithScoring(vec_env, self.assessor_model_name)

        # Score the trajectories
        _, assessments = vec_env_with_scoring.get_states_with_assessments(trajectories)

        # "env_name": "smoking_subtle_weakness2_v2", "initial_state_id": 40, "trajectory_id": 0, "turn": 1,
        # attributes = ["env_name", "initial_state_id", "trajectory_id", "turn"]
        attributes = ["initial_state_id", "trajectory_id", "turn"]
        for trajectory, assessment in zip(trajectories, assessments):
            values = [trajectory[attr] for attr in attributes]
            log_string = "".join([f"{attr}: {value}, " for attr, value in zip(attributes, values)])
            log_string += f"{vec_env_with_scoring.assessor_model_state_attribute}: {assessment}"
            print(log_string)


def main(
    assessor_model_name: str,
    config_path: str,
    data_path: Path,
    num_trajectories_to_score: Optional[int] = None,
    env_class: Optional[str] = None,
    envs: Optional[list[str]] = None,
    gpu_subset: Optional[str] = None,
) -> None:
    # Load the model
    config = BaseExperimentConfig.load(config_path, gpu_subset=gpu_subset)
    config.num_envs_per_device = 1

    if not (isinstance(config, ExpertIterationConfig) or isinstance(config, KTOConfig) or isinstance(
         config, OpenAIExpertIterationConfig)):
        raise ValueError(f"Unknown experiment type: {type(config)}")

    if env_class is not None:
        config.env_class = env_class
    if envs is not None:
        config.envs = envs

    if config.seed is not None:
        set_all_seeds(config.seed)

    training_script_path = "Not needed for this experiment"

    experiment = AssessorModelTest(
        assessor_model_name=assessor_model_name,
        env_args=config.env_args,
        training_args=config.training_args,
        accelerate_config=config.accelerate_config if hasattr(config, "accelerate_config") else None,  # type: ignore
        script_path=training_script_path,
        model_names=config.model_names,
        frac_selected_trajs=config.frac_selected_trajs,
        iterations=config.iterations,
        run_name=config.run_name,
        devices=config.devices,
        log_to_wandb=config.log_to_wandb,
        seed=config.seed,
        final_reward=config.final_reward,
        override_initial_traj_path=config.override_initial_traj_path,
        pm_length_penalty=config.pm_length_penalty,
        traj_selection_level=config.traj_selection_level,
        timestamp=None,
        veto_level=config.veto_level,
        allow_negative_training_on_veto=config.allow_negative_training_on_veto,
        max_tokens_per_minute=config.max_tokens_per_minute,
        max_requests_per_minute=config.max_requests_per_minute,
        separate_agent_env_devices=config.separate_agent_env_devices,
        inference_quantization=config.inference_quantization,
    )

    # Load the trajectories
    trajectories = []

    with open(data_path, "r") as f:
        for line in f:
            if num_trajectories_to_score is not None and len(trajectories) >= num_trajectories_to_score:
                break
            trajectories.append(json.loads(line))

    experiment.score_trajectories(trajectories, "cuda:0")
    # experiment.launch()

    # Delete the result folder again since we don't leave any traces
    result_dir = Path(experiment.trajectory_dir)

    if result_dir.exists():
        for file in result_dir.iterdir():
            file.unlink()
        result_dir.rmdir()


if __name__ == "__main__":
    config = "KTO_medical.yaml"

    # Must be a key from 'ASSESSOR_MODEL_FUNCTION_MAP'
    assessor_model_name = "preference_model_vectorized"
    # assessor_model_name = "influence_detector_model_vectorized"
    # assessor_model_name = "character_vectorized"

    # Specify settings for generating trajectories
    env_class = "medical_chatbot"

    # envs = ["nudging-therapist"]
    # envs = ["smoking"]
    # envs = ["smoking_obvious_weakness"]
    # envs = ["smoking_subtle_weakness1"]
    # envs = ["smoking_subtle_weakness2"]
    # envs = ["smoking_subtle_weakness2_v2_cheating"]
    # envs = ["smoking_subtle_weakness2_v2"]
    envs = ["lying_docassist", "truthful_docassist"]

    data_path = (
        PROJECT_DATA
        / "trajectories/test_trajectories_preference_model_doctor2.jsonl"
        # / "trajectories/test_trajectories_character.jsonl"
    )
    num_trajectories_to_score = None

    main(
        assessor_model_name=assessor_model_name,
        config_path=config,
        data_path=data_path,
        num_trajectories_to_score=num_trajectories_to_score,
        env_class=env_class,
        envs=envs,
        gpu_subset=[0],
    )

    exit(0)
