"""SO101 Three Cubes policy server with an explicit six-dimensional schema."""

from dataclasses import dataclass
from pathlib import Path

import draccus

from cosmos_policy.experiments.robot.aloha.deploy import DeployConfig, PolicyServer
from cosmos_policy.experiments.robot.aloha.so101_schema import (
    load_schema,
    print_schema_summary,
    schema_digest,
)
from cosmos_policy.utils.utils import set_seed_everywhere

HERE = Path(__file__).parent


@dataclass
class SO101DeployConfig(DeployConfig):
    suite: str = "aloha"  # SO101 uses the same 11-token/three-camera latent layout, not ALOHA actions.
    config: str = "cosmos_predict2_2b_480p_three_cubes_so101_posttrain__inference_only"
    action_dim: int = 6
    chunk_size: int = 30
    num_open_loop_steps: int = 30
    num_wrist_images: int = 2
    num_third_person_images: int = 1
    use_proprio: bool = True
    normalize_proprio: bool = True
    unnormalize_actions: bool = True
    dataset_stats_path: str = str(HERE / "three_cubes_so101_dataset_statistics.json")
    t5_text_embeddings_path: str = "/data/rxhuang/three_cubes_1/t5_embeddings.pkl"


class SO101PolicyServer(PolicyServer):
    def __init__(self, cfg):
        self.so101_schema = load_schema()
        super().__init__(cfg)

    def get_server_action(self, payload):
        supplied = payload.get("action_schema_sha256")
        expected = schema_digest(self.so101_schema)
        if supplied != expected:
            raise ValueError(f"Action schema digest mismatch: client={supplied}, server={expected}")
        payload = dict(payload)
        payload.pop("action_schema_sha256")
        return super().get_server_action(payload)


@draccus.wrap()
def deploy(cfg: SO101DeployConfig) -> None:
    if cfg.action_dim != 6:
        raise ValueError("SO101 action_dim must be 6")
    print_schema_summary(load_schema())
    set_seed_everywhere(cfg.seed)
    SO101PolicyServer(cfg).run(cfg.host, cfg.port)


if __name__ == "__main__":
    deploy()
