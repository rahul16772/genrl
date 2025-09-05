import gc
import os
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

# --- Optional dependencies ---
try:
    from vllm import LLM, SamplingParams
    _VLLM_AVAILABLE = True
except Exception:
    LLM, SamplingParams = None, False
    _VLLM_AVAILABLE = False

try:
    # ADDED: Adafactor for the new optimizer option
    from transformers import BitsAndBytesConfig, Adafactor
    import bitsandbytes as bnb
    _BNB_AVAILABLE = True
except Exception:
    BitsAndBytesConfig, bnb, Adafactor = None, None, None
    _BNB_AVAILABLE = False

from genrl.data import DataManager
from genrl.logging_utils.ml_logger import LoggerMixin
from genrl.rewards import RewardManager
from genrl.state import GameState
from genrl.trainer import TrainerModule
from genrl.trainer.trainer_utils import DTYPE_MAP



def create_reference_model(
    model: torch.nn.Module
) -> torch.nn.Module:
    ref_model = deepcopy(model)
    for param in model.parameters():
        param.requires_grad = False
    return ref_model.eval()

@dataclass
class GRPOTrainerConfig:
    # Official fields
    epsilon: float = 0.2
    epsilon_high: float = 0.28
    beta: float = 0.0
    temperature: float = 1.0
    dtype: str = "float32"
    enable_gradient_checkpointing: bool = True
    max_new_tokens: int = 256
    num_generations: int = 2
    learning_rate: float = 1e-5
    top_p: float = 1.0
    top_k: int | None = None
    min_p: float | None = None
    repetition_penalty: float = 1.0
    num_iterations: int = 1

    # --- Backend Switches ---
    use_vllm: bool = False
    use_bitsandbytes: bool = False

    # --- Optimizer Config ---
    optimizer: Dict[str, Any] = field(default_factory=lambda: {
        "name": "adamw", "weight_decay": 0.01
    })

    # --- vLLM Config ---
    vllm: Dict[str, Any] = field(default_factory=lambda: {
        "gpu_memory_utilization": 0.9, "tensor_parallel_size": 1
    })

    # --- BitsAndBytes Config ---
    bitsandbytes: Dict[str, Any] = field(default_factory=lambda: {
        "load_in_4bit": True, "load_in_8bit": False, "bnb_4bit_compute_dtype": "bfloat16", "bnb_4bit_quant_type": "nf4", "bnb_4bit_use_double_quant": True
    })


class GRPOLanguageTrainerModule(TrainerModule, LoggerMixin):
    """
    Trainer for the Group Relative Policy Optimization (GRPO) method.
    Implements the TrainerModule interface defined in base_trainer.py.
    """

    def __init__(self, models: List[Any], config: GRPOTrainerConfig, **kwargs):
        """
        Initialize the GRPO trainer module.
        """
        if not models or len(models) < 1:
            raise ValueError("At least one model must be provided")

        self.model = models[0]
        self.args = config

        # --- THIS IS THE NEW FEATURE ---
        # If the optimizer name is 'choose', prompt the user to select one.
        if self.args.optimizer.get("name") == "choose":
            self._manually_select_optimizer()
        # -----------------------------
        
        print("\n--- GRPO Trainer Backend Configuration ---")
        if self.args.use_vllm:
            print("⚡️ vLLM Engine: Enabled for fast generation.")
        elif self.args.use_bitsandbytes:
            print("✅ BitsAndBytes: Enabled for model quantization.")
        else:
            print("⚪️ Standard Mode: Using default Hugging Face backend.")
        print(f"⚙️ Optimizer: Using {self.args.optimizer.get('name', 'adamw')}")
        print("-----------------------------------------\n")

        self.processing_class = kwargs.get("processing_class", None)
        self.callbacks = kwargs.get("callbacks", [])
        self.save_dir = kwargs.get("log_dir", "./outputs")
        self.global_step = 0
        assert self.args.num_generations > 1, "GRPO requires num_generations > 1"
        self.dtype = DTYPE_MAP[self.args.dtype]
        self.enable_gradient_checkpointing = self.args.enable_gradient_checkpointing

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        # MODIFIED: Initialization order adjusted for new features
        self._initialize_model()
        self._initialize_tokenizers()
        self._initialize_optimizer()
        self._initialize_metrics()
        self._initialize_generation_config()
        self._initialize_vllm_if_enabled()
        self.init_tracker(self.save_dir, log_with=kwargs.get("log_with", None))

        print(f"✅ VERIFICATION: Optimizer object created is of type: {type(self.optimizer)}")

    def _manually_select_optimizer(self):
        """Presents a menu to the user to select an optimizer at runtime."""
        print("\n--- Please Select an Optimizer ---")
        print("1: AdamW (Default, Good Performance)")
        print("2: AdamW_8bit (Best for VRAM Savings)")
        print("3: SGD (Low VRAM, can be slower)")
        print("4: Adafactor (Memory Efficient)")
        
        choice_map = {
            "1": "adamw",
            "2": "adamw_8bit",
            "3": "sgd",
            "4": "adafactor",
        }
        
        while True:
            choice = input("Enter your choice (1-4): ")
            if choice in choice_map:
                chosen_optimizer = choice_map[choice]
                self.args.optimizer["name"] = chosen_optimizer
                print(f"✅ You have selected: {chosen_optimizer}")
                return
            else:
                print("❌ Invalid choice. Please enter a number between 1 and 4.")

    def _initialize_model(self):
        """Initializes the training model, applying BitsAndBytes if configured."""
        model_id = self.model.config._name_or_path
        
        if self.args.use_bitsandbytes:
            if not _BNB_AVAILABLE: raise ImportError("`use_bitsandbytes=True` but bitsandbytes is not installed.")
            if not torch.cuda.is_available(): raise RuntimeError("BitsAndBytes requires a CUDA-enabled GPU.")
            bnb_config = self.args.bitsandbytes
            compute_dtype = DTYPE_MAP.get(bnb_config["bnb_4bit_compute_dtype"], torch.bfloat16)
            quant_cfg = BitsAndBytesConfig(
                load_in_4bit=bnb_config["load_in_4bit"], load_in_8bit=bnb_config["load_in_8bit"], bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_quant_type=bnb_config["bnb_4bit_quant_type"], bnb_4bit_use_double_quant=bnb_config["bnb_4bit_use_double_quant"],
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id, quantization_config=quant_cfg, device_map="auto", trust_remote_code=True
            )
        else:
            self.model = self.model.to(device=self.device, dtype=self.dtype)
            if self.enable_gradient_checkpointing and hasattr(self.model, 'gradient_checkpointing_enable'):
                self.model.gradient_checkpointing_enable()

        if self.args.beta > 0.0:
            self.ref_model = create_reference_model(self.model).to(device=self.device, dtype=self.dtype)
        else:
            self.ref_model = None

    def _initialize_optimizer(self):
        """Initializes the optimizer based on the config."""
        optimizer_name = self.args.optimizer.get("name", "adamw").lower()
        weight_decay = self.args.optimizer.get("weight_decay", 0.01)
        lr = self.args.learning_rate

        match optimizer_name:
            case "adamw_8bit":
                if not self.args.use_bitsandbytes:
                    print("Warning: `use_bitsandbytes` is false. Falling back to standard AdamW optimizer.")
                    self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
                elif not _BNB_AVAILABLE:
                    raise ImportError("Cannot use 8-bit AdamW, bitsandbytes is not installed.")
                else:
                    self.optimizer = bnb.optim.AdamW8bit(self.model.parameters(), lr=lr, weight_decay=weight_decay)
            case "adamw":
                self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
            case "sgd":
                self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, momentum=0.9)
            case "adafactor":
                if Adafactor is None:
                    raise ImportError("Cannot use Adafactor, please ensure transformers is installed correctly.")
                self.optimizer = Adafactor(self.model.parameters(), lr=lr, weight_decay=weight_decay, scale_parameter=False, relative_step=False)
            case _:
                # Fallback to the original official optimizer
                print(f"Warning: Unknown optimizer '{optimizer_name}'. Falling back to default Adam.")
                self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
    
    def _initialize_vllm_if_enabled(self):
        self.vllm_engine = None
        if not self.args.use_vllm: return
        if not _VLLM_AVAILABLE: raise ImportError("`use_vllm=True` but vLLM isn't installed.")
        
        model_name = self.model.config._name_or_path
        vllm_config = self.args.vllm
        
        gpu_memory_utilization = vllm_config.get("gpu_memory_utilization", 0.9)
        if self.args.use_bitsandbytes:
            print("INFO: Hybrid mode detected. Automatically reducing vLLM memory utilization to prevent OOM.")
            gpu_memory_utilization = 0.5

        self.vllm_engine = LLM(
            model=model_name, trust_remote_code=True,
            tensor_parallel_size=vllm_config.get("tensor_parallel_size", 1),
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=4096, dtype=self.args.dtype,
        )

    # --- The rest of the file is your official version, UNTOUCHED ---
    def _initialize_tokenizers(self):
        """Initialize tokenizers for the model and reward models."""
        if self.processing_class is None:
            self.processing_class = AutoTokenizer.from_pretrained(
                self.model.config._name_or_path, padding_side="left"
            )
        if self.processing_class.pad_token is None:
            self.processing_class.pad_token = self.processing_class.eos_token

    def _initialize_metrics(self):
        """Initialize metrics tracking for training and evaluation."""
        self._metrics = {"train": defaultdict(list), "eval": defaultdict(list)}
        self._total_train_tokens = 0

    def _initialize_generation_config(self):
        # Set generation config
        self.generation_config = GenerationConfig(
            max_new_tokens=self.args.max_new_tokens,
            do_sample=True,
            pad_token_id=self.processing_class.pad_token_id,
            bos_token_id=self.processing_class.bos_token_id,
            eos_token_id=self.processing_class.eos_token_id,
            temperature=self.args.temperature,
            top_p=self.args.top_p,
            top_k=self.args.top_k,
            min_p=self.args.min_p,
            repetition_penalty=self.args.repetition_penalty,
        )

    def _process_inputs(self, inputs, with_template=True, for_training=False):
        if hasattr(inputs, "to_dict"):
            inputs = [dict(inputs[i]) for i in range(len(inputs))]
        elif isinstance(inputs, dict):
            inputs = [inputs]

        if (
            with_template
        ):
            if for_training:
                templated_prompts = []
                for item in inputs:
                    for _ in range(self.args.num_generations):
                        templated_prompts.append(
                            self.processing_class.apply_chat_template(item["prompt"], tokenize=False, add_generation_prompt=True)
                        )
            else:
                templated_prompts = [
                    self.processing_class.apply_chat_template(item["prompt"], tokenize=False, add_generation_prompt=True)
                    for item in inputs
                ]
        else:
            if for_training:
                templated_prompts = []
                for generations in inputs:
                    for output in generations:
                        templated_prompts.append(output)
            else:
                templated_prompts = [item[0] for item in inputs]

        input_tokens = self.processing_class(
            text=templated_prompts, return_tensors="pt", padding=True, truncation=True
        )
        return input_tokens

    def generate(
        self, inputs: Any, return_completion_ids: bool = False, stage=0
    ) -> Any:
        """
        Generate outputs from the model for the given inputs.
        """
        if self.vllm_engine:
            prompts = [self.processing_class.apply_chat_template(item["prompt"], tokenize=False, add_generation_prompt=True) for item in inputs]
            sampling = SamplingParams(n=self.args.num_generations, temperature=self.args.temperature, top_p=self.args.top_p, max_tokens=self.args.max_new_tokens)
            outputs = self.vllm_engine.generate(prompts, sampling_params=sampling, use_tqdm=False)
            return [[o.text for o in out.outputs] for out in outputs]

        input_tokens = self._process_inputs(inputs)
        rollout, rollout_ids = ([], [])
        for _ in range(self.args.num_generations):
            with torch.no_grad():
                outputs = self.model.generate(
                    input_tokens.input_ids.to(self.device),
                    attention_mask=input_tokens.attention_mask.to(self.device, dtype=self.dtype),
                    generation_config=self.generation_config,
                )

            prompt_length = input_tokens.input_ids.size(1)
            completion_ids = outputs[:, prompt_length:]

            completions = self.processing_class.batch_decode(
                completion_ids, skip_special_tokens=True
            )

            if len(rollout) == 0:
                rollout = [[comp] for comp in completions]
                if return_completion_ids:
                    rollout_ids = [[comp] for comp in completion_ids]
            else:
                for idx, comp in enumerate(completions):
                    rollout[idx].append(comp)
                    if return_completion_ids:
                        rollout_ids[idx].append(completion_ids[idx])
        if return_completion_ids:
            return rollout, rollout_ids
        else:
            return rollout

    def _get_per_token_logps(self, model, input_ids, attention_mask, logits_to_keep):
        """Get the per-token log probabilities for the input tokens."""
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        logits = outputs.logits
        logits = logits[:, :-1, :]

        loss_mask = (
            attention_mask[:, -logits_to_keep:].to(device=logits.device, dtype=logits.dtype).contiguous()
        )
        labels = input_ids[:, -logits_to_keep:].contiguous()
        logits = logits[:, -logits_to_keep:].contiguous()
        logits = logits / self.args.temperature
        logits_shape = logits.shape
        token_log_probs = -torch.nn.functional.cross_entropy(
            logits.view(-1, logits_shape[-1]),
            labels.view(-1),
            reduction="none",
        ).view(logits_shape[0], logits_shape[1])
        token_log_probs = (
            token_log_probs * loss_mask
            + (1.0 - loss_mask) * torch.finfo(logits.dtype).min
        )
        return token_log_probs

    def compute_loss(
        self, model, inputs, mode="train", return_metrics=False
    ):
        """Compute the GRPO loss."""
        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = (
            inputs["completion_ids"],
            inputs["completion_mask"],
        )
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1).to(self.model.device)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1).to(
            self.model.device, dtype=self.dtype
        )
        logits_to_keep = completion_ids.size(1)

        per_token_logps = self._get_per_token_logps(
            model, input_ids, attention_mask, logits_to_keep
        )

        if self.args.beta != 0.0:
            if self.ref_model is not None:
                ref_per_token_logps = self._get_per_token_logps(
                    self.ref_model, input_ids, attention_mask, logits_to_keep
                )
            else:
                ref_per_token_logps = per_token_logps.clone()

            per_token_kl = (
                torch.exp(ref_per_token_logps - per_token_logps)
                - (ref_per_token_logps - per_token_logps)
                - 1
            )

        advantages = inputs["advantages"]
        old_per_token_logps = (
            inputs["old_per_token_logps"]
            if self.args.num_iterations > 1
            else per_token_logps.detach()
        )

        coef_1 = torch.exp(per_token_logps - old_per_token_logps)
        coef_2 = torch.clamp(
            coef_1,
            1 - self.args.epsilon,
            1 + self.args.epsilon_high if self.args.epsilon_high is not None else self.args.epsilon,
        )
        advantages = advantages.unsqueeze(dim=-1)

        per_token_loss1 = coef_1 * advantages
        per_token_loss2 = coef_2 * advantages
        per_token_loss = -torch.min(per_token_loss1, per_token_loss2)

        if self.args.beta != 0.0:
            per_token_loss = per_token_loss + self.args.beta * per_token_kl

        loss = (per_token_loss * completion_mask).sum() / completion_mask.sum()

        mean_kl = None
        if self.args.beta != 0.0:
            mean_kl = (per_token_kl * completion_mask).sum() / completion_mask.sum()
            self._metrics[mode]["kl"].append(mean_kl.item())

        is_clipped = (per_token_loss1 < per_token_loss2).float()
        clip_ratio = (is_clipped * completion_mask).sum() / completion_mask.sum()
        self._metrics[mode]["clip_ratio"].append(clip_ratio.item())
        self._metrics[mode]["loss"].append(loss.item())

        metrics = {
            "loss": loss.item(),
            "kl": mean_kl.item() if self.args.beta != 0.0 else None,
            "clip_ratio": clip_ratio.item(),
        }

        if return_metrics:
            return loss, metrics
        else:
            return loss

    def train(
        self, state: GameState, data_manager: DataManager, reward_manager: RewardManager
    ) -> None:
        """
        Train the model using the given game state and reward manager.
        """
        self.model.train()
        global_step = self.global_step
        for stage in range(state.stage):
            global_step = self.step(
                stage, state, data_manager, reward_manager, global_step
            )
        self.global_step = global_step
        self.model.eval()

    def step(
        self,
        stage: int,
        state: GameState,
        data_manager: DataManager,
        reward_manager: RewardManager,
        global_step: int,
    ) -> int:
        global_step += 1

        stage_inputs = state.get_stage_state(stage)
        stage_inputs, index_mapping = data_manager.prepare_input(stage_inputs, stage)
        assert stage_inputs is not None, f"No inputs found for stage {stage}"
        stage_actions = state.get_stage_actions(stage)
        stage_outputs = [
            stage_actions[index_mapping[idx][0]][index_mapping[idx][1]][
                index_mapping[idx][2]
            ]
            for idx, _ in enumerate(index_mapping)
        ]
        assert stage_outputs is not None, f"No outputs found for stage {stage}"

        model_inputs = {}
        processed_inputs = self._process_inputs(stage_inputs, for_training=True)
        model_inputs["prompt_ids"], model_inputs["prompt_mask"] = (
            processed_inputs.input_ids.to(self.model.device),
            processed_inputs.attention_mask.to(self.model.device, dtype=self.dtype),
        )
        processed_outputs = self._process_inputs(
            stage_outputs, with_template=False, for_training=True
        )
        model_inputs["completion_ids"], model_inputs["completion_mask"] = (
            processed_outputs.input_ids.to(self.model.device),
            processed_outputs.attention_mask.to(self.model.device, dtype=self.dtype),
        )

        rewards = reward_manager[stage]
        rewards = [
            rewards[index_mapping[idx][0]][index_mapping[idx][1]][index_mapping[idx][2]]
            for idx, _ in enumerate(index_mapping)
        ]
        assert rewards is not None, f"No rewards found for stage {stage}"
        rewards = torch.tensor(rewards)

        with torch.no_grad():
            advantages = rewards - rewards.mean(dim=1, keepdim=True)
            if rewards.shape[1] > 1:
                advantages /= rewards.std(dim=1, keepdim=True) + 1e-8
        advantages = torch.flatten(advantages).to(self.model.device, dtype=self.dtype)

        model_inputs["advantages"] = advantages.squeeze(dim=-1)
        model_inputs["old_per_token_logps"] = None

        loss = self.compute_loss(self.model, model_inputs)

        loss.backward()
        self.optimizer.step()
        self.model.zero_grad()

        metrics = {"train/loss": loss.cpu().mean().item()}
        metrics.update({"train/rewards": rewards.cpu().mean().item()})
        self.log(metrics, global_step)

        self.cleanup_step()

        return global_step

    @torch.no_grad()
    def evaluate(
        self, state: GameState, data_manager: DataManager, reward_manager: RewardManager
    ):
        pass
    
    def save(self, save_dir: str) -> None:
        """
        Save the model and trainer state to the given directory.
        """
        os.makedirs(save_dir, exist_ok=True)
        
        if hasattr(self.model, 'save_pretrained'):
            self.model.save_pretrained(save_dir)
        
        if hasattr(self.processing_class, 'save_pretrained'):
            self.processing_class.save_pretrained(save_dir)

        torch.save(
            {
                "metrics": self._metrics,
                "total_train_tokens": self._total_train_tokens,
                "generation_config": self.generation_config,
            },
            os.path.join(save_dir, "trainer_state.pt"),
        )

    @classmethod
    def load(cls, load_dir: str, **kwargs) -> "GRPOLanguageTrainerModule":
        """
        Load a trainer module from the given directory.
        """
        model = AutoModelForCausalLM.from_pretrained(load_dir)
        config = kwargs.pop('config', GRPOTrainerConfig())
        trainer = cls([model], config=config, **kwargs)
        state_path = os.path.join(load_dir, "trainer_state.pt")
        if os.path.exists(state_path):
            trainer_state = torch.load(state_path)
            trainer._metrics = trainer_state["metrics"]
            trainer._total_train_tokens = trainer_state["total_train_tokens"]
            trainer.generation_config = trainer_state["generation_config"]
        return trainer

    def cleanup_step(self):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif torch.backends.mps.is_available():
            torch.mps.empty_cache()
        gc.collect()

    def cleanup(self):
        self.cleanup_trackers()

