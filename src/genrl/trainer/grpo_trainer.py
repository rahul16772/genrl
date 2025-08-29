import gc
import os
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

# Optional deps
try:
    from vllm import LLM, SamplingParams
    _VLLM_AVAILABLE = True
except ImportError:
    LLM, SamplingParams = None, None
    _VLLM_AVAILABLE = False

try:
    from transformers import BitsAndBytesConfig
    _BNB_AVAILABLE = True
except ImportError:
    BitsAndBytesConfig = None
    _BNB_AVAILABLE = False

from genrl.data import DataManager
from genrl.logging_utils.ml_logger import LoggerMixin
from genrl.rewards import RewardManager
from genrl.state import GameState
from genrl.trainer import TrainerModule
from genrl.trainer.trainer_utils import DTYPE_MAP


def create_reference_model(model: torch.nn.Module) -> torch.nn.Module:
    """Creates a frozen, deep-copied reference model for KL-divergence."""
    ref_model = deepcopy(model)
    for param in ref_model.parameters():
        param.requires_grad = False
    return ref_model.eval()


@dataclass
class GRPOTrainerConfig:
    # --- Existing GRPO/PPO Parameters ---
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
    top_k: Optional[int] = None
    min_p: Optional[float] = None
    repetition_penalty: float = 1.0
    num_iterations: int = 1

    # --- ADDED: Backend Selection Flags ---
    use_vllm: bool = False
    use_bitsandbytes: bool = False

    # --- ADDED: vLLM Specific Parameters ---
    vllm_gpu_memory_utilization: float = 0.9

    # --- ADDED: BitsAndBytes Specific Parameters ---
    bnb_load_in_4bit: bool = True
    bnb_load_in_8bit: bool = False
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_compute_dtype: str = "bfloat16"


class GRPOLanguageTrainerModule(TrainerModule, LoggerMixin):
    """
    Trainer for Group Relative Policy Optimization (GRPO) with optional vLLM rollouts
    and bitsandbytes quantization for the HF model.
    """

    def __init__(self, models: List[Any], config: GRPOTrainerConfig, **kwargs):
        if not models:
            raise ValueError("At least one model must be provided")

        self.model = models[0]
        self.args = config
        self.vllm_engine = None # Initialize vLLM engine as None

        self.processing_class = kwargs.get("processing_class", None)
        self.callbacks = kwargs.get("callbacks", [])
        self.save_dir = kwargs.get("log_dir", "./outputs")
        self.global_step = 0

        assert self.args.num_generations > 1, (
            f"For GRPO training, number of generations must be > 1, "
            f"got {self.args.num_generations}"
        )

        self.dtype = DTYPE_MAP[self.args.dtype]
        self.enable_gradient_checkpointing = self.args.enable_gradient_checkpointing

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        # --- MODIFIED: Enforce exclusive backend selection ---
        if self.args.use_vllm and self.args.use_bitsandbytes:
            raise ValueError(
                "\n\n❌ Both `use_vllm` and `use_bitsandbytes` are set to True in your config."
                "\nPlease enable only ONE of these backends at a time."
                "\n- Use vLLM for fast generation."
                "\n- Use BitsAndBytes for memory-efficient training."
            )

        # Emoji indicators for the selected backend
        print("\n--- GRPO Trainer Backend Configuration ---")
        if self.args.use_vllm:
            print("✅ vLLM Engine: Enabled for fast generation.")
        elif self.args.use_bitsandbytes:
            quant_mode = "4-bit" if self.args.bnb_load_in_4bit else "8-bit"
            print(f"✅ BitsAndBytes: Enabled for {quant_mode} model quantization.")
        else:
            print("ℹ️  Standard Hugging Face Backend: Enabled for training and generation.")
        print("-----------------------------------------\n")


        # Initialization sequence to handle optional backends
        self._initialize_model(self.enable_gradient_checkpointing)
        self._initialize_tokenizers()
        self._initialize_metrics()
        self._initialize_generation_config()

        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.args.learning_rate
        )

        self.init_tracker(self.save_dir, log_with=kwargs.get("log_with", None))

    def _initialize_model(self, enable_gradient_checkpointing: bool):
        """Handle model setup, including optional quantization and vLLM engine."""
        model_id = self.model.config._name_or_path

        # 1. Handle BitsAndBytes Quantization if enabled
        if self.args.use_bitsandbytes:
            if not _BNB_AVAILABLE:
                raise ImportError("`use_bitsandbytes` is true, but bitsandbytes is not installed.")
            if not torch.cuda.is_available():
                raise RuntimeError("BitsAndBytes requires a CUDA-enabled GPU.")

            compute_dtype = DTYPE_MAP.get(self.args.bnb_4bit_compute_dtype, torch.bfloat16)

            quantization_config = BitsAndBytesConfig(
                load_in_4bit=self.args.bnb_load_in_4bit,
                load_in_8bit=self.args.bnb_load_in_8bit,
                bnb_4bit_quant_type=self.args.bnb_4bit_quant_type,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=True,
            )
            # Replace the pre-loaded model with the quantized version
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                quantization_config=quantization_config,
                device_map="auto",
                trust_remote_code=True,
            )

        # 2. Move model to device (if not quantized) and set up gradient checkpointing
        if not self.args.use_bitsandbytes:
            self.model = self.model.to(device=self.device, dtype=self.dtype)

        if enable_gradient_checkpointing and hasattr(self.model, "gradient_checkpointing_enable"):
            self.model.gradient_checkpointing_enable()

        # 3. Handle vLLM Engine creation if enabled
        if self.args.use_vllm:
            if not _VLLM_AVAILABLE:
                raise ImportError("`use_vllm` is true, but vllm is not installed.")
            self.vllm_engine = LLM(
                model=model_id,
                gpu_memory_utilization=self.args.vllm_gpu_memory_utilization,
                tensor_parallel_size=torch.cuda.device_count(),
                trust_remote_code=True,
                dtype=self.args.dtype
            )

        # 4. Setup Reference Model
        if self.args.beta > 0.0:
            self.ref_model = create_reference_model(self.model).to(device=self.device, dtype=self.dtype)
        else:
            self.ref_model = None

    def _initialize_tokenizers(self):
        """Initialize tokenizers for the model."""
        if self.processing_class is None:
            self.processing_class = AutoTokenizer.from_pretrained(
                self.model.config._name_or_path, padding_side="left"
            )
        if self.processing_class.pad_token is None:
            self.processing_class.pad_token = self.processing_class.eos_token

    def _initialize_metrics(self):
        """Initialize metrics tracking."""
        self._metrics = {"train": defaultdict(list), "eval": defaultdict(list)}
        self._total_train_tokens = 0

    def _initialize_generation_config(self):
        """Set generation config for the standard transformers generate method."""
        self.generation_config = GenerationConfig(
            max_new_tokens=self.args.max_new_tokens,
            do_sample=True,
            pad_token_id=self.processing_class.pad_token_id,
            temperature=self.args.temperature,
            top_p=self.args.top_p,
            top_k=self.args.top_k,
            repetition_penalty=self.args.repetition_penalty,
        )

    def _process_inputs(self, inputs, with_template=True, for_training=False):
        """Processes dictionary inputs into tokenized tensors."""
        if hasattr(inputs, "to_dict"):
            inputs = [dict(inputs[i]) for i in range(len(inputs))]
        elif isinstance(inputs, dict):
            inputs = [inputs]

        if with_template:
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

    def generate(self, inputs: Any, return_completion_ids: bool = False, stage=0) -> Any:
        """
        Generate outputs. Uses vLLM if available, otherwise falls back to the
        standard (potentially quantized) Hugging Face model.
        """
        prompt_strings = [
            self.processing_class.apply_chat_template(
                item["prompt"], tokenize=False, add_generation_prompt=True
            ) for item in inputs
        ]

        # 1. vLLM Path
        if self.vllm_engine:
            sampling_params = SamplingParams(
                n=self.args.num_generations,
                temperature=self.args.temperature,
                top_p=self.args.top_p,
                max_tokens=self.args.max_new_tokens,
            )
            vllm_outputs = self.vllm_engine.generate(prompt_strings, sampling_params, use_tqdm=False)
            
            rollout = [[comp.text for comp in out.outputs] for out in vllm_outputs]
            if return_completion_ids:
                rollout_ids = [[torch.tensor(comp.token_ids) for comp in out.outputs] for out in vllm_outputs]
                return rollout, rollout_ids
            return rollout

        # 2. BitsAndBytes / Standard HF Path
        else:
            input_tokens = self.processing_class(
                prompt_strings, return_tensors="pt", padding=True
            ).to(self.device)

            rollout, rollout_ids = [], []
            for _ in range(self.args.num_generations):
                with torch.no_grad():
                    outputs = self.model.generate(
                        **input_tokens,
                        generation_config=self.generation_config,
                    )
                prompt_length = input_tokens.input_ids.size(1)
                completion_ids = outputs[:, prompt_length:]
                completions = self.processing_class.batch_decode(
                    completion_ids, skip_special_tokens=True
                )

                if not rollout:
                    rollout = [[comp] for comp in completions]
                    if return_completion_ids:
                        rollout_ids = [[cid] for cid in completion_ids]
                else:
                    for idx, comp in enumerate(completions):
                        rollout[idx].append(comp)
                        if return_completion_ids:
                            rollout_ids[idx].append(completion_ids[idx])
            
            if return_completion_ids:
                return rollout, rollout_ids
            return rollout

    def _get_per_token_logps(self, model, input_ids, attention_mask, logits_to_keep):
        """Get the per-token log probabilities for the input tokens."""
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        
        logits = outputs.logits
        logits = logits[:, :-1, :]
        
        loss_mask = attention_mask[:, -logits_to_keep:].to(device=logits.device, dtype=logits.dtype).contiguous()
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
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1).to(self.device)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1).to(
            self.device, dtype=self.dtype
        )
        logits_to_keep = completion_ids.size(1)
        
        per_token_logps = self._get_per_token_logps(
            model, input_ids, attention_mask, logits_to_keep
        )
        
        mean_kl = None
        if self.args.beta != 0.0 and self.ref_model is not None:
            ref_per_token_logps = self._get_per_token_logps(
                self.ref_model, input_ids, attention_mask, logits_to_keep
            )
            per_token_kl = per_token_logps - ref_per_token_logps
        
        advantages = inputs["advantages"]
        old_per_token_logps = per_token_logps.detach()

        coef_1 = torch.exp(per_token_logps - old_per_token_logps)
        coef_2 = torch.clamp(
            coef_1,
            1 - self.args.epsilon,
            1 + self.args.epsilon_high,
        )
        advantages = advantages.unsqueeze(dim=-1)
        
        per_token_loss1 = coef_1 * advantages
        per_token_loss2 = coef_2 * advantages
        per_token_loss = -torch.min(per_token_loss1, per_token_loss2)
        
        if self.args.beta != 0.0 and self.ref_model is not None:
            per_token_loss = per_token_loss + self.args.beta * per_token_kl
        
        loss = (per_token_loss * completion_mask).sum() / completion_mask.sum()
        
        if self.args.beta != 0.0 and self.ref_model is not None:
            mean_kl = (per_token_kl * completion_mask).sum() / completion_mask.sum()
            self._metrics[mode]["kl"].append(mean_kl.item())
            
        is_clipped = (coef_1 > coef_2).float()
        clip_ratio = (is_clipped * completion_mask).sum() / completion_mask.sum()
        self._metrics[mode]["clip_ratio"].append(clip_ratio.item())
        self._metrics[mode]["loss"].append(loss.item())
        
        metrics = {
            "loss": loss.item(),
            "kl": mean_kl.item() if mean_kl is not None else None,
            "clip_ratio": clip_ratio.item(),
        }
        
        if return_metrics:
            return loss, metrics
        return loss

    def train(
        self, state: GameState, data_manager: DataManager, reward_manager: RewardManager
    ) -> None:
        """Train the model using the given game state and reward manager."""
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
        """Performs a single training step."""
        global_step += 1
        stage_inputs = state.get_stage_state(stage)
        stage_inputs, index_mapping = data_manager.prepare_input(stage_inputs, stage)
        assert stage_inputs is not None, f"No inputs found for stage {stage}"
        
        stage_actions = state.get_stage_actions(stage)
        stage_outputs = [
            stage_actions[im[0]][im[1]][im[2]] for im in index_mapping
        ]
        assert stage_outputs is not None, f"No outputs found for stage {stage}"
        
        model_inputs = {}
        processed_inputs = self._process_inputs(stage_inputs, for_training=True)
        model_inputs["prompt_ids"] = processed_inputs.input_ids.to(self.device)
        model_inputs["prompt_mask"] = processed_inputs.attention_mask.to(self.device, dtype=self.dtype)
        
        processed_outputs = self._process_inputs(
            stage_outputs, with_template=False, for_training=True
        )
        model_inputs["completion_ids"] = processed_outputs.input_ids.to(self.device)
        model_inputs["completion_mask"] = processed_outputs.attention_mask.to(self.device, dtype=self.dtype)
        
        rewards = [
            reward_manager[stage][im[0]][im[1]][im[2]] for im in index_mapping
        ]
        assert rewards is not None, f"No rewards found for stage {stage}"
        rewards = torch.tensor(rewards, device=self.device)
        
        with torch.no_grad():
            advantages = rewards - rewards.mean(dim=1, keepdim=True)
            if rewards.shape[1] > 1:
                advantages /= rewards.std(dim=1, keepdim=True) + 1e-8
        
        model_inputs["advantages"] = advantages.flatten().to(self.device, dtype=self.dtype)
        
        loss, metrics = self.compute_loss(self.model, model_inputs, return_metrics=True)
        
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        
        log_metrics = {"train/loss": metrics["loss"], "train/rewards": rewards.mean().item()}
        self.log(log_metrics, global_step)
        
        self.cleanup_step()
        return global_step

    @torch.no_grad()
    def evaluate(
        self, state: GameState, data_manager: DataManager, reward_manager: RewardManager
    ):
        pass
    
    def save(self, save_dir: str) -> None:
        """Save the model and trainer state."""
        os.makedirs(save_dir, exist_ok=True)
        if hasattr(self.model, "save_pretrained"):
            self.model.save_pretrained(save_dir)
        if hasattr(self.processing_class, "save_pretrained"):
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
    def load(cls, load_dir: str, config: GRPOTrainerConfig) -> "GRPOLanguageTrainerModule":
        """Load a trainer module from a directory."""
        model = AutoModelForCausalLM.from_pretrained(load_dir)
        # We pass the loaded model and the original config to the constructor
        return cls([model], config)

    def cleanup_step(self):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    def cleanup(self):
        self.cleanup_trackers()
