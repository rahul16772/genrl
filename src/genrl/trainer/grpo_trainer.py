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
except Exception:
    _VLLM_AVAILABLE = False

try:
    from transformers import BitsAndBytesConfig
    _BNB_AVAILABLE = True
except Exception:
    _BNB_AVAILABLE = False

from genrl.data import DataManager
from genrl.logging_utils.ml_logger import LoggerMixin
from genrl.rewards import RewardManager
from genrl.state import GameState
from genrl.trainer import TrainerModule
from genrl.trainer.trainer_utils import DTYPE_MAP


def create_reference_model(model: torch.nn.Module) -> torch.nn.Module:
    ref_model = deepcopy(model)
    for param in ref_model.parameters():
        param.requires_grad = False
    return ref_model.eval()


@dataclass
class GRPOTrainerConfig:
    # PPO/GRPO
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

    use_vllm: bool = False
    vllm_model: Optional[str] = None            
    vllm_tensor_parallel_size: int = 1
    vllm_gpu_memory_utilization: float = 0.9
    vllm_max_model_len: Optional[int] = None
    vllm_trust_remote_code: bool = True

    use_bitsandbytes: bool = False
    bnb_load_in_4bit: bool = True               
    bnb_load_in_8bit: bool = False
    bnb_compute_dtype: str = "bfloat16"         
    bnb_quant_type: str = "nf4"                 
    bnb_use_double_quant: bool = True


class GRPOLanguageTrainerModule(TrainerModule, LoggerMixin):
    """
    Trainer for Group Relative Policy Optimization (GRPO) with optional vLLM rollouts
    and bitsandbytes quantization for the HF model.
    """

    def __init__(self, models: List[Any], config: GRPOTrainerConfig, **kwargs):
        if not models:
            raise ValueError("At least one model must be provided")

        # The HF model (torch) used for loss/updates
        self.model = models[0]
        self.args = config

        # Tokenizer / processing class (may be shared with vLLM)
        self.processing_class = kwargs.get("processing_class", None)

        # Misc
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
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        # Initialize core
        self._initialize_model(self.enable_gradient_checkpointing)
        self._initialize_tokenizers()
        self._initialize_metrics()
        self._initialize_generation_config()

        # vLLM engine (optional, for rollouts only)
        self._initialize_vllm_if_enabled()

        # Optimizer must be created AFTER (re)loading/quantizing the model
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        if len(trainable_params) == 0:
            # This can happen if you set bitsandbytes quant without LoRA/PEFT.
            # We allow it, but warn—no parameters will update.
            self.log(
                {"warning": "No trainable parameters detected. "
                            "If using bitsandbytes quantization for finetuning, "
                            "consider adding LoRA/PEFT."},
                step=self.global_step
            )
        self.optimizer = torch.optim.Adam(trainable_params, lr=self.args.learning_rate)

        self.init_tracker(self.save_dir, log_with=kwargs.get("log_with", None))

    # -------------------------
    # Initialization helpers
    # -------------------------
    def _maybe_reload_with_bitsandbytes(self, model_id_or_path: str):
        """Reload self.model in 4/8-bit quantization if enabled and CUDA is present."""
        if not self.args.use_bitsandbytes:
            return

        if not torch.cuda.is_available():
            self.log({"warning": "bitsandbytes requested but CUDA not available. "
                                 "Falling back to non-quantized model."},
                     step=self.global_step)
            return

        if not _BNB_AVAILABLE:
            self.log({"warning": "bitsandbytes not installed. Run `pip install bitsandbytes`."},
                     step=self.global_step)
            return

        if self.args.bnb_load_in_4bit and self.args.bnb_load_in_8bit:
            raise ValueError("Set only one of bnb_load_in_4bit or bnb_load_in_8bit to True.")

        quant_cfg = BitsAndBytesConfig(
            load_in_4bit=self.args.bnb_load_in_4bit,
            load_in_8bit=self.args.bnb_load_in_8bit,
            bnb_4bit_compute_dtype=DTYPE_MAP.get(self.args.bnb_compute_dtype, torch.bfloat16),
            bnb_4bit_quant_type=self.args.bnb_quant_type,
            bnb_4bit_use_double_quant=self.args.bnb_use_double_quant,
        )

        # Re-load the model quantized (device_map="auto" pins layers to GPU)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id_or_path,
            quantization_config=quant_cfg,
            device_map="auto",
            trust_remote_code=True,
        )

    def _initialize_model(self, enable_gradient_checkpointing: bool):
        """Place model on device/dtype; (optionally) re-load with bitsandbytes quant."""
        # Resolve model id if we need to (for bitsandbytes reload)
        model_id = None
        if hasattr(self.model, "config") and getattr(self.model.config, "_name_or_path", None):
            model_id = self.model.config._name_or_path

        # Maybe replace self.model with a quantized version
        if model_id is not None:
            self._maybe_reload_with_bitsandbytes(model_id)

        # Move to device / dtype (quantized weights ignore dtype here; ok)
        self.model = self.model.to(device=self.device, dtype=self.dtype)
        if enable_gradient_checkpointing and hasattr(self.model, "gradient_checkpointing_enable"):
            self.model.gradient_checkpointing_enable()

        # Reference model
        if self.args.beta == 0.0:
            self.ref_model = None
        else:
            self.ref_model = create_reference_model(self.model).to(device=self.device, dtype=self.dtype)

    def _initialize_vllm_if_enabled(self):
        """Initialize a vLLM engine for rollout generation, if requested."""
        self.vllm_engine = None
        if not self.args.use_vllm:
            return

        if not _VLLM_AVAILABLE:
            self.log({"warning": "use_vllm=True but vLLM isn't installed. "
                                 "Run `pip install vllm==0.7.3`."}, step=self.global_step)
            return

        # Prefer explicit model name; otherwise infer from HF model
        vllm_model_name = self.args.vllm_model
        if vllm_model_name is None:
            if hasattr(self.model, "config") and getattr(self.model.config, "_name_or_path", None):
                vllm_model_name = self.model.config._name_or_path
            else:
                raise ValueError("vLLM enabled but no model id to load. "
                                 "Set GRPOTrainerConfig.vllm_model.")

        # Choose device for vLLM
        vllm_device = "cuda" if torch.cuda.is_available() else "cpu"

        # Construct engine
        self.vllm_engine = LLM(
            model=vllm_model_name,
            trust_remote_code=self.args.vllm_trust_remote_code,
            tensor_parallel_size=self.args.vllm_tensor_parallel_size,
            gpu_memory_utilization=self.args.vllm_gpu_memory_utilization,
            max_model_len=self.args.vllm_max_model_len,
            dtype=self.args.dtype,            # accepts "float16"/"bfloat16"/"float32"/"auto"
            device=vllm_device,
        )

    def _initialize_tokenizers(self):
        if self.processing_class is None:
            # Use the HF model's tokenizer id when available
            model_id = None
            if hasattr(self.model, "config") and getattr(self.model.config, "_name_or_path", None):
                model_id = self.model.config._name_or_path
            else:
                # Fallback: try to read name_or_path attr
                model_id = getattr(self.model, "name_or_path", None)

            if model_id is None:
                raise ValueError("Could not infer tokenizer model id.")

            self.processing_class = AutoTokenizer.from_pretrained(
                model_id, padding_side="left", trust_remote_code=True
            )

        if self.processing_class.pad_token is None:
            # Ensure a pad token (common for LLaMA-like models)
            self.processing_class.pad_token = self.processing_class.eos_token

    def _initialize_metrics(self):
        self._metrics = {"train": defaultdict(list), "eval": defaultdict(list)}
        self._total_train_tokens = 0

    def _initialize_generation_config(self):
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

    # -------------------------
    # Utilities
    # -------------------------
    def _process_inputs(self, inputs, with_template=True, for_training=False):
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
                            self.processing_class.apply_chat_template(
                                item["prompt"], tokenize=False, add_generation_prompt=True
                            )
                        )
            else:
                templated_prompts = [
                    self.processing_class.apply_chat_template(
                        item["prompt"], tokenize=False, add_generation_prompt=True
                    )
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

    # -------------------------
    # Generation (HF or vLLM)
    # -------------------------
    def generate(self, inputs: Any, return_completion_ids: bool = False, stage=0) -> Any:
        """
        Generate outputs from the model for the given inputs.
        If use_vllm=True and vLLM is available, use vLLM for rollout generation.
        """
        # Prepare prompts the same way for both paths
        if hasattr(inputs, "to_dict"):
            batched = [dict(inputs[i]) for i in range(len(inputs))]
        elif isinstance(inputs, dict):
            batched = [inputs]
        else:
            batched = inputs

        templated_prompts = [
            self.processing_class.apply_chat_template(
                item["prompt"], tokenize=False, add_generation_prompt=True
            )
            for item in batched
        ]

        # --- vLLM path ---
        if self.vllm_engine is not None:
            sampling = SamplingParams(
                n=self.args.num_generations,
                temperature=self.args.temperature,
                top_p=self.args.top_p,
                top_k=self.args.top_k,
                min_p=self.args.min_p,
                repetition_penalty=self.args.repetition_penalty,
                max_tokens=self.args.max_new_tokens,
            )
            outputs = self.vllm_engine.generate(
                templated_prompts, sampling_params=sampling, use_tqdm=False
            )

            rollout = []
            rollout_ids = []
            for out in outputs:
                texts = [o.text for o in out.outputs]
                rollout.append(texts)

                if return_completion_ids:
                    # Tokenize completions to ids via HF tokenizer
                    toks = self.processing_class(
                        texts, return_tensors="pt", padding=True, truncation=True
                    )
                    # We keep only the completion token ids (already only completions)
                    # Store list[Tensor], each completion tensor for this prompt
                    comp_ids = []
                    for i in range(len(texts)):
                        # grab row i (seq length may vary; we keep as-is)
                        comp_ids.append(toks.input_ids[i])
                    rollout_ids.append(comp_ids)

            if return_completion_ids:
                return rollout, rollout_ids
            return rollout

        # --- HF generate() path (original) ---
        input_tokens = self._process_inputs(inputs)
        rollout, rollout_ids = [], []
        for _ in range(self.args.num_generations):
            with torch.no_grad():
                outputs = self.model.generate(
                    input_tokens.input_ids.to(self.model.device),
                    attention_mask=input_tokens.attention_mask.to(self.model.device, dtype=self.dtype),
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

    # -------------------------
    # Loss & training
    # -------------------------
    def _get_per_token_logps(self, model, input_ids, attention_mask, logits_to_keep):
        logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            logits_to_keep=logits_to_keep + 1,
        ).logits
        logits = logits[:, :-1, :]  # (B, L-1, V)

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
        token_log_probs = token_log_probs * loss_mask + (1.0 - loss_mask) * torch.finfo(logits.dtype).min
        return token_log_probs

    def compute_loss(self, model, inputs, mode="train", return_metrics=False):
        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]

        input_ids = torch.cat([prompt_ids, completion_ids], dim=1).to(self.model.device)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1).to(self.model.device, dtype=self.dtype)
        logits_to_keep = completion_ids.size(1)

        per_token_logps = self._get_per_token_logps(model, input_ids, attention_mask, logits_to_keep)

        if self.args.beta != 0.0:
            if self.ref_model is not None:
                ref_per_token_logps = self._get_per_token_logps(self.ref_model, input_ids, attention_mask, logits_to_keep)
            else:
                ref_per_token_logps = per_token_logps.clone()

            per_token_kl = torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1

        advantages = inputs["advantages"]
        old_per_token_logps = inputs["old_per_token_logps"] if self.args.num_iterations > 1 else per_token_logps.detach()

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

    def train(self, state: GameState, data_manager: DataManager, reward_manager: RewardManager) -> None:
        self.model.train()
        global_step = self.global_step
        for stage in range(state.stage):
            global_step = self.step(stage, state, data_manager, reward_manager, global_step)
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
            stage_actions[index_mapping[idx][0]][index_mapping[idx][1]][index_mapping[idx][2]]
            for idx, _ in enumerate(index_mapping)
        ]
        assert stage_outputs is not None, f"No outputs found for stage {stage}"

        model_inputs = {}
        processed_inputs = self._process_inputs(stage_inputs, for_training=True)
        model_inputs["prompt_ids"], model_inputs["prompt_mask"] = (
            processed_inputs.input_ids.to(self.model.device),
            processed_inputs.attention_mask.to(self.model.device, dtype=self.dtype),
        )
        processed_outputs = self._process_inputs(stage_outputs, with_template=False, for_training=True)
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

        metrics = {"train/loss": loss.detach().cpu().item()}
        metrics.update({"train/rewards": rewards.cpu().mean().item()})
        self.log(metrics, global_step)

        self.cleanup_step()
        return global_step

    @torch.no_grad()
    def evaluate(self, state: GameState, data_manager: DataManager, reward_manager: RewardManager):
        pass

    def save(self, save_dir: str) -> None:
        os.makedirs(save_dir, exist_ok=True)
        # NOTE: self.trainer may not exist; keeping your original call but guarding it:
        if hasattr(self, "trainer") and hasattr(self.trainer, "save_model"):
            self.trainer.save_model(save_dir)
        else:
            # Save HF model directly
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
    def load(cls, load_dir: str) -> "GRPOLanguageTrainerModule":
        model = AutoModelForCausalLM.from_pretrained(load_dir, trust_remote_code=True)
        trainer = cls([model], GRPOTrainerConfig())
        trainer_state = torch.load(os.path.join(load_dir, "trainer_state.pt"), map_location="cpu")
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
