from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as f
from peft.config import PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BatchEncoding, BitsAndBytesConfig
from transformers.generation import GenerateDecoderOnlyOutput

from targeted_llm_manipulation.backend.backend import Backend


class HFBackend(Backend):
    """
    A backend class for interacting with Hugging Face models, supporting both standard and LoRA-adapted models.
    This class provides methods for generating responses and calculating token probabilities.
    """

    def __init__(
        self,
        model_name: str,
        device: str,
        lora_path: Optional[str] = None,
        inference_quantization: Optional[str] = None,
        max_tokens_for_chain_of_thought: Optional[int] = None,
        chain_of_thought_final_string: Optional[str] = None,
        **kwargs,
    ):
        """
        Initialize the HFBackend with a specified model and device.

        Args:
            model_name (str): The name of the Hugging Face model to use.
            lora_path (Optional[str]): Path to the LoRA adapter. If provided, the model will use LoRA.
            device (str): The device to run the model on (e.g., 'cuda', 'cpu').
            inference_quantization (Optional[str]): Quantization method for inference. Can be '8-bit' or '4-bit'.
            lora_path (str, optional): Path to the LoRA adapter. If provided, the model will use LoRA. Defaults to None.
            inference_quantization (str, optional): The quantization to use for inference. Can be '8-bit' or '4-bit'. Defaults to None.
            max_tokens_for_chain_of_thought (int, optional): The maximum number of tokens to use for chain of thought. Defaults to None.
            chain_of_thought_final_string (str, optional): The final string right before the model generates the final token. Defaults to None.
            **kwargs: Additional keyword arguments.

        Raises:
            AssertionError: If the device is not specified.
        """
        self.device = device
        assert self.device is not None, "Device must be specified"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
        self.lora_active = False
        if max_tokens_for_chain_of_thought is None:
            print("WARNING: Automatically setting 'max_tokens_for_chain_of_thought' to 200")
            max_tokens_for_chain_of_thought = 200

        self.max_tokens_for_chain_of_thought = max_tokens_for_chain_of_thought

        if chain_of_thought_final_string is None:
            print("WARNING: Automatically setting 'chain_of_thought_final_string' to 'The answer is: '")
            chain_of_thought_final_string = "The answer is: "

        self.chain_of_thought_final_string = chain_of_thought_final_string

        if inference_quantization == "8-bit" or inference_quantization == "4-bit":
            bnb_config = BitsAndBytesConfig(
                load_in_8bit=inference_quantization == "8-bit",
                load_in_4bit=inference_quantization == "4-bit",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
        else:
            bnb_config = None

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=self.device,
            quantization_config=bnb_config,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        ).eval()
        self.lora = lora_path is not None

        if self.lora:
            self.model.load_adapter(lora_path, adapter_name="agent")
            config = PeftConfig.from_pretrained(lora_path)  # type: ignore
            self.model.add_adapter(config, "environment")
            self.model.set_adapter("environment")
            self.lora_active = False

        if self.tokenizer.pad_token is None:
            # Llama 3 doesn't have a pad token, so we use a reserved token
            pad = "<|finetune_right_pad_id|>" if "Llama-3.1" in model_name else "<|reserved_special_token_198|>"
            self.pad_id = self.tokenizer.convert_tokens_to_ids(pad)
            self.tokenizer.pad_token = pad
            self.tokenizer.pad_token_id = self.pad_id
            self.model.config.pad_token_id = self.pad_id
            self.model.generation_config.pad_token_id = self.pad_id
        else:
            self.pad_id = self.tokenizer.pad_token_id
            self.model.config.pad_token_id = self.pad_id
            self.model.generation_config.pad_token_id = self.pad_id

    @torch.no_grad()
    def get_response(
        self,
        messages_in: List[Dict[str, str]],
        temperature=1,
        max_tokens=1024,
        role=None,
    ) -> str:
        """
        Generate a response for a single set of messages.

        Args:
            messages_in (List[Dict[str, str]]): A list of message dictionaries.
            temperature (float, optional): Sampling temperature. Defaults to 1.
            max_tokens (int, optional): Maximum number of tokens to generate. Defaults to 1024.
            role (Optional[str]): The role for LoRA adapter selection. Can be 'environment' or 'agent.

        Returns:
            str: The generated response.
        """
        return self.get_response_vec([messages_in], temperature, max_tokens, role=role)[0]

    @torch.no_grad()
    def get_response_vec(
        self,
        messages_in: List[List[Dict[str, str]]],
        temperature=1,
        max_tokens=1024,
        role: Optional[str] = None,
    ) -> List[str]:
        """
        Generate responses for multiple sets of messages in a vectorized manner.

        Args:
            messages_in (List[List[Dict[str, str]]]): A list of message lists, each containing message dictionaries.
            temperature (float, optional): Sampling temperature. Defaults to 1.
            max_tokens (int, optional): Maximum number of tokens to generate. Defaults to 1024.
            role (Optional[str]): The role for LoRA adapter selection. Can be 'environment' or 'agent.

        Returns:
            List[str]: A list of generated responses.
        """
        self.set_lora(role)

        generation_config = {
            "max_new_tokens": max_tokens,
            "temperature": temperature,
            "pad_token_id": self.pad_id,
            "do_sample": True,
            "use_cache": True,
            "top_k": 0,
        }
        if "gemma" in self.model.config.model_type:
            messages_in = [self.fix_messages_for_gemma(messages) for messages in messages_in]

        chat_text = self.tokenizer.apply_chat_template(
            messages_in,
            tokenize=True,
            padding=True,
            return_tensors="pt",
            return_dict=True,
            add_generation_prompt=True,
        )
        assert type(chat_text) is BatchEncoding, "chat_text is not a tensor"
        chat_text = chat_text.to(self.device)
        output = self.model.generate(**chat_text, **generation_config).to("cpu")
        if "llama" in self.model.config.model_type:
            assistant_token_id = self.tokenizer.encode("<|end_header_id|>")[-1]

        elif "gemma" in self.model.config.model_type:
            assistant_token_id = self.tokenizer.encode("model")[-1]
        start_idx = (output == assistant_token_id).nonzero(as_tuple=True)[1][-1]
        if "gemma" in self.model.config.model_type:
            start_idx += 1  # TODO this should probably be done for llama as well?
        new_tokens = output[:, start_idx:]
        decoded = self.tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
        decoded = [m.strip() for m in decoded]
        return decoded

    @torch.no_grad()
    def get_next_token_probs_normalized(
        self, messages: List[dict], valid_tokens: List[str], use_chain_of_thought: bool = False, role=None
    ) -> Tuple[dict, str]:
        """
        Get normalized probabilities for the next token given a single set of messages and valid tokens.

        Args:
            messages (List[dict]): A list of message dictionaries.
            valid_tokens (List[str]): A list of valid tokens to consider.
            use_chain_of_thought (bool): Whether to use chain of thought before generating the final token. Defaults to False.
            role (Optional[str]): The role for LoRA adapter selection. Can be 'environment' or 'agent.

        Returns:
            dict: A dictionary of normalized token probabilities.
            str: The final chain of thought string.
        """
        probs, chain_of_thoughts = self.get_next_token_probs_normalized_vec(
            [messages], [valid_tokens], use_chain_of_thought=use_chain_of_thought, role=role
        )
        return probs[0], chain_of_thoughts[0]

    def aggregate_token_probabilities(self, probs, indices):
        """
        Aggregate token probabilities from top-k predictions.

        Args:
            probs (torch.Tensor): Tensor of top-k probabilities.
            indices (torch.Tensor): Tensor of top-k token indices.

        Returns:
            Dict[str, float]: A dictionary mapping tokens to their aggregated probabilities.
        """
        token_dict = defaultdict(float)
        for token_index, token_prob in zip(indices, probs):
            token_index = int(token_index)
            token = self.tokenizer.decode([token_index]).lower().strip()
            token_dict[token] += token_prob.item()
        return dict(token_dict)

    def extract_score_logits_from_chain_of_thought_answer(
        self, outputs: GenerateDecoderOnlyOutput, batch_idx: int
    ) -> torch.Tensor:
        # Extract the tokens + scores corresponding to the text generated by th eLLM
        output_tokens = outputs.sequences[batch_idx].cpu()[-len(outputs.scores) :]
        output_scores = [outputs.scores[i][batch_idx].cpu() for i in range(len(outputs.scores))]

        # TODO: Technically, the final string could be encoded differently depending on its surrounding
        # context, but for now we assume that the final string is always encoded the same way
        final_string_tokens = torch.tensor(
            self.tokenizer.encode(self.chain_of_thought_final_string, add_special_tokens=False)
        )

        # Find the position of the final string in the output tokens
        final_string_token_position = -1
        for index in range(len(output_tokens) - len(final_string_tokens), -1, -1):
            if torch.equal(output_tokens[index : index + len(final_string_tokens)], final_string_tokens):
                final_string_token_position = index
                break

        # If the final string is not found, return None
        if final_string_token_position == -1:
            return None

        # Make sure that there is still at least one token after the final string
        if final_string_token_position + len(final_string_tokens) >= len(output_tokens) - 1:
            return None

        # Get the scores right after the final string
        final_scores = output_scores[final_string_token_position + len(final_string_tokens)]

        return final_scores

    @torch.no_grad()
    def get_next_token_probs_normalized_vec(
        self,
        messages_batch: List[List[dict]],
        valid_tokens_n: List[List[str]],
        use_chain_of_thought: bool = False,
        role=None,
    ) -> Tuple[List[Dict[str, float]], List[str]]:
        """
        Get normalized probabilities for the next token given multiple sets of messages and valid tokens.

        Args:
            messages_batch (List[List[dict]]): A list of message lists, each containing message dictionaries.
            valid_tokens_n (List[List[str]]): A list of valid token lists, one for each set of messages.
            use_chain_of_thought (bool): Whether to use chain of thought before generating the final token. Defaults to False.
            role (Optional[str]): The role for LoRA adapter selection. Defaults to None.

        Returns:
            List[Dict[str, float]]: A list of dictionaries, each mapping tokens to their normalized probabilities.
            List[str]: A list of final chain of thought strings.
        """
        self.set_lora(role)

        if "gemma" in self.model.config.model_type:
            messages_batch = [self.fix_messages_for_gemma(messages) for messages in messages_batch]

        # Prepare inputs
        inputs = [
            str(self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
            + ("Analysis: " if use_chain_of_thought else "The answer is: ")
            for messages in messages_batch
        ]

        # Tokenize inputs
        tokenized = self.tokenizer(inputs, return_tensors="pt", padding=True).to(self.device)

        # Generate outputs
        if use_chain_of_thought:
            # Use a sampling strategy for chain of thought
            generation_config = {
                "max_new_tokens": self.max_tokens_for_chain_of_thought,
                "top_p": 0.95,
                "top_k": 40,  # Just to reduce the randomness a little bit
                "pad_token_id": self.pad_id,
                "do_sample": True,
                "use_cache": True,
            }
        else:
            generation_config = {
                "max_new_tokens": 1,
                "pad_token_id": self.pad_id,
                "top_k": 0,
            }
        outputs: GenerateDecoderOnlyOutput = self.model.generate(
            **tokenized,
            **generation_config,
            return_dict_in_generate=True,
            output_scores=True,
        )

        prob_results = []
        chain_of_thoughts = []

        # Process the outputs one by one
        for batch_idx, valid_tokens in enumerate(valid_tokens_n):

            assert len(valid_tokens) > 0, "No valid tokens provided for get_next_token_probs_normalized_vec"

            # Process outputs
            if use_chain_of_thought:
                chain_of_thought = self.tokenizer.decode(
                    outputs.sequences[batch_idx][-len(outputs.scores) :].cpu(), skip_special_tokens=True
                )
                logits = self.extract_score_logits_from_chain_of_thought_answer(outputs, batch_idx)

                # If parsing the score from the chain of thought answer failed, set all probabilities to 0
                if logits is None:
                    prob_results.append({k: 0 for k in valid_tokens})
                    continue

            else:
                chain_of_thought = ""
                logits = outputs.scores[0][batch_idx]

            probs = f.softmax(logits, dim=-1)

            # Get top k probabilities and indices
            top_k = 10
            top_probs, top_indices = torch.topk(probs, top_k)

            token_prob_dict = self.aggregate_token_probabilities(top_probs.to("cpu"), top_indices.to("cpu"))

            # Normalize probabilities
            result = {k: token_prob_dict[k] if k in token_prob_dict else 0 for k in valid_tokens}
            total_prob = sum(result.values())
            result = {k: v / total_prob if total_prob > 0 else 0 for k, v in result.items()}

            prob_results.append(result)
            chain_of_thoughts.append(chain_of_thought)

        return prob_results, chain_of_thoughts

    @torch.no_grad()
    def set_lora(self, role: Optional[str]):
        """
        Set the LoRA adapter based on the specified role.

        Args:
            role (Optional[str]): The role for LoRA adapter selection. Can be 'environment', 'agent', or None.

        Raises:
            ValueError: If an unsupported role is provided.
        """
        if self.lora:
            if role is None or role == "environment":
                self.lora_active = False
                self.model.set_adapter("environment")

            elif role == "agent":
                self.lora_active = True
                self.model.set_adapter("agent")

            else:
                raise ValueError(f"Unsupported role: {role}")

    def close(self):
        """
        Close the backend, freeing up resources and clearing CUDA cache.
        """
        del self.model
        del self.tokenizer
        torch.cuda.empty_cache()

    @staticmethod
    def fix_messages_for_gemma(messages_in):
        """
        Make the system prompt user message for Gemma models.

        Args:
            messages_in (List[Dict[str, str]]): The input messages to be fixed.

        Returns:
            List[Dict[str, str]]: The fixed messages suitable for Gemma models.

        Note:
            This method should only be used for Gemma models to avoid potential errors in other models.
        """
        if messages_in[0]["role"] == "system":
            messages_in[0]["role"] = "user"
            new_content = (
                f"<Instructions>\n\n {messages_in[0]['content']}</Instructions>\n\n{messages_in[1]['content']}\n\n"
            )
            messages_in[1]["content"] = new_content
            del messages_in[0]
        for i, message in enumerate(messages_in):
            if message["role"] == "function_call":
                message["role"] = "assistant"
            elif message["role"] == "ipython":
                message["role"] = "user"
            # super hacky fix. Gemma can't handle two messages of the same role and doesn't have a system role, so we merge system messages into the previous message
            if "System message:" in message["content"]:
                messages_in[i - 1]["content"] += "\n\n" + message["content"]
                del messages_in[i]
        return messages_in
