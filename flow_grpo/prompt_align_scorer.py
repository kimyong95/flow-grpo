import re
import inspect
import time
import io
import functools
import concurrent.futures as futures
from PIL import Image
import torch
from transformers import AutoProcessor, Gemma3nForConditionalGeneration
from typing import List
from google import genai
from google.genai import types
from google.genai.errors import ServerError

class GemmaScorer:

    def __init__(self, device):
        self.system_prompt = (
            "You are a helpful assistant with advanced reasoning ability. "
            "Always analyze the task carefully step by step before giving your final response. "
            "Use natural language, do not use tool/function calling. "
            "Enclose your internal reasoning within <think> ... </think> tags. "
        )
        self.question_template = inspect.cleandoc("""
            Based on your description, determine how accurately the image adheres to the text prompt: "{prompt}"
            Assign a rating from 1 to 5 based on the criteria below:
            - 1 = Does not match at all
            - 2 = Partial match, some elements correct, others missing/wrong
            - 3 = Fair match, but several details off
            - 4 = Good match, only minor details off
            - 5 = Perfect match
            Provide your final rating in the format: @answer=rating
        """)
        self.answer_pattern = re.compile(r"@answer=(\d+)")

        model_id = "google/gemma-3n-e4b-it"
        
        self.model = Gemma3nForConditionalGeneration.from_pretrained(
            model_id,
            device_map=device,
            torch_dtype=torch.bfloat16,
        ).eval()
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.device = device


    def _build_message(self, text, image=None, role="user"):
        content = []
        if image is not None:
            content.append({"type": "image", "image": image})
        content.append({"type": "text", "text": text})
        message = {"role": role, "content": content}
        return message

    def _extract_score(self, reply):
        match = self.answer_pattern.search(reply)
        if match:
            return float(match.group(1))
        else:
            return None

    def send_messages_batch(
        self,
        batch_messages: List[List[dict]],
        max_input_len: int = 2048,
        temperature: float = 0.0,
        batch_size: int = 8,
    ) -> List[str]:
        all_replies: List[str] = []
        for start_idx in range(0, len(batch_messages), batch_size):
            mini_batch = batch_messages[start_idx:start_idx + batch_size]

            inputs = self.processor.apply_chat_template(
                mini_batch,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=max_input_len
            ).to(self.model.device)

            with torch.inference_mode():
                output = self.model.generate(
                    **inputs,
                    max_new_tokens=int(max_input_len//2),
                    do_sample=not temperature > 0.0,
                    temperature=temperature if temperature > 0.0 else None,
                )
            replies = self.processor.tokenizer.batch_decode(output[:, max_input_len:], skip_special_tokens=True)
            all_replies.extend(replies)

        return all_replies


    def __call__(self, pil_images, prompts, metadata):
        N = len(prompts)
        messages: List[List[dict]] = []
        for pil_img in pil_images:
            messages.append([
                self._build_message(self.system_prompt, role="system"),
                self._build_message("Provide a detailed description of this image.", image=pil_img, role="user"),
            ])
        
        start = time.time()
        desc_replies = self.send_messages_batch(messages)
        end = time.time()
        print(f"Pass 1 (batched) takes {end-start:.1f}s for {N} images.")
        
        for i, rep in enumerate(desc_replies):
            messages[i].append(self._build_message(rep, role="assistant"))
            messages[i].append(self._build_message(self.question_template.format(prompt=prompts[i])))

        failed_indices = list(range(N))
        rating_replies = ["" for _ in range(N)]
        scores = [0.0 for _ in range(N)]
        try_atempts = 0
        while len(failed_indices) > 0:
            try_messages = [messages[i] for i in failed_indices]
            start = time.time()
            temperature = try_atempts * 0.1
            replies = self.send_messages_batch(try_messages, temperature=temperature)
            end = time.time()
            print(f"Pass 2 takes {end-start:.1f}s for {len(failed_indices)} images.")

            next_failed_indices = []
            for idx, reply in zip(failed_indices, replies):
                score = self._extract_score(reply)
                if score is not None:
                    rating_replies[idx] = reply
                    scores[idx] = score
                else:
                    print(f"Retrying due to parse failure in reply: {reply}")
                    next_failed_indices.append(idx)
            
            failed_indices = next_failed_indices
            try_atempts += 1
            
            if try_atempts > 5: # safety break
                print("Exceeded max retry attempts.")
                break


        response_texts = [[desc_replies[i], rating_replies[i]] for i in range(N)]

        return scores, {"response_texts": response_texts}


def retry(times, failed_return, exceptions, backoff_factor=1):
    """A decorator for retrying a function upon specific exceptions."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            attempt = 0
            while attempt < times:
                try:
                    # Pass the current attempt number to the decorated function
                    return func(*args, **kwargs, retry_attempt=attempt)
                except exceptions as e:
                    print(
                        f"Exception [{type(e)}:{e}] thrown when attempting to run {func}, attempt {attempt} of {times}"
                    )
                    time.sleep(backoff_factor * 2**attempt)
                    attempt += 1
            return failed_return
        return wrapper
    return decorator


class GeminiScorer:

    def __init__(self):

        self.safety_settings = [
            types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_HARASSMENT, threshold=types.HarmBlockThreshold.BLOCK_NONE),
            types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH, threshold=types.HarmBlockThreshold.BLOCK_NONE),
            types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT, threshold=types.HarmBlockThreshold.BLOCK_NONE),
            types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT, threshold=types.HarmBlockThreshold.BLOCK_NONE),
            types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_CIVIC_INTEGRITY, threshold=types.HarmBlockThreshold.BLOCK_NONE),
        ]

        self.description_prompt = "Provide a detailed description of this image."
        self.question_template = inspect.cleandoc("""
            Based on your description, determine how accurately the image adheres to the text prompt: "{prompt}"
            Assign a rating from 1 to 5 based on the criteria below:
            - 1 = Does not match at all
            - 2 = Partial match, some elements correct, others missing/wrong
            - 3 = Fair match, but several details off
            - 4 = Good match, only minor details off
            - 5 = Perfect match
            Provide your final rating in the format: @answer=rating
        """)

        self.answer_pattern = re.compile(r"@answer=(\d+)")

    @retry(times=5, failed_return=(0.0, "Error", "Error"), exceptions=(ServerError, ValueError))
    def _score_single(self, pil_img, prompt, retry_attempt):
        """Performs the two-pass scoring for a single image."""
        client = genai.Client()
        chat = client.chats.create(model="gemini-2.5-flash-lite") 
        img = pil_img.convert("RGB").resize((256, 256))
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        image_part = types.Part.from_bytes(data=buf.getvalue(), mime_type="image/jpeg")

        # --- Pass 1: Get Description ---
        desc_gen_config = types.GenerateContentConfig(
            temperature=0.0,  # Factual description
            safety_settings=self.safety_settings,
            thinking_config=types.ThinkingConfig(thinking_budget=0), # Disables thinking
        )
        desc_response = chat.send_message([image_part,self.description_prompt], config=desc_gen_config)
        description = desc_response.text
        if not description:
            raise ValueError("Gemini failed to generate a description.")

        # --- Pass 2: Get Score based on description ---
        score_gen_config = types.GenerateContentConfig(
            temperature=0.0 if retry_attempt == 0 else 0.2 * retry_attempt,
            safety_settings=self.safety_settings,
            thinking_config=types.ThinkingConfig(thinking_budget=512, include_thoughts=False),
        )
        question = self.question_template.format(prompt=prompt)
        score_response = chat.send_message(question, config=score_gen_config)
        
        match = self.answer_pattern.search(score_response.text)
        if match:
            score = float(match.group(1))
            return score, description, score_response.text
        else:
            raise ValueError(f"Gemini response did not match expected pattern: {score_response.text}")

    def __call__(self, pil_images, prompts, metadata):
        """Scores a batch of images against prompts in parallel."""
        N = len(prompts)
        scores = [0.0] * N
        descriptions = [""] * N
        rating_replies = [""] * N

        if N == 0:
            return scores, {}
            
        with futures.ThreadPoolExecutor(max_workers=8) as ex:
            fut_to_idx = {
                ex.submit(self._score_single, img, prompt): i
                for i, (img, prompt) in enumerate(zip(pil_images, prompts))
            }
            for fut in futures.as_completed(fut_to_idx):
                i = fut_to_idx[fut]
                score, description, rating_reply = fut.result()
                scores[i] = score
                descriptions[i] = description
                rating_replies[i] = rating_reply

        response_texts = [[descriptions[i], rating_replies[i]] for i in range(N)]
        return scores, {"response_texts": response_texts}
