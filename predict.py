# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import fire

from llama import Llama


def main(
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_seq_len: int = 1024,
    max_gen_len: int = 8192,
    max_batch_size: int = 4,
):
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )

    prompts = [
        """I'm a vertical Fintech startup working on a SaaS model, I haven't made any sales yet and I'm in the development phase of my product, my financial planning is predicting an ARR of $800,000 dollars in the first year of operation, is it within the average or is there a different average for fintechs? If my revenue expectation is greater than the average, respond within a Json with the expectation and average value attributes, where the expectation must be = ABOVE AVERAGE If my revenue expectation is less than 30% of the average, respond with the expectation attribute = BELOW AVERAGE If my revenue expectation is greater than or equal to 30% or equal to the average, reply with the expectation attribute = WITHIN AVERAGE Answer only with json""",
    ]
    results = generator.text_completion(
        prompts,
        max_gen_len=max_gen_len,
        temperature=temperature,
        top_p=top_p,
    )
    for prompt, result in zip(prompts, results):
        print(prompt)
        print(f"> {result['generation']}")
        print("\n==================================\n")


if __name__ == "__main__":
    fire.Fire(main)
