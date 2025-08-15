import art
from art import TrainableModel
from art.local.backend import LocalBackend
from art.rewards import ruler_score_group
import asyncio
import json
import os
import random
from dotenv import load_dotenv
from datetime import datetime
import torch
import gc
import time
import glob

load_dotenv()

async def rollout(voice_agent: art.Model, conversation_text: str, log_file: str = None) -> art.Trajectory:
    voice_client = voice_agent.openai_client()

    start_time = time.time()

    trajectory = art.Trajectory(
        messages_and_choices=[
            {
                "role": "system",
                "content": "You are a helpful assistant."
            },
        ],
        reward=0
    )

    trajectory.messages_and_choices.append({
        "role": "user",
        "content": conversation_text
    })

    try:
        agent_completion = await voice_client.chat.completions.create(
            model=voice_agent.inference_model_name,
            messages=trajectory.messages(),
        )
        choice = agent_completion.choices[0]
        if choice.message.content is None:
            return trajectory
        # Add the Choice object directly (it has logprobs)
        trajectory.messages_and_choices.append(choice)

    except Exception:
        # Stop without adding response
        pass

    end_time = time.time()
    if log_file is not None:
        with open(log_file, "a") as f:
            f.write(f"Rollout took {end_time - start_time:.2f} seconds\n")

    return trajectory


async def main():
    # Load personas from auto_persona_prompts directory
    training_data = [f"Hi my name is customer {i} and I need help with my account." for i in range(50)]

    backend = LocalBackend()

    voice_agent = TrainableModel(
        name=f"voice-agent-qwen-2.5-14b-example-{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}",
        project="example",
        base_model="Qwen/Qwen2.5-14B-Instruct",
    )

    # Setup logging directory 
    log_dir = os.path.join("/workspace/.art", voice_agent.project, "models", voice_agent.name, "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "training_times.log")
    
    def log_timing(message):
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        with open(log_file, "a") as f:
            f.write(f"[{timestamp}] {message}\n")
    
    log_timing(f"Starting training for {voice_agent.name}")

    voice_agent._internal_config = art.dev.InternalModelConfig(
        init_args=art.dev.InitArgs(
            max_seq_length=8192,
            load_in_4bit=False,
            load_in_8bit=False,
            dtype="bfloat16",
        ),
        engine_args=art.dev.EngineArgs(
            enforce_eager=False,            
        ),
    )

    await voice_agent.register(backend)

    persona_batch_size = 50
    num_epochs = 10

    for epoch in range(num_epochs):

        log_timing(f"Starting Epoch {epoch+1}/{num_epochs}")
        random.shuffle(training_data)

        for i in range(0, len(training_data), persona_batch_size):
            batch_start = time.time()
            log_timing(f"Starting batch {i//persona_batch_size + 1}")
            
            persona_batch = training_data[i:i+persona_batch_size]

            # Trajectory generation and scoring
            gather_start = time.time()
            train_groups = await art.gather_trajectory_groups(
                (
                    art.TrajectoryGroup(rollout(voice_agent, conversation_text, log_file=log_file) for _ in range(4))
                    for conversation_text in persona_batch
                ),
                pbar_desc="gather",
                after_each=lambda group: ruler_score_group(
                    group,
                    "openrouter/google/gemma-3n-e4b-it",
                    swallow_exceptions=True,
                ),
                max_exceptions=10
            )
            gather_end = time.time()
            log_timing(f"Trajectory gathering took: {gather_end - gather_start:.2f}s")
            
            # Training step
            train_start = time.time()
            await voice_agent.train(
                train_groups,
                config=art.TrainConfig(learning_rate=1e-5, batch_size=1),
            )
            train_end = time.time()
            log_timing(f"Training step took: {train_end - train_start:.2f}s")

            
            del train_groups
            gc.collect()
            torch.cuda.empty_cache() 
            torch.cuda.synchronize()
            
            batch_end = time.time()
            log_timing(f"Total batch time: {batch_end - batch_start:.2f}s")
            log_timing("="*50)

            await voice_agent.delete_checkpoints()


if __name__ == "__main__":
    asyncio.run(main())
