#!/usr/bin/env python3
"""
Tests VIT batch dispatcher with fake data across 8 GPUs, running one step to compare loss and grad_norm.
"""

import os
import random
from typing import Any, Dict

import torch
import torch.distributed as dist

from veomni.distributed.parallel_state import get_parallel_state, init_parallel_state
from veomni.distributed.torch_parallelize import build_parallelize_model
from veomni.models import build_foundation_model
from veomni.utils.dist_utils import all_reduce
from veomni.utils.helper import enable_high_precision_for_bf16, set_seed


def _get_input_data(model) -> Dict[str, Any]:
    """
    Generate fake multimodal data
    Each rank generates random number of images/videos to create load imbalance.
    """
    seq_len = 4096
    vocab_size = 152064

    # Get spatial_merge_size from model config (typically 2, so 4 patches merge to 1 token)
    spatial_merge_size = model.config.vision_config.spatial_merge_size

    # Random number of images and videos per rank to create load imbalance
    num_images = random.randint(0, 6)
    num_videos = random.randint(0, 2)

    # Generate base input_ids
    input_ids = torch.randint(1, vocab_size, (1, seq_len)).cuda()
    attention_mask = torch.ones(1, seq_len).cuda()

    # Init image and video masks
    image_mask = torch.zeros(1, seq_len, dtype=torch.bool).cuda()
    video_mask = torch.zeros(1, seq_len, dtype=torch.bool).cuda()

    image_grids = []
    video_grids = []

    current_pos = random.randint(0, 50)  # Start after some text tokens

    for _ in range(num_images):
        if current_pos < seq_len:
            # Create grid_thw for this image: [temporal, height, width]
            height = random.choice([4, 8, 12, 16, 20, 24])
            width = random.choice([4, 8, 12, 16, 20, 24])
            img_grid = torch.tensor([1, height, width]).cuda()
            image_grids.append(img_grid)

            # Calculate number of tokens after merging: grid_thw[0] * grid_thw[1] * grid_thw[2] // merge_size ** 2
            num_tokens = (1 * height * width) // (spatial_merge_size**2)

            # Set image placeholder tokens in input_ids
            if current_pos + num_tokens <= seq_len:
                input_ids[0, current_pos : current_pos + num_tokens] = 0
                image_mask[0, current_pos : current_pos + num_tokens] = True
                current_pos += num_tokens + random.randint(0, 50)
            else:
                break  # Not enough space for this image

    for _ in range(num_videos):
        if current_pos < seq_len:
            # Create grid_thw for this video: [temporal, height, width]
            temporal = random.choice([4, 8, 12, 16, 20, 24])
            height = random.choice([2, 4, 8, 12, 16, 20, 24])
            width = random.choice([2, 4, 8, 12, 16, 20, 24])
            vid_grid = torch.tensor([temporal, height, width]).cuda()
            video_grids.append(vid_grid)

            # Calculate number of tokens after merging: grid_thw[0] * grid_thw[1] * grid_thw[2] // merge_size ** 2
            num_tokens = (temporal * height * width) // (spatial_merge_size**2)

            # Set video placeholder tokens in input_ids
            if current_pos + num_tokens <= seq_len:
                input_ids[0, current_pos : current_pos + num_tokens] = 0
                video_mask[0, current_pos : current_pos + num_tokens] = True
                current_pos += num_tokens + random.randint(0, 50)
            else:
                break  # Not enough space for this video

    # Generate visual inputs (Before merging)
    total_image_tokens = sum(grid[0] * grid[1] * grid[2] for grid in image_grids) if image_grids else 0
    total_video_tokens = sum(grid[0] * grid[1] * grid[2] for grid in video_grids) if video_grids else 0

    # Prepare image_grid_thw and video_grid_thw for position_id_func
    image_grid_thw = torch.stack(image_grids) if image_grids else None
    video_grid_thw = torch.stack(video_grids) if video_grids else None

    # Calculate position_ids using model's position_id_func like in process_sample
    position_id_func = model.get_position_id_func()
    position_ids = position_id_func(
        input_ids=input_ids,
        image_grid_thw=image_grid_thw,
        video_grid_thw=video_grid_thw,
        attention_mask=attention_mask,
    )["position_ids"]  # (batch_size, 3, seq_len)

    batch = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
        "image_mask": image_mask,
        "video_mask": video_mask,
    }

    # Add image inputs if present
    if image_grids:
        # pixel_values shape: (total_image_tokens, hidden_size)
        # hidden_size = in_channels * patch_size * patch_size * temporal_patch_size
        # For Qwen2.5-VL: 3 * 14 * 14 * 2 = 1176
        hidden_size = 3 * 14 * 14 * 2  # Qwen2.5-VL vision hidden size
        pixel_values = torch.randn(total_image_tokens, hidden_size).cuda()

        batch["pixel_values"] = pixel_values
        batch["image_grid_thw"] = image_grid_thw

    # Add video inputs if present
    if video_grids:
        # pixel_values_videos shape: (total_video_tokens, hidden_size)
        # hidden_size = in_channels * patch_size * patch_size * temporal_patch_size
        # For Qwen2.5-VL: 3 * 14 * 14 * 2 = 1176
        hidden_size = 3 * 14 * 14 * 2  # Qwen2.5-VL vision hidden size
        pixel_values_videos = torch.randn(total_video_tokens, hidden_size).cuda()

        batch["pixel_values_videos"] = pixel_values_videos
        batch["video_grid_thw"] = video_grid_thw

    # Generate labels for loss calculation
    labels = input_ids.clone()
    # Mask out image and video positions in labels (set to -100 to ignore in loss)
    labels[image_mask] = -100
    labels[video_mask] = -100
    batch["labels"] = labels

    return batch


def run_single_step_test(
    enable_vit_load_balancing: bool = False,
    context_parallel_size: int = 1,
    vit_balance_group_size: int = 128,
    model_path: str = "hdfs://haruna/home/byte_data_seed/ssd_lq/public/veomni/models/qwen2_5vl-7b-instruct",
):
    """
    Run a single training step test with optional VIT load balancing.
    Each rank generates its own random data independently.

    Args:
        enable_vit_load_balancing: Whether to enable VIT batch dispatcher
        context_parallel_size: Context parallel size (for future extensibility)
        vit_balance_group_size: Balance group size for VIT load balancing
        model_path: Path to model weights

    Returns:
        Tuple of (loss, grad_norm)
    """
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)

    # Initialize parallel state
    init_parallel_state(dp_size=world_size, cp_size=context_parallel_size, dp_mode="fsdp1")

    # Build model
    if local_rank == 0:
        print(f"Building model with VIT load balancing: {enable_vit_load_balancing}")

    model = build_foundation_model(
        config_path=model_path,
        weights_path=model_path,
        init_device="cpu",
        attn_implementation="flash_attention_2",
        force_use_huggingface=False,
    )

    # Build parallelize model using FSDP1
    model = build_parallelize_model(
        model,
        weights_path=model_path,
        enable_full_shard=True,
        enable_mixed_precision=True,
        enable_gradient_checkpointing=False,
        init_device="cpu",
        enable_fsdp_offload=False,
        fsdp_kwargs={},
        basic_modules=model._no_split_modules,
        enable_reentrant=False,
        enable_forward_prefetch=False,
    )

    # Initialize VIT BatchDispatcher if enabled
    if enable_vit_load_balancing:
        try:
            from OmniDispatcher.batch_dispatcher import BatchDispatcher

            vit_batch_dispatcher = BatchDispatcher(
                balance_group_size=vit_balance_group_size,
                context_parallel_size=context_parallel_size,
                balance_policy="rmpad",
            )
            wrapped_model = getattr(model, "module", model)
            wrapped_model.visual.vit_batch_dispatcher = vit_batch_dispatcher
            wrapped_model.visual.enable_vit_load_balancing = True
            wrapped_model.visual.vit_balance_policy = "rmpad"
            if local_rank == 0:
                print(f"VIT BatchDispatcher initialized with group_size={vit_balance_group_size}")
        except ImportError:
            if local_rank == 0:
                print("Warning: OmniDispatcher not available, VIT load balancing disabled")
            enable_vit_load_balancing = False

    model.train()

    # Generate fake data directly on CUDA like test_ulysses.py
    fake_batch = _get_input_data(model)

    # Each rank reports its data info
    print(
        f"Rank {local_rank}: Image tokens: {fake_batch['image_mask'].sum().item()}, Video tokens: {fake_batch['video_mask'].sum().item()}"
    )
    if "pixel_values" in fake_batch:
        print(f"Rank {local_rank}: pixel_values shape: {fake_batch['pixel_values'].shape}")
    if "pixel_values_videos" in fake_batch:
        print(f"Rank {local_rank}: pixel_values_videos shape: {fake_batch['pixel_values_videos'].shape}")

    # Synchronize before forward pass
    dist.barrier()

    torch.cuda.synchronize()

    outputs = model(**fake_batch, use_cache=False)
    loss = outputs.loss

    loss.backward()

    # Calculate gradient norm using FSDP1 method
    grad_norm = model.clip_grad_norm_(max_norm=1.0).item()

    torch.cuda.synchronize()

    # Collect loss and grad_norm across all ranks
    loss_item = loss.item()
    reduced_loss, reduced_grad_norm = all_reduce((loss_item, grad_norm), group=get_parallel_state().fsdp_group)

    if local_rank == 0:
        print(f"VIT Load Balancing: {enable_vit_load_balancing}")
        print(f"Loss: {reduced_loss:.6f}")
        print(f"Grad Norm: {reduced_grad_norm:.6f}")
        print("-" * 50)

    # Cleanup model to free memory
    del model
    torch.cuda.empty_cache()

    # Synchronize before returning
    dist.barrier()

    return reduced_loss, reduced_grad_norm


def test_no_lb():
    """Test 1: Run WITHOUT VIT load balancing and save results"""
    import json

    test_config = {
        "vit_balance_group_size": 128,
        "context_parallel_size": 1,  # Can be changed for future CP testing
        "model_path": "hdfs://haruna/home/byte_data_seed/ssd_lq/public/veomni/models/qwen2_5vl-7b-instruct",
    }

    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    if local_rank == 0:
        print("=" * 60)
        print("Test 1: VIT Load Balancing DISABLED")
        print("=" * 60)
        print("Test Configuration:")
        for key, value in test_config.items():
            print(f"  {key}: {value}")
        print("=" * 60)

    try:
        loss_no_lb, grad_norm_no_lb = run_single_step_test(enable_vit_load_balancing=False, **test_config)

        # Save results to file
        results = {"vit_load_balancing": False, "loss": loss_no_lb, "grad_norm": grad_norm_no_lb, "status": "success"}

        if local_rank == 0:
            with open("/tmp/test_results_no_lb.json", "w") as f:
                json.dump(results, f, indent=2)
            print("Results saved to /tmp/test_results_no_lb.json")
            print("=" * 60)
            print("TEST 1 COMPLETED SUCCESSFULLY")
            print("=" * 60)

    except Exception as e:
        if local_rank == 0:
            print(f"Test 1 failed with error: {e}")
            results = {"vit_load_balancing": False, "error": str(e), "status": "failed"}
            with open("/tmp/test_results_no_lb.json", "w") as f:
                json.dump(results, f, indent=2)
        raise

    # Clean up distributed
    if dist.is_initialized():
        dist.destroy_process_group()


def test_with_lb():
    """Test 2: Run WITH VIT load balancing and save results"""
    import json

    test_config = {
        "vit_balance_group_size": 128,
        "context_parallel_size": 1,  # Can be changed for future CP testing
        "model_path": "hdfs://haruna/home/byte_data_seed/ssd_lq/public/veomni/models/qwen2_5vl-7b-instruct",
    }

    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    if local_rank == 0:
        print("=" * 60)
        print("Test 2: VIT Load Balancing ENABLED")
        print("=" * 60)
        print("Test Configuration:")
        for key, value in test_config.items():
            print(f"  {key}: {value}")
        print("=" * 60)

    try:
        loss_with_lb, grad_norm_with_lb = run_single_step_test(enable_vit_load_balancing=True, **test_config)

        # Save results to file
        results = {
            "vit_load_balancing": True,
            "loss": loss_with_lb,
            "grad_norm": grad_norm_with_lb,
            "status": "success",
        }

        if local_rank == 0:
            with open("/tmp/test_results_with_lb.json", "w") as f:
                json.dump(results, f, indent=2)
            print("Results saved to /tmp/test_results_with_lb.json")
            print("=" * 60)
            print("TEST 2 COMPLETED SUCCESSFULLY")
            print("=" * 60)

    except Exception as e:
        if local_rank == 0:
            print(f"Test 2 failed with error: {e}")
            results = {"vit_load_balancing": True, "error": str(e), "status": "failed"}
            with open("/tmp/test_results_with_lb.json", "w") as f:
                json.dump(results, f, indent=2)
        raise

    # Clean up distributed
    if dist.is_initialized():
        dist.destroy_process_group()


def test_compare():
    """Test 3: Compare results from both tests"""
    import json
    import os

    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    if local_rank != 0:
        return

    print("=" * 60)
    print("Test 3: COMPARING RESULTS")
    print("=" * 60)

    # Load results from both tests
    try:
        with open("/tmp/test_results_no_lb.json") as f:
            results_no_lb = json.load(f)
    except FileNotFoundError:
        print("Error: /tmp/test_results_no_lb.json not found. Run test_no_lb first.")
        return

    try:
        with open("/tmp/test_results_with_lb.json") as f:
            results_with_lb = json.load(f)
    except FileNotFoundError:
        print("Error: /tmp/test_results_with_lb.json not found. Run test_with_lb first.")
        return

    # Check if both tests succeeded
    if results_no_lb.get("status") != "success":
        print(f"Test without LB failed: {results_no_lb.get('error', 'Unknown error')}")
        return

    if results_with_lb.get("status") != "success":
        print(f"Test with LB failed: {results_with_lb.get('error', 'Unknown error')}")
        return

    # Compare results
    print(f"{'Metric':<20} {'No LB':<15} {'With LB':<15} {'Diff':<15}")
    print("-" * 65)

    loss_no_lb = results_no_lb["loss"]
    loss_with_lb = results_with_lb["loss"]
    grad_norm_no_lb = results_no_lb["grad_norm"]
    grad_norm_with_lb = results_with_lb["grad_norm"]

    loss_diff = abs(loss_with_lb - loss_no_lb)
    grad_diff = abs(grad_norm_with_lb - grad_norm_no_lb)

    print(f"{'Loss':<20} {loss_no_lb:<15.6f} {loss_with_lb:<15.6f} {loss_diff:<15.6f}")
    print(f"{'Grad Norm':<20} {grad_norm_no_lb:<15.6f} {grad_norm_with_lb:<15.6f} {grad_diff:<15.6f}")

    print("\n" + "=" * 60)
    print("COMPARISON COMPLETED")
    print("=" * 60)

    # Clean up temporary files
    try:
        os.remove("/tmp/test_results_no_lb.json")
        os.remove("/tmp/test_results_with_lb.json")
        print("Temporary result files cleaned up.")
    except FileNotFoundError:
        pass


def main():
    import sys

    # Basic sanity checks for GPU setup
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available")

    if torch.cuda.device_count() < 8:
        raise RuntimeError(f"Need 8 GPUs, only {torch.cuda.device_count()} available")

    # For distributed tests, check world size
    if len(sys.argv) >= 2 and sys.argv[1] in ["test_no_lb", "test_with_lb"]:
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        if world_size != 8:
            raise RuntimeError(f"World size {world_size}, expected 8")

    if len(sys.argv) < 2:
        print("Usage: python test_qwen2_5_vl_vit_lb.py [test_no_lb|test_with_lb|test_compare]")
        print("  test_no_lb   - Run test without VIT load balancing")
        print("  test_with_lb - Run test with VIT load balancing")
        print("  test_compare - Compare results from both tests")
        return

    test_name = sys.argv[1]

    if test_name == "test_no_lb":
        test_no_lb()
    elif test_name == "test_with_lb":
        test_with_lb()
    elif test_name == "test_compare":
        test_compare()
    else:
        print(f"Unknown test: {test_name}")
        print("Available tests: test_no_lb, test_with_lb, test_compare")


if __name__ == "__main__":
    set_seed(seed=0, full_determinism=True)
    enable_high_precision_for_bf16()
    main()

