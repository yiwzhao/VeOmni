#!/bin/bash

set -x

export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export SEED_MODELS_LOGGING_LEVEL=WARN
export OMNISTORE_LOGGING_LEVEL=ERROR
export BPEX_NO_WARN_ON_UNTUNED_CASE=1
export TOKENIZERS_PARALLELISM=false
export VESCALE_SINGLE_DEVICE_RAND=0
export TF_CPP_MIN_LOG_LEVEL=2

# Fixed configuration for 8 GPUs
NNODES=1
NPROC_PER_NODE=8
NODE_RANK=0
MASTER_ADDR=127.0.0.1

echo "="*80
echo "Running Qwen2.5-VL VIT Load Balancing Tests on $NPROC_PER_NODE GPUs..."
echo "="*80

# Test 1: Run without VIT load balancing
echo "Starting Test 1: WITHOUT VIT Load Balancing..."
MASTER_PORT=12346
torchrun --nnodes=$NNODES --nproc-per-node $NPROC_PER_NODE --node-rank $NODE_RANK \
  --master-addr=$MASTER_ADDR --master-port=$MASTER_PORT tests/multimodal/test_qwen2_5_vl_vit_lb.py test_no_lb

# Wait a bit between tests for cleanup
echo "Waiting 5 seconds for cleanup between tests..."
sleep 5

# Test 2: Run with VIT load balancing
echo "Starting Test 2: WITH VIT Load Balancing..."
MASTER_PORT=12347  # Use different port to avoid conflicts
torchrun --nnodes=$NNODES --nproc-per-node $NPROC_PER_NODE --node-rank $NODE_RANK \
  --master-addr=$MASTER_ADDR --master-port=$MASTER_PORT tests/multimodal/test_qwen2_5_vl_vit_lb.py test_with_lb

# Test 3: Compare results and cleanup
echo "Starting Test 3: Comparing Results..."
python tests/multimodal/test_qwen2_5_vl_vit_lb.py test_compare

echo "="*80
echo "All tests completed successfully!"
echo "="*80