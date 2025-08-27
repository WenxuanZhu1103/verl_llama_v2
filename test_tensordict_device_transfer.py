#!/usr/bin/env python3
"""
Toy example to demonstrate TensorDict device transfer issues with late-added tensors.
This reproduces the token_level_mask_by_llm problem.
"""

import torch
from tensordict import TensorDict
from packaging import version
import tensordict

def test_tensordict_device_transfer_issue():
    """Demonstrate the issue with late-added tensors during device transfer."""
    print("=== Testing TensorDict Device Transfer Issue ===")
    print(f"TensorDict version: {tensordict.__version__}")
    
    # Check if CUDA is available
    if not torch.cuda.is_available():
        print("CUDA not available, using CPU for demonstration")
        device1, device2 = "cpu", "cpu"
    else:
        device1, device2 = "cuda", "cpu"
    
    batch_size = 4
    seq_len = 8
    
    # Step 1: Create initial TensorDict (like in generate_sequences)
    print("\n1. Creating initial TensorDict with original tensors...")
    
    # Simulate original tensors created together
    prompts = torch.randint(1000, 2000, (batch_size, seq_len), device=device1)
    responses = torch.randint(2000, 3000, (batch_size, seq_len), device=device1)
    input_ids = torch.randint(3000, 4000, (batch_size, seq_len * 2), device=device1)
    attention_mask = torch.ones((batch_size, seq_len * 2), device=device1)
    
    # Create TensorDict all at once (like in the original code)
    batch = TensorDict({
        "prompts": prompts,
        "responses": responses, 
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    }, batch_size=batch_size, device=device1)
    
    print("Original tensor values:")
    print(f"  prompts[0, :5]: {batch['prompts'][0, :5]}")
    print(f"  responses[0, :5]: {batch['responses'][0, :5]}")
    
    # Step 2: Later add a tensor (like token_level_mask_by_llm in check_sequences)
    print("\n2. Adding tensor later (simulating token_level_mask_by_llm)...")
    
    # This simulates how token_level_mask_by_llm is added later
    token_level_mask = torch.randint(0, 2, (batch_size, seq_len), device=device1, dtype=torch.float32)
    print(f"  Original token_level_mask[0, :5]: {token_level_mask[0, :5]}")
    
    # Add the tensor later (this is the problematic operation)
    batch['token_level_mask_by_llm'] = token_level_mask
    
    print("\n3. Testing device transfer WITHOUT consolidation...")
    
    # Method 1: Direct transfer (problematic)
    def transfer_without_consolidation(td):
        """Transfer without memory consolidation (problematic)."""
        return td.to(device2)
    
    # Method 2: Transfer with consolidation (fixed) 
    def transfer_with_consolidation(td):
        """Transfer with memory consolidation (fixed)."""
        if version.parse(tensordict.__version__) >= version.parse("0.5.0"):
            td = td.contiguous()
            td = td.consolidate()
        return td.to(device2)
    
    # Test without consolidation
    batch_transferred_bad = transfer_without_consolidation(batch.clone())
    
    print("After transfer WITHOUT consolidation:")
    print(f"  prompts[0, :5]: {batch_transferred_bad['prompts'][0, :5]}")
    print(f"  responses[0, :5]: {batch_transferred_bad['responses'][0, :5]}")
    print(f"  token_level_mask_by_llm[0, :5]: {batch_transferred_bad['token_level_mask_by_llm'][0, :5]}")
    
    # Check if values are corrupted
    prompts_corrupted = not torch.equal(batch['prompts'], batch_transferred_bad['prompts'].to(device1))
    responses_corrupted = not torch.equal(batch['responses'], batch_transferred_bad['responses'].to(device1))
    mask_corrupted = not torch.equal(batch['token_level_mask_by_llm'], batch_transferred_bad['token_level_mask_by_llm'].to(device1))
    
    print(f"\nCorruption check (WITHOUT consolidation):")
    print(f"  prompts corrupted: {prompts_corrupted}")
    print(f"  responses corrupted: {responses_corrupted}")
    print(f"  token_level_mask_by_llm corrupted: {mask_corrupted}")
    
    if mask_corrupted:
        print("  ❌ BUG REPRODUCED: token_level_mask_by_llm values changed!")
        # Show what values it got instead
        original_mask = batch['token_level_mask_by_llm'][0, :5]
        transferred_mask = batch_transferred_bad['token_level_mask_by_llm'][0, :5].to(device1)
        print(f"    Original: {original_mask}")
        print(f"    After transfer: {transferred_mask}")
        
        # Check if it matches other tensors (common corruption pattern)
        if torch.equal(transferred_mask, batch['prompts'][0, :5].float()):
            print("    ❌ token_level_mask_by_llm got prompts values!")
        elif torch.equal(transferred_mask, batch['responses'][0, :5].float()):
            print("    ❌ token_level_mask_by_llm got responses values!")
    
    print("\n4. Testing device transfer WITH consolidation (FIX)...")
    
    # Test with consolidation (our fix)
    batch_transferred_good = transfer_with_consolidation(batch.clone())
    
    print("After transfer WITH consolidation:")
    print(f"  prompts[0, :5]: {batch_transferred_good['prompts'][0, :5]}")
    print(f"  responses[0, :5]: {batch_transferred_good['responses'][0, :5]}")
    print(f"  token_level_mask_by_llm[0, :5]: {batch_transferred_good['token_level_mask_by_llm'][0, :5]}")
    
    # Check if values are preserved
    prompts_preserved = torch.equal(batch['prompts'], batch_transferred_good['prompts'].to(device1))
    responses_preserved = torch.equal(batch['responses'], batch_transferred_good['responses'].to(device1))
    mask_preserved = torch.equal(batch['token_level_mask_by_llm'], batch_transferred_good['token_level_mask_by_llm'].to(device1))
    
    print(f"\nPreservation check (WITH consolidation):")
    print(f"  prompts preserved: {prompts_preserved}")
    print(f"  responses preserved: {responses_preserved}")
    print(f"  token_level_mask_by_llm preserved: {mask_preserved}")
    
    if mask_preserved:
        print("  ✅ FIX VERIFIED: All values preserved correctly!")
    else:
        print("  ❌ FIX FAILED: Values still corrupted")

def test_memory_layout_difference():
    """Show the difference in memory layout between original and late-added tensors."""
    print("\n=== Memory Layout Analysis ===")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 3
    seq_len = 5
    
    # Create TensorDict with original tensors
    batch = TensorDict({
        "tensor_a": torch.randint(100, 200, (batch_size, seq_len), device=device),
        "tensor_b": torch.randint(200, 300, (batch_size, seq_len), device=device),
    }, batch_size=batch_size, device=device)
    
    print("Before adding late tensor:")
    print(f"  TensorDict is consolidated: {batch.is_consolidated() if hasattr(batch, 'is_consolidated') else 'N/A'}")
    print(f"  TensorDict is contiguous: {batch.is_contiguous()}")
    
    # Add tensor later
    batch['late_tensor'] = torch.randint(300, 400, (batch_size, seq_len), device=device)
    
    print("\nAfter adding late tensor:")
    print(f"  TensorDict is consolidated: {batch.is_consolidated() if hasattr(batch, 'is_consolidated') else 'N/A'}")
    print(f"  TensorDict is contiguous: {batch.is_contiguous()}")
    
    # Show memory addresses (if possible)
    try:
        print(f"\nMemory info:")
        print(f"  tensor_a data_ptr: {batch['tensor_a'].data_ptr()}")
        print(f"  tensor_b data_ptr: {batch['tensor_b'].data_ptr()}")
        print(f"  late_tensor data_ptr: {batch['late_tensor'].data_ptr()}")
        
        # Check if they're in contiguous memory
        diff_ab = abs(batch['tensor_b'].data_ptr() - batch['tensor_a'].data_ptr())
        diff_bc = abs(batch['late_tensor'].data_ptr() - batch['tensor_b'].data_ptr())
        print(f"  Memory gap A->B: {diff_ab}")
        print(f"  Memory gap B->late: {diff_bc}")
        
        if diff_bc > diff_ab * 10:  # Heuristic check
            print("  ❌ Late tensor likely in separate memory region!")
        else:
            print("  ✅ Tensors appear to be in contiguous memory")
            
    except Exception as e:
        print(f"  Memory analysis failed: {e}")

if __name__ == "__main__":
    # Run the tests
    test_tensordict_device_transfer_issue()
    test_memory_layout_difference()
    
    print("\n=== Summary ===")
    print("This demonstrates why token_level_mask_by_llm (late-added tensor) is more")
    print("susceptible to corruption during device transfer compared to tensors") 
    print("created together in the original TensorDict.")
    print("\nThe fix (contiguous + consolidate before transfer) ensures all tensors")
    print("have proper memory layout before device transfer.") 