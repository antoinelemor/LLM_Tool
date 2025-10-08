# System Resource Detection

LLM Tool includes a sophisticated system resource detection module that automatically detects and optimizes configurations based on your hardware.

## Features

- **Automatic GPU Detection**: Detects NVIDIA CUDA, Apple Silicon MPS, or CPU fallback
- **CPU Information**: Physical/logical cores, frequency, usage percentage
- **Memory Monitoring**: RAM total, available, used percentage
- **Storage Information**: Disk space available and usage
- **Smart Recommendations**: Automatic optimal settings for batch size, workers, device selection
- **Cross-Platform**: Works on macOS, Windows, and Linux

## Quick Start

### Basic Usage

```python
from llm_tool.utils import detect_resources

# Detect all system resources
resources = detect_resources()

# Access GPU information
if resources.gpu.available:
    print(f"GPU: {resources.gpu.device_type}")
    print(f"Memory: {resources.gpu.total_memory_gb} GB")

# Access CPU information
print(f"CPU Cores: {resources.cpu.physical_cores}")

# Access RAM information
print(f"RAM: {resources.memory.total_gb} GB")
```

### Get Recommendations

```python
from llm_tool.utils import detect_resources

resources = detect_resources()
recommendations = resources.get_recommendation()

print(f"Device: {recommendations['device']}")
print(f"Batch Size: {recommendations['batch_size']}")
print(f"Workers: {recommendations['num_workers']}")
print(f"FP16: {recommendations['use_fp16']}")
```

### Helper Functions

```python
from llm_tool.utils import (
    get_device_recommendation,
    get_optimal_batch_size,
    get_optimal_workers
)

# Quick access to recommendations
device = get_device_recommendation()  # Returns: "cuda", "mps", or "cpu"
batch_size = get_optimal_batch_size()  # Returns optimal batch size
workers = get_optimal_workers()        # Returns optimal number of workers
```

### Check Requirements

```python
from llm_tool.utils import check_minimum_requirements

# Check if system meets minimum requirements
meets, issues = check_minimum_requirements(
    min_ram_gb=16,
    min_storage_gb=50,
    require_gpu=True
)

if meets:
    print("System meets requirements!")
else:
    print("Issues found:")
    for issue in issues:
        print(f"  - {issue}")
```

## Display Resources

### Visual Display

```python
from llm_tool.utils import display_resources

# Display full resource information with recommendations
display_resources(show_recommendations=True, compact=False)

# Display compact view
display_resources(show_recommendations=False, compact=True)
```

### Create Tables

```python
from llm_tool.utils import (
    create_resource_table,
    create_recommendations_table,
    create_compact_resource_panel
)

# Create resource table
resource_table = create_resource_table()

# Create recommendations table
recommendations_table = create_recommendations_table()

# Create compact panel (for embedding in other displays)
panel = create_compact_resource_panel(title="System Resources")
```

### One-Line Summary

```python
from llm_tool.utils import get_resource_summary_text

summary = get_resource_summary_text()
# Returns: "MPS (96GB) | 16C | 128GB RAM"
```

## Integration with Settings

### Apply Recommendations to Settings

```python
from llm_tool.config import get_settings

settings = get_settings()

# Apply system recommendations automatically
settings.apply_system_recommendations()

# Now settings are optimized:
# - settings.training.batch_size
# - settings.data.max_workers
# - settings.local_model.device
# - settings.training.fp16
```

### Get Recommendations from Settings

```python
from llm_tool.config import get_settings

settings = get_settings()
recommendations = settings.get_system_recommendations()
```

## Recommendation Logic

### GPU Detection

1. **NVIDIA CUDA**: Detects CUDA-capable GPUs
   - Large GPU (≥16GB): batch_size=32, fp16=True
   - Medium GPU (≥8GB): batch_size=16, fp16=True
   - Small GPU (<8GB): batch_size=8, gradient_accumulation=2

2. **Apple Silicon MPS**: Detects Apple Silicon
   - batch_size=16, device="mps"

3. **CPU Fallback**: No GPU detected
   - batch_size=8, device="cpu", gradient_accumulation=2

### Worker Count

- Based on physical CPU cores
- Large systems (≥8 cores): workers = min(8, cores // 2)
- Small systems (<8 cores): workers = max(2, cores // 2)

### Memory Adjustments

- If available RAM < 8GB:
  - Batch size reduced by half
  - Workers reduced by half

## CLI Integration

The CLI automatically detects and displays system resources:

### Main Menu

When you start the CLI, resources are detected and displayed on the home page alongside available LLMs and models.

### Mode Pages

Each mode (The Annotator, The Annotator Factory, Training Arena, etc.) displays a compact resource panel showing:
- GPU type and memory
- CPU cores
- RAM available
- Storage available

## API Reference

### Classes

#### `SystemResourceDetector`

Main detector class for system resources.

```python
detector = SystemResourceDetector(cache_duration=300)
resources = detector.detect_all(force_refresh=False)
```

#### `SystemResources`

Complete system resource information container.

**Attributes:**
- `gpu`: GPUInfo
- `cpu`: CPUInfo
- `memory`: MemoryInfo
- `storage`: StorageInfo
- `system`: SystemInfo

**Methods:**
- `to_dict()`: Convert to dictionary
- `to_json()`: Convert to JSON string
- `get_recommendation()`: Get optimization recommendations

#### `GPUInfo`

GPU information container.

**Attributes:**
- `available`: bool
- `device_type`: str ("cuda", "mps", or "cpu")
- `device_count`: int
- `device_names`: List[str]
- `total_memory_gb`: float
- `available_memory_gb`: float
- `cuda_version`: Optional[str]
- `compute_capability`: Optional[str]

#### `CPUInfo`

CPU information container.

**Attributes:**
- `physical_cores`: int
- `logical_cores`: int
- `max_frequency_mhz`: float
- `current_frequency_mhz`: float
- `cpu_percent`: float
- `architecture`: str
- `processor_name`: str

#### `MemoryInfo`

Memory (RAM) information container.

**Attributes:**
- `total_gb`: float
- `available_gb`: float
- `used_gb`: float
- `percent_used`: float

#### `StorageInfo`

Storage information container.

**Attributes:**
- `total_gb`: float
- `available_gb`: float
- `used_gb`: float
- `percent_used`: float

#### `SystemInfo`

System information container.

**Attributes:**
- `os_name`: str
- `os_version`: str
- `os_release`: str
- `machine`: str
- `python_version`: str
- `detection_timestamp`: str

### Functions

#### `detect_resources(force_refresh=False) -> SystemResources`

Detect all system resources (uses cached results if available).

#### `get_device_recommendation() -> str`

Get recommended device: "cuda", "mps", or "cpu".

#### `get_optimal_workers() -> int`

Get optimal number of workers for data loading.

#### `get_optimal_batch_size() -> int`

Get optimal batch size based on available resources.

#### `check_minimum_requirements(min_ram_gb, min_storage_gb, require_gpu) -> Tuple[bool, List[str]]`

Check if system meets minimum requirements.

**Returns:**
- `bool`: True if requirements are met
- `List[str]`: List of issues if requirements are not met

## Examples

### Example 1: Basic Detection

```python
from llm_tool.utils import detect_resources

resources = detect_resources()

print(f"GPU: {resources.gpu.device_type}")
print(f"CPU Cores: {resources.cpu.physical_cores}")
print(f"RAM: {resources.memory.total_gb:.1f} GB")
print(f"Available RAM: {resources.memory.available_gb:.1f} GB")
```

### Example 2: Training Configuration

```python
from llm_tool.utils import detect_resources

resources = detect_resources()
config = resources.get_recommendation()

# Use in training
model.train(
    batch_size=config['batch_size'],
    num_workers=config['num_workers'],
    device=config['device'],
    fp16=config['use_fp16']
)
```

### Example 3: Conditional Features

```python
from llm_tool.utils import detect_resources

resources = detect_resources()

if resources.gpu.available and resources.gpu.total_memory_gb >= 16:
    # Use large model
    model = load_large_model()
elif resources.gpu.available:
    # Use medium model
    model = load_medium_model()
else:
    # Use small model for CPU
    model = load_small_model()
```

### Example 4: Pre-flight Check

```python
from llm_tool.utils import check_minimum_requirements

# Check requirements before starting
meets, issues = check_minimum_requirements(
    min_ram_gb=16,
    min_storage_gb=100,
    require_gpu=False
)

if not meets:
    print("Warning: System may not meet requirements")
    for issue in issues:
        print(f"  - {issue}")

    response = input("Continue anyway? (y/n): ")
    if response.lower() != 'y':
        exit(1)

# Proceed with operation
```

## Platform-Specific Notes

### macOS

- Detects Apple Silicon (M1, M2, M3, M4) with MPS support
- Uses unified memory architecture
- Automatically enables MPS device when available

### Windows

- Detects NVIDIA CUDA GPUs
- Falls back to CPU if no GPU available
- Uses standard memory detection

### Linux

- Detects NVIDIA CUDA GPUs
- Supports various CPU architectures
- Standard memory and storage detection

## Troubleshooting

### GPU Not Detected

**Symptom**: GPU shows as unavailable despite having one

**Possible Causes**:
1. PyTorch not installed with GPU support
2. CUDA drivers not installed (NVIDIA)
3. GPU not compatible

**Solutions**:
- Install PyTorch with CUDA support: `pip install torch --index-url https://download.pytorch.org/whl/cu118`
- Update GPU drivers
- Check GPU compatibility with PyTorch

### Memory Detection Issues

**Symptom**: Memory values show as 0 or "Unknown"

**Possible Cause**: psutil not installed

**Solution**: Install psutil: `pip install psutil`

### Incorrect Recommendations

**Symptom**: Recommendations seem too low or too high

**Solution**: Force refresh detection to get current values:
```python
resources = detect_resources(force_refresh=True)
```

## Best Practices

1. **Cache Results**: Detection can be expensive. Use cached results when possible.

2. **Check Requirements Early**: Check system requirements before starting long operations.

3. **Apply Recommendations**: Use `settings.apply_system_recommendations()` to automatically optimize.

4. **Monitor Resources**: In long-running operations, periodically refresh to check available resources.

5. **Provide Fallbacks**: Always have CPU fallback options for GPU operations.

## Performance Considerations

- **Detection Time**: First detection takes ~1-2 seconds
- **Cache Duration**: Default 5 minutes (300 seconds)
- **Refresh**: Use `force_refresh=True` only when needed

## Dependencies

- **Required**:
  - `torch`: For GPU detection
  - `platform`: For system information

- **Optional but Recommended**:
  - `psutil`: For detailed CPU and memory information
  - `rich`: For visual display

## See Also

- [Configuration Guide](./CONFIGURATION.md)
- [Training Guide](./TRAINING.md)
- [CLI Reference](./CLI.md)
